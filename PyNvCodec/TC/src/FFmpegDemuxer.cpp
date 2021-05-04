/*
 * Copyright 2019 NVIDIA Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "FFmpegDemuxer.h"
#include "NvCodecUtils.h"
#include "libavutil/avstring.h"
#include "libavutil/avutil.h"
#include <iostream>
#include <limits>
#include <sstream>

using namespace std;

static string AvErrorToString(int av_error_code) {
  const auto buf_size = 1024U;
  char *err_string = (char *)calloc(buf_size, sizeof(*err_string));
  if (!err_string) {
    return string();
  }

  if (0 != av_strerror(av_error_code, err_string, buf_size - 1)) {
    free(err_string);
    stringstream ss;
    ss << "Unknown error with code " << av_error_code;
    return ss.str();
  }

  string str(err_string);
  free(err_string);
  return str;
}

class DataProvider {
public:
  virtual ~DataProvider() = default;
  virtual int GetData(uint8_t *pBuf, int nBuf) = 0;
};

FFmpegDemuxer::FFmpegDemuxer(const char *szFilePath,
                             const map<string, string> &ffmpeg_options)
    : FFmpegDemuxer(CreateFormatContext(szFilePath, ffmpeg_options)) {}

FFmpegDemuxer::FFmpegDemuxer(DataProvider *pDataProvider,
                             const map<string, string> &ffmpeg_options)
    : FFmpegDemuxer(CreateFormatContext(pDataProvider, ffmpeg_options)) {
  avioc = fmtc->pb;
}

uint32_t FFmpegDemuxer::GetWidth() const { return width; }

uint32_t FFmpegDemuxer::GetHeight() const { return height; }

uint32_t FFmpegDemuxer::GetGopSize() const { return gop_size; }

uint32_t FFmpegDemuxer::GetNumFrames() const {return nb_frames;}

double FFmpegDemuxer::GetFramerate() const { return framerate; }

double FFmpegDemuxer::GetTimebase() const { return timebase; }

uint32_t FFmpegDemuxer::GetVideoStreamIndex() const { return videoStream; }

AVPixelFormat FFmpegDemuxer::GetPixelFormat() const { return eChromaFormat; }

bool FFmpegDemuxer::Demux(uint8_t *&pVideo, size_t &rVideoBytes,
                          PacketData &pktData, uint8_t **ppSEI,
                          size_t *pSEIBytes) {
  if (!fmtc) {
    return false;
  }

  if (pktSrc.data) {
    av_packet_unref(&pktSrc);
  }

  if (!annexbBytes.empty()) {
    annexbBytes.clear();
  }

  if (!seiBytes.empty()) {
    seiBytes.clear();
  }

  auto appendBytes = [](vector<uint8_t> &elementaryBytes, AVPacket &avPacket,
                        AVPacket &avPacketOut, AVBSFContext *pAvbsfContext,
                        int streamId, bool isFilteringNeeded) {
    if (avPacket.stream_index != streamId) {
      return;
    }

    if (isFilteringNeeded) {
      if (avPacketOut.data) {
        av_packet_unref(&avPacketOut);
      }

      av_bsf_send_packet(pAvbsfContext, &avPacket);
      av_bsf_receive_packet(pAvbsfContext, &avPacketOut);

      if (avPacketOut.data && avPacketOut.size) {
        elementaryBytes.insert(elementaryBytes.end(), avPacketOut.data,
                               avPacketOut.data + avPacketOut.size);
      }
    } else if (avPacket.data && avPacket.size) {
      elementaryBytes.insert(elementaryBytes.end(), avPacket.data,
                             avPacket.data + avPacket.size);
    }
  };

  int ret = 0;
  bool isDone = false, gotVideo = false;

  while (!isDone) {
    ret = av_read_frame(fmtc, &pktSrc);
    gotVideo = (pktSrc.stream_index == videoStream);
    isDone = (ret < 0) || gotVideo;

    if (pSEIBytes && ppSEI) {
      // Bitstream filter lazy init;
      // We don't do this in constructor as user may not be needing SEI
      // extraction at all;
      if (!bsfc_sei) {
        cout << "Initializing SEI filter;" << endl;

        // SEI has NAL type 6 for H.264 and NAL type 39 & 40 for H.265;
        const string sei_filter =
            is_mp4H264
                ? "filter_units=pass_types=6"
                : is_mp4HEVC ? "filter_units=pass_types=39-40" : "unknown";
        ret = av_bsf_list_parse_str(sei_filter.c_str(), &bsfc_sei);
        if (0 > ret) {
          throw runtime_error("Error initializing " + sei_filter +
                              " bitstream filter: " + AvErrorToString(ret));
        }

        ret = avcodec_parameters_copy(bsfc_sei->par_in,
                                      fmtc->streams[videoStream]->codecpar);
        if (0 != ret) {
          throw runtime_error("Error copying codec parameters: " +
                              AvErrorToString(ret));
        }

        ret = av_bsf_init(bsfc_sei);
        if (0 != ret) {
          throw runtime_error("Error initializing " + sei_filter +
                              " bitstream filter: " + AvErrorToString(ret));
        }
      }

      // Extract SEI NAL units from packet;
      auto pCopyPacket = av_packet_clone(&pktSrc);
      appendBytes(seiBytes, *pCopyPacket, pktSei, bsfc_sei, videoStream, true);
      av_packet_free(&pCopyPacket);
    }

    /* Unref non-desired packets as we don't support them yet;
     */
    if (pktSrc.stream_index != videoStream) {
      av_packet_unref(&pktSrc);
      continue;
    }
  }

  if (ret < 0) {
    cerr << "Failed to read frame: " << AvErrorToString(ret) << endl;
    return false;
  }

  const bool bsf_needed = is_mp4H264 || is_mp4HEVC;
  appendBytes(annexbBytes, pktSrc, pktDst, bsfc_annexb, videoStream,
              bsf_needed);

  pVideo = annexbBytes.data();
  rVideoBytes = annexbBytes.size();

  /* Save packet props to PacketData, decoder will use it later.
   * If no BSF filters were applied, copy input packet props.
   */
  if (!bsf_needed) {
    av_packet_copy_props(&pktDst, &pktSrc);
  }

  last_packet_data.pts = pktDst.pts;
  last_packet_data.dts = pktDst.dts;
  last_packet_data.pos = pktDst.pos;
  last_packet_data.duration = pktDst.duration;

  pktData = last_packet_data;

  if (pSEIBytes && ppSEI && !seiBytes.empty()) {
    *ppSEI = seiBytes.data();
    *pSEIBytes = seiBytes.size();
  }

  return true;
}

void FFmpegDemuxer::Flush() {
  avio_flush(fmtc->pb);
  avformat_flush(fmtc);
}

bool FFmpegDemuxer::Seek(SeekContext &seekCtx, uint8_t *&pVideo,
                         size_t &rVideoBytes, PacketData &pktData,
                         uint8_t **ppSEI, size_t *pSEIBytes) {
  if (!is_seekable) {
    cerr << "Seek isn't supported for this input." << endl;
    return false;
  }

  // Convert frame number to timestamp;
  auto frame_ts = [&](int64_t frame_num) {
    auto const ts_sec = (double)seekCtx.seek_frame / GetFramerate();
    auto const ts_tbu = (int64_t)(ts_sec * AV_TIME_BASE);
    AVRational factor;
    factor.num = 1;
    factor.den = AV_TIME_BASE;
    return av_rescale_q(ts_tbu, factor, fmtc->streams[videoStream]->time_base);
  };

  // Seek for single frame;
  auto seek_frame = [&](SeekContext const &seek_ctx, int flags) {
    bool const seek_b =
        last_packet_data.dts > seek_ctx.seek_frame * pktDst.duration;
    auto ret = av_seek_frame(fmtc, GetVideoStreamIndex(),
                             frame_ts(seek_ctx.seek_frame),
                             seek_b ? AVSEEK_FLAG_BACKWARD | flags : flags);
    if (ret < 0) {
      throw runtime_error("Error seeking for frame: " + AvErrorToString(ret));
    }

    return;
  };

  // Check if frame satisfies seek conditions;
  auto is_seek_done = [&](PacketData &pkt_data, SeekContext const &seek_ctx) {
    auto const target_ts = frame_ts(seek_ctx.seek_frame);
    if (pkt_data.dts == target_ts) {
      return 0;
    } else if (pkt_data.dts > target_ts) {
      return 1;
    } else {
      return -1;
    };
  };
  
  // This will seek for exact frame number;
  // Note that decoder may not be able to decode such frame;
  auto seek_for_exact_frame = [&](PacketData &pkt_data,
                                  SeekContext &seek_ctx) {
    // Repetititive seek until seek condition is satisfied;
    SeekContext tmp_ctx(seek_ctx.seek_frame);
    seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);

    int condition = 0;
    do {
      Demux(pVideo, rVideoBytes, pkt_data, ppSEI, pSEIBytes);
      condition = is_seek_done(pkt_data, seek_ctx);

      // We've gone too far and need to seek backwards;
      if (condition > 0) {
        tmp_ctx.seek_frame--;
        seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);
      }
      // Need to read more frames until we reach requested number;
      else if (condition < 0) {
        continue;
      }
    } while (0 != condition);

    seek_ctx.out_frame_pts = pkt_data.pts;
    seek_ctx.out_frame_duration = pkt_data.duration;
  };

  // Seek for closest key frame in the past;
  auto seek_for_prev_key_frame = [&](PacketData &pkt_data,
                                    SeekContext &seek_ctx) {
    // Repetititive seek until seek condition is satisfied;
    SeekContext tmp_ctx(seek_ctx.seek_frame);
    seek_frame(tmp_ctx, AVSEEK_FLAG_BACKWARD);

    Demux(pVideo, rVideoBytes, pkt_data, ppSEI, pSEIBytes);
    seek_ctx.out_frame_pts = pkt_data.pts;
    seek_ctx.out_frame_duration = pkt_data.duration;
  };

  switch (seekCtx.mode) {
  case EXACT_FRAME:
    seek_for_exact_frame(pktData, seekCtx);
    break;
  case PREV_KEY_FRAME:
    seek_for_prev_key_frame(pktData, seekCtx);
    break;
  default:
    throw runtime_error("Unsupported seek mode");
    break;
  }

  return true;
}

int FFmpegDemuxer::ReadPacket(void *opaque, uint8_t *pBuf, int nBuf) {
  return ((DataProvider *)opaque)->GetData(pBuf, nBuf);
}

AVCodecID FFmpegDemuxer::GetVideoCodec() const { return eVideoCodec; }

FFmpegDemuxer::~FFmpegDemuxer() {
  if (pktSrc.data) {
    av_packet_unref(&pktSrc);
  }
  if (pktDst.data) {
    av_packet_unref(&pktDst);
  }

  if (bsfc_annexb) {
    av_bsf_free(&bsfc_annexb);
  }

  if (bsfc_annexb) {
    av_bsf_free(&bsfc_sei);
  }

  avformat_close_input(&fmtc);

  if (avioc) {
    av_freep(&avioc->buffer);
    av_freep(&avioc);
  }
}

AVFormatContext *
FFmpegDemuxer::CreateFormatContext(DataProvider *pDataProvider,
                                   const map<string, string> &ffmpeg_options) {
  AVFormatContext *ctx = avformat_alloc_context();
  if (!ctx) {
    cerr << "Can't allocate AVFormatContext at " << __FILE__ << " " << __LINE__;
    return nullptr;
  }

  uint8_t *avioc_buffer = nullptr;
  int avioc_buffer_size = 8 * 1024 * 1024;
  avioc_buffer = (uint8_t *)av_malloc(avioc_buffer_size);
  if (!avioc_buffer) {
    cerr << "Can't allocate avioc_buffer at " << __FILE__ << " " << __LINE__;
    return nullptr;
  }
  avioc = avio_alloc_context(avioc_buffer, avioc_buffer_size, 0, pDataProvider,
                             &ReadPacket, nullptr, nullptr);

  if (!avioc) {
    cerr << "Can't allocate AVIOContext at " << __FILE__ << " " << __LINE__;
    return nullptr;
  }
  ctx->pb = avioc;

  // Set up format context options;
  AVDictionary *options = NULL;
  for (auto &pair : ffmpeg_options) {
    auto err =
        av_dict_set(&options, pair.first.c_str(), pair.second.c_str(), 0);
    if (err < 0) {
      cerr << "Can't set up dictionary option: " << pair.first << " "
           << pair.second << ": " << AvErrorToString(err) << "\n";
      return nullptr;
    }
  }

  auto err = avformat_open_input(&ctx, nullptr, nullptr, &options);
  if (0 != err) {
    cerr << "Can't open input. Error message: " << AvErrorToString(err);
    return nullptr;
  }

  return ctx;
}

AVFormatContext *
FFmpegDemuxer::CreateFormatContext(const char *szFilePath,
                                   const map<string, string> &ffmpeg_options) {
  avformat_network_init();

  // Set up format context options;
  AVDictionary *options = NULL;
  for (auto &pair : ffmpeg_options) {
    cout << pair.first << ": " << pair.second << endl;
    auto err =
        av_dict_set(&options, pair.first.c_str(), pair.second.c_str(), 0);
    if (err < 0) {
      cerr << "Can't set up dictionary option: " << pair.first << " "
           << pair.second << ": " << AvErrorToString(err) << "\n";
      return nullptr;
    }
  }

  AVFormatContext *ctx = nullptr;
  av_register_all();
  auto err = avformat_open_input(&ctx, szFilePath, nullptr, &options);
  if (err < 0) {
    cerr << "Can't open " << szFilePath << ": " << AvErrorToString(err) << "\n";
    return nullptr;
  }

  return ctx;
}

FFmpegDemuxer::FFmpegDemuxer(AVFormatContext *fmtcx) : fmtc(fmtcx) {
  pktSrc = {};
  pktDst = {};

  memset(&last_packet_data, 0, sizeof(last_packet_data));

  if (!fmtc) {
    stringstream ss;
    ss << __FUNCTION__ << ": no AVFormatContext provided." << endl;
    throw invalid_argument(ss.str());
  }

  auto ret = avformat_find_stream_info(fmtc, nullptr);
  if (0 != ret) {
    stringstream ss;
    ss << __FUNCTION__ << ": can't find stream info;" << AvErrorToString(ret)
       << endl;
    throw runtime_error(ss.str());
  }

  videoStream =
      av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (videoStream < 0) {
    stringstream ss;
    ss << __FUNCTION__ << ": can't find video stream in input file." << endl;
    throw runtime_error(ss.str());
  }

  gop_size = fmtc->streams[videoStream]->codec->gop_size;
  eVideoCodec = fmtc->streams[videoStream]->codecpar->codec_id;
  width = fmtc->streams[videoStream]->codecpar->width;
  height = fmtc->streams[videoStream]->codecpar->height;
  framerate = (double)fmtc->streams[videoStream]->r_frame_rate.num /
              (double)fmtc->streams[videoStream]->r_frame_rate.den;
  timebase = (double)fmtc->streams[videoStream]->time_base.num /
             (double)fmtc->streams[videoStream]->time_base.den;
  eChromaFormat = (AVPixelFormat)fmtc->streams[videoStream]->codecpar->format;
  nb_frames = fmtc->streams[videoStream]->nb_frames;

  is_mp4H264 = (eVideoCodec == AV_CODEC_ID_H264);
  is_mp4HEVC = (eVideoCodec == AV_CODEC_ID_HEVC);
  is_VP9 = (eVideoCodec == AV_CODEC_ID_VP9);
  av_init_packet(&pktSrc);
  pktSrc.data = nullptr;
  pktSrc.size = 0;
  av_init_packet(&pktDst);
  pktDst.data = nullptr;
  pktDst.size = 0;
  av_init_packet(&pktSei);
  pktSei.data = nullptr;
  pktSei.size = 0;

  // Initialize Annex.B BSF;
  const string bfs_name =
      is_mp4H264 ? "h264_mp4toannexb"
                 : is_mp4HEVC ? "hevc_mp4toannexb" : is_VP9 ? string() : "unknown";

  if (!bfs_name.empty()) {
    const AVBitStreamFilter *toAnnexB = av_bsf_get_by_name(bfs_name.c_str());
    if (!toAnnexB) {
      throw runtime_error("can't get " + bfs_name + " filter by name");
    }
    ret = av_bsf_alloc(toAnnexB, &bsfc_annexb);
    if (0 != ret) {
      throw runtime_error("Error allocating " + bfs_name +
                          " filter: " + AvErrorToString(ret));
    }

    ret = avcodec_parameters_copy(bsfc_annexb->par_in,
                                  fmtc->streams[videoStream]->codecpar);
    if (0 != ret) {
      throw runtime_error("Error copying codec parameters: " +
                          AvErrorToString(ret));
    }

    ret = av_bsf_init(bsfc_annexb);
    if (0 != ret) {
      throw runtime_error("Error initializing " + bfs_name +
                          " bitstream filter: " + AvErrorToString(ret));
    }
  }

  // SEI extraction filter has lazy init as this feature is optional;
  bsfc_sei = nullptr;

  /* Some inputs doesn't allow seek functionality.
   * Check this ahead of time. */
  is_seekable = fmtc->iformat->read_seek || fmtc->iformat->read_seek2;
}
