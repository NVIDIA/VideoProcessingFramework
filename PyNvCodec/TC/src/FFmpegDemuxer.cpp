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

double FFmpegDemuxer::GetFramerate() const { return framerate; }

double FFmpegDemuxer::GetTimebase() const { return timebase; }

uint32_t FFmpegDemuxer::GetVideoStreamIndex() const { return videoStream; }

AVPixelFormat FFmpegDemuxer::GetPixelFormat() const { return eChromaFormat; }

bool FFmpegDemuxer::Demux(uint8_t *&pVideo, size_t &rVideoBytes,
                          PacketData &rCtx, uint8_t **ppSEI,
                          size_t *pSEIBytes) {
  if (!fmtc) {
    return false;
  }

  if (pkt.data) {
    av_packet_unref(&pkt);
  }

  if (!annexbBytes.empty()) {
    annexbBytes.clear();
  }

  if (!seiBytes.empty()) {
    seiBytes.clear();
  }

  auto appendBytes = [](vector<uint8_t> &elementaryBytes, AVPacket &avPacket,
                        AVPacket &avPacketBsf, AVBSFContext *pAvbsfContext,
                        int streamId, bool isFilteringNeeded, ...) {
    if (avPacket.stream_index != streamId) {
      return;
    }

    if (isFilteringNeeded) {
      if (avPacketBsf.data) {
        av_packet_unref(&avPacketBsf);
      }

      av_bsf_send_packet(pAvbsfContext, &avPacket);
      av_bsf_receive_packet(pAvbsfContext, &avPacketBsf);

      if (avPacketBsf.data && avPacketBsf.size) {
        elementaryBytes.insert(elementaryBytes.end(), avPacketBsf.data,
                               avPacketBsf.data + avPacketBsf.size);
      }
    } else if (avPacket.data && avPacket.size) {
      elementaryBytes.insert(elementaryBytes.end(), avPacket.data,
                             avPacket.data + avPacket.size);
    }
  };

  int ret = 0;
  bool isDone = false, gotVideo = false;

  while (!isDone) {
    ret = av_read_frame(fmtc, &pkt);
    gotVideo = (pkt.stream_index == videoStream);
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
      auto pCopyPacket = av_packet_clone(&pkt);
      appendBytes(seiBytes, *pCopyPacket, pktSei, bsfc_sei, videoStream, true);
      av_packet_free(&pCopyPacket);
    }

    /* Unref non-desired packets as we don't support them yet;
     */
    if (pkt.stream_index != videoStream) {
      av_packet_unref(&pkt);
      continue;
    }
  }

  if (ret < 0) {
    cerr << "Failed to read frame: " << AvErrorToString(ret) << endl;
    return false;
  }

  appendBytes(annexbBytes, pkt, pktAnnexB, bsfc_annexb, videoStream,
              is_mp4H264 || is_mp4HEVC);

  pVideo = annexbBytes.data();
  rVideoBytes = annexbBytes.size();

  // Save packet timestamp & duration;
  rCtx.pts = pktAnnexB.pts;
  rCtx.dts = pktAnnexB.dts;
  rCtx.pos = pktAnnexB.pos;
  rCtx.duration = pktAnnexB.duration;

  if (pSEIBytes && ppSEI && !seiBytes.empty()) {
    *ppSEI = seiBytes.data();
    *pSEIBytes = seiBytes.size();
  }

  return true;
}

bool FFmpegDemuxer::Seek(SeekContext *p_ctx) {
  // Seek direction:
  bool const seek_b = pktAnnexB.dts > p_ctx->seek_frame * pktAnnexB.duration;
  // Timestamp in seconds;
  auto const ts_sec = (double)p_ctx->seek_frame / GetFramerate();
  // Timestamp in time base units;
  auto const ts_tbu = (int64_t)(ts_sec * AV_TIME_BASE);
  // Rescaled timestamp;
  AVRational factor;
  factor.num = 1;
  factor.den = AV_TIME_BASE;
  auto const ts_rsc =
      av_rescale_q(ts_tbu, factor, fmtc->streams[videoStream]->time_base);

  auto ret = av_seek_frame(fmtc, GetVideoStreamIndex(), ts_rsc,
                           seek_b ? AVSEEK_FLAG_BACKWARD : 0);

  if (ret < 0) {
    throw runtime_error("Error seeking for frame: " + AvErrorToString(ret));
  } else {
    avio_flush(fmtc->pb);
    avformat_flush(fmtc);
  }

  return true;
}

int FFmpegDemuxer::ReadPacket(void *opaque, uint8_t *pBuf, int nBuf) {
  return ((DataProvider *)opaque)->GetData(pBuf, nBuf);
}

AVCodecID FFmpegDemuxer::GetVideoCodec() const { return eVideoCodec; }

FFmpegDemuxer::~FFmpegDemuxer() {
  if (pkt.data) {
    av_packet_unref(&pkt);
  }
  if (pktAnnexB.data) {
    av_packet_unref(&pktAnnexB);
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
  pkt = {};
  pktAnnexB = {};

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

  is_mp4H264 = (eVideoCodec == AV_CODEC_ID_H264);
  is_mp4HEVC = (eVideoCodec == AV_CODEC_ID_HEVC);
  av_init_packet(&pkt);
  pkt.data = nullptr;
  pkt.size = 0;
  av_init_packet(&pktAnnexB);
  pktAnnexB.data = nullptr;
  pktAnnexB.size = 0;
  av_init_packet(&pktSei);
  pktSei.data = nullptr;
  pktSei.size = 0;

  // Initialize Annex.B BSF;
  const string bfs_name = is_mp4H264
                              ? "h264_mp4toannexb"
                              : is_mp4HEVC ? "hevc_mp4toannexb" : "unknown";
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

  // SEI extraction filter has lazy init as this feature is optional;
  bsfc_sei = nullptr;
}
