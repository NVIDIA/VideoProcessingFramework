/*
 * Copyright 2020 NVIDIA Corporation
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

#include "Tasks.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/motion_vector.h>
}

using namespace VPF;
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

namespace VPF {

enum DECODE_STATUS { DEC_SUCCESS, DEC_ERROR, DEC_MORE, DEC_EOS };

struct FfmpegDecodeFrame_Impl {
  AVFormatContext *fmt_ctx = nullptr;
  AVCodecContext *avctx = nullptr;
  AVStream *video_stream = nullptr;
  AVFrame *frame = nullptr;
  AVCodec *p_codec = nullptr;
  AVPacket pktSrc = {0};

  Buffer *dec_frame = nullptr;
  map<AVFrameSideDataType, Buffer *> side_data;

  int video_stream_idx = -1;
  bool end_encode = false;

  FfmpegDecodeFrame_Impl(const char *URL, AVDictionary *pOptions) {
    av_log_set_level(AV_LOG_PANIC); //Reduce ffmpeg logs
    av_register_all();

    auto res = avformat_open_input(&fmt_ctx, URL, NULL, &pOptions);
    if (res < 0) {
      stringstream ss;
      ss << "Could not open source file" << URL << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      throw runtime_error(ss.str());
    }

    res = avformat_find_stream_info(fmt_ctx, NULL);
    if (res < 0) {
      stringstream ss;
      ss << "Could not find stream information" << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      throw runtime_error(ss.str());
    }

    res = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (res < 0) {
      stringstream ss;
      ss << "Could not find " << av_get_media_type_string(AVMEDIA_TYPE_VIDEO)
         << " stream in file " << URL << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      throw runtime_error(ss.str());
    }

    video_stream_idx = res;
    video_stream = fmt_ctx->streams[video_stream_idx];

    if (!video_stream) {
      cerr << "Could not find video stream in the input, aborting" << endl;
    }

    avctx = fmt_ctx->streams[video_stream_idx]->codec;
    if (!avctx) {
      stringstream ss;
      ss << "Failed to use codec context from AVFormatContext" << endl;
      throw runtime_error(ss.str());
    }

    p_codec = avcodec_find_decoder(avctx->codec_id);
    if (!p_codec) {
      stringstream ss;
      ss << "Failed to find codec from AVCodecContext" << endl;
      throw runtime_error(ss.str());
    }

    res = avcodec_open2(avctx, p_codec, &pOptions);
    if (res < 0) {
      stringstream ss;
      ss << "Failed to open codec "
         << av_get_media_type_string(AVMEDIA_TYPE_VIDEO) << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      throw runtime_error(ss.str());
    }

    //av_dump_format(fmt_ctx, 0, URL, 0);

    frame = av_frame_alloc();
    if (!frame) {
      cerr << "Could not allocate frame" << endl;
    }
  }

  bool SaveYUV420(AVFrame *pframe) {
    // Detect frame size & allocate memory if necessary;
    size_t size = frame->width * frame->height * 3 / 2;

    if (!dec_frame) {
      dec_frame = Buffer::MakeOwnMem(size);
    } else if (size != dec_frame->GetRawMemSize()) {
      delete dec_frame;
      dec_frame = Buffer::MakeOwnMem(size);
    }

    // Copy pixels;
    auto plane = 0U;
    auto *dst = dec_frame->GetDataAs<uint8_t>();

    for (plane = 0; plane < 3; plane++) {
      auto *src = frame->data[plane];
      auto width = (0 == plane) ? frame->width : frame->width / 2;
      auto height = (0 == plane) ? frame->height : frame->height / 2;

      for (int i = 0; i < height; i++) {
        memcpy(dst, src, width);
        dst += width;
        src += frame->linesize[plane];
      }
    }

    return true;
  }

  bool DecodeSingleFrame() {
    if (end_encode) {
      return false;
    }

    // Send packets to decoder until it outputs frame;
    do {
      // Read packets from stream until we find a video packet;
      do {
        auto ret = av_read_frame(fmt_ctx, &pktSrc);
        if (ret < 0) {
          // Flush decoder;
          end_encode = true;
          return DecodeSinglePacket(nullptr);
        }
      } while (pktSrc.stream_index != video_stream_idx);

      auto status = DecodeSinglePacket(&pktSrc);

      switch (status) {
      case DEC_SUCCESS:
        return true;
      case DEC_ERROR:
        return false;
      case DEC_EOS:
        return false;
      case DEC_MORE:
        continue;
      }
    } while (true);

    return true;
  }

  bool SaveVideoFrame(AVFrame *frame) {
    // Only YUV420P is supported so far;
    if (AV_PIX_FMT_YUV420P != frame->format) {
      return false;
    }

    return SaveYUV420(frame);
  }

  void SaveMotionVectors(AVFrame *frame) {
    AVFrameSideData *sd =
        av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);

    if (sd) {
      auto it = side_data.find(AV_FRAME_DATA_MOTION_VECTORS);
      if (it == side_data.end()) {
        // Add entry if not found (usually upon first call);
        side_data[AV_FRAME_DATA_MOTION_VECTORS] = Buffer::MakeOwnMem(sd->size);
        it = side_data.find(AV_FRAME_DATA_MOTION_VECTORS);
        memcpy(it->second->GetRawMemPtr(), sd->data, sd->size);
      } else if (it->second->GetRawMemSize() != sd->size) {
        // Update entry size if changed (e. g. on video resolution change);
        it->second->Update(sd->size, sd->data);
      }
    }
  }

  bool SaveSideData(AVFrame *frame) {
    SaveMotionVectors(frame);
    return true;
  }

  DECODE_STATUS DecodeSinglePacket(const AVPacket *pktSrc) {
    auto res = avcodec_send_packet(avctx, pktSrc);
    if (res < 0) {
      cerr << "Error while sending a packet to the decoder" << endl;
      cerr << "Error description: " << AvErrorToString(res) << endl;
      return DEC_ERROR;
    }

    while (res >= 0) {
      res = avcodec_receive_frame(avctx, frame);
      if (res == AVERROR_EOF) {
        cerr << "Input file is over" << endl;
        return DEC_EOS;
      } else if (res == AVERROR(EAGAIN)) {
        return DEC_MORE;
      } else if (res < 0) {
        cerr << "Error while receiving a frame from the decoder" << endl;
        cerr << "Error description: " << AvErrorToString(res) << endl;
        return DEC_ERROR;
      }

      SaveVideoFrame(frame);
      SaveSideData(frame);
      return DEC_SUCCESS;
    }

    return DEC_SUCCESS;
  }

  ~FfmpegDecodeFrame_Impl() {
    avformat_close_input(&fmt_ctx);
    av_frame_free(&frame);

    for (auto &output : side_data) {
      if (output.second) {
        delete output.second;
        output.second = nullptr;
      }
    }

    if (dec_frame) {
      delete dec_frame;
    }
  }
};
} // namespace VPF

TaskExecStatus FfmpegDecodeFrame::Run() {
  ClearOutputs();

  if (pImpl->DecodeSingleFrame()) {
    SetOutput((Token *)pImpl->dec_frame, 0U);
    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }

  return TaskExecStatus::TASK_EXEC_FAIL;
}

TaskExecStatus FfmpegDecodeFrame::GetSideData(AVFrameSideDataType data_type) {
  SetOutput(nullptr, 1U);
  auto it = pImpl->side_data.find(data_type);
  if (it != pImpl->side_data.end()) {
    SetOutput((Token *)it->second, 1U);
    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }

  return TaskExecStatus::TASK_EXEC_FAIL;
}

FfmpegDecodeFrame *FfmpegDecodeFrame::Make(const char *URL,
                                           NvDecoderClInterface &cli_iface) {
  return new FfmpegDecodeFrame(URL, cli_iface);
}

FfmpegDecodeFrame::FfmpegDecodeFrame(const char *URL,
                                     NvDecoderClInterface &cli_iface)
    : Task("FfmpegDecodeFrame", FfmpegDecodeFrame::num_inputs,
           FfmpegDecodeFrame::num_outputs) {
  pImpl = new FfmpegDecodeFrame_Impl(URL, cli_iface.GetOptions());
}

FfmpegDecodeFrame::~FfmpegDecodeFrame() { delete pImpl; }