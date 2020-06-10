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
struct FfmpegDecodeFrame_Impl {
  AVFormatContext *fmt_ctx = nullptr;
  AVCodecContext *video_dec_ctx = nullptr;
  AVStream *video_stream = nullptr;
  AVFrame *frame = nullptr;
  AVPacket pkt = {0};

  Buffer *dec_frame;
  map<AVFrameSideDataType, Buffer *> side_data;

  int video_stream_idx = -1;
  bool end_encode = false;

  string src_filename;

  FfmpegDecodeFrame_Impl(const char *URL, AVDictionary *pOptions)
      : src_filename(URL) {

    auto res = avformat_open_input(&fmt_ctx, src_filename.c_str(), NULL, NULL);
    if (res < 0) {
      stringstream ss;
      ss << "Could not open source file" << src_filename << endl;
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

    OpenCodecContext(fmt_ctx, AVMEDIA_TYPE_VIDEO, pOptions);

    av_dump_format(fmt_ctx, 0, src_filename.c_str(), 0);

    if (!video_stream) {
      cerr << "Could not find video stream in the input, aborting" << endl;
    }

    frame = av_frame_alloc();
    if (!frame) {
      cerr << "Could not allocate frame" << endl;
    }
  }

  bool SaveYUV420(AVFrame *pframe) {
    // Detect frame size & allocate memory if necessary;
    size_t size = frame->width * frame->height * 3 / 2;

    if (!dec_frame) {
      dec_frame = Buffer::Make(size);
    } else if (size != dec_frame->GetRawMemSize()) {
      delete dec_frame;
      dec_frame = Buffer::Make(size);
    }

    // Copy pixels;
    auto plane = 0U;
    ptrdiff_t pos = 0U;

    while (frame->data[plane]) {
      auto *dst = dec_frame->GetDataAs<uint8_t>() + pos;
      auto *src = frame->data[plane];

      auto width = (0 == plane) ? frame->width : frame->width / 2;
      auto height = (0 == plane) ? frame->height : frame->height / 2;

      for (int i = 0; i < height; i++) {
        memcpy(dst, src, width);
        dst += width;
        src += frame->linesize[plane];
      }

      plane++;
      pos += width * height;
    }

    return true;
  }

  bool DecodeSingleFrame() {
    if (end_encode) {
      return false;
    }

    auto ret = av_read_frame(fmt_ctx, &pkt);
    if (ret >= 0) {
      auto res = DecodeSinglePacket(&pkt);
      av_packet_unref(&pkt);
      return res;
    } else {
      // Flush decoder;
      end_encode = true;
      return DecodeSinglePacket(nullptr);
    }
  }

  // Saves reconstructed pixels;
  bool SaveVideoFrame(AVFrame *frame) {
    // Only YUV420P is supported so far;
    if (AV_PIX_FMT_YUV420P != frame->format) {
      return false;
    }

    SaveYUV420(frame);
  }

  // Saves frame side data;
  bool SaveSideData(AVFrame *frame) { return true; }

  bool DecodeSinglePacket(const AVPacket *pkt) {
    auto res = avcodec_send_packet(video_dec_ctx, pkt);
    if (res < 0) {
      stringstream ss;
      ss << "Error while sending a packet to the decoder" << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      cerr << ss.str();
      return false;
    }

    while (res >= 0) {
      res = avcodec_receive_frame(video_dec_ctx, frame);
      if (res == AVERROR(EAGAIN) || res == AVERROR_EOF) {
        break;
      } else if (res < 0) {
        stringstream ss;
        ss << "Error while receiving a frame from the decoder" << endl;
        ss << "Error description: " << AvErrorToString(res) << endl;
        cerr << ss.str();
        return false;
      }

      if (res >= 0) {
        SaveVideoFrame(frame);
        SaveSideData(frame);
        av_frame_unref(frame);
      }
    }

    return true;
  }

  int OpenCodecContext(AVFormatContext *fmt_ctx, enum AVMediaType type,
                       AVDictionary *opts) {
    AVStream *st;
    AVCodecContext *dec_ctx = NULL;
    AVCodec *dec = NULL;

    auto res = av_find_best_stream(fmt_ctx, type, -1, -1, &dec, 0);
    if (res < 0) {
      stringstream ss;
      ss << "Could not find " << av_get_media_type_string(type)
         << " stream in file " << src_filename << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      throw runtime_error(ss.str());
    } else {
      int stream_idx = res;
      st = fmt_ctx->streams[stream_idx];

      dec_ctx = avcodec_alloc_context3(dec);
      if (!dec_ctx) {
        cerr << "Failed to allocate codec context" << endl;
        return AVERROR(EINVAL);
      }

      res = avcodec_parameters_to_context(dec_ctx, st->codecpar);
      if (res < 0) {
        stringstream ss;
        ss << "Failed to copy codec parameters to codec context" << endl;
        ss << "Error description: " << AvErrorToString(res) << endl;
        throw runtime_error(ss.str());
      }

      /* Init the video decoder */
      res = avcodec_open2(dec_ctx, dec, &opts);
      if (res < 0) {
        stringstream ss;
        ss << "Failed to open codec " << av_get_media_type_string(type) << endl;
        ss << "Error description: " << AvErrorToString(res) << endl;
        throw runtime_error(ss.str());
      }

      video_stream_idx = stream_idx;
      video_stream = fmt_ctx->streams[video_stream_idx];
      video_dec_ctx = dec_ctx;
    }

    return 0;
  }

  ~FfmpegDecodeFrame_Impl() {
    avcodec_free_context(&video_dec_ctx);
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

TaskExecStatus FfmpegDecodeFrame::Execute() {
  ClearOutputs();

  if (pImpl->DecodeSingleFrame()) {
    SetOutput((Token *)pImpl->dec_frame, 0U);
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
  pImpl = new FfmpegDecodeFrame_Impl(URL, nullptr /*cli_iface.GetOptions()*/);
}

FfmpegDecodeFrame::~FfmpegDecodeFrame() { delete pImpl; }