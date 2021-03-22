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

#pragma once

#if defined(_WIN32)
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
}

#include "CodecsSupport.hpp"
#include "NvCodecUtils.h"
#include "cuviddec.h"
#include <map>
#include <string>
#include <vector>

namespace VPF {
struct SeekContext {
  /* Frame we want to get. Set by user. */
  int64_t seek_frame;

  /* Current frame pts to determine seek direction. */
  int64_t curr_pts;

  /* Number of frames decoder had to decode to return desired frame.
   * Set by decoder. */
  int64_t dec_frames;

  SeekContext() : seek_frame(-1), dec_frames(0), curr_pts(0) {}

  SeekContext(int64_t frame_num)
      : seek_frame(frame_num), dec_frames(0), curr_pts(0) {}

  SeekContext(const SeekContext &other)
      : seek_frame(other.seek_frame), dec_frames(other.dec_frames),
        curr_pts(other.curr_pts) {}

  SeekContext &operator=(const SeekContext &other) {
    seek_frame = other.seek_frame;
    dec_frames = other.dec_frames;
    curr_pts = other.curr_pts;
    return *this;
  }
};

} 

class DataProvider;

class DllExport FFmpegDemuxer {
  AVIOContext *avioc = nullptr;
  AVBSFContext *bsfc_annexb = nullptr, *bsfc_sei = nullptr;
  AVFormatContext *fmtc = nullptr;

  AVPacket pkt, pktAnnexB, pktSei;
  AVCodecID eVideoCodec = AV_CODEC_ID_NONE;
  AVPixelFormat eChromaFormat;

  uint32_t width;
  uint32_t height;
  uint32_t gop_size;
  double framerate;
  double timebase;

  int videoStream = -1;

  bool is_mp4H264;
  bool is_mp4HEVC;
  bool is_EOF = false;

  std::vector<uint8_t> annexbBytes;
  std::vector<uint8_t> seiBytes;

  explicit FFmpegDemuxer(AVFormatContext *fmtcx);

  AVFormatContext *
  CreateFormatContext(DataProvider *pDataProvider,
                      const std::map<std::string, std::string> &ffmpeg_options);

  AVFormatContext *
  CreateFormatContext(const char *szFilePath,
                      const std::map<std::string, std::string> &ffmpeg_options);

public:
  explicit FFmpegDemuxer(
      const char *szFilePath,
      const std::map<std::string, std::string> &ffmpeg_options);
  explicit FFmpegDemuxer(
      DataProvider *pDataProvider,
      const std::map<std::string, std::string> &ffmpeg_options);
  ~FFmpegDemuxer();

  AVCodecID GetVideoCodec() const;

  uint32_t GetWidth() const;

  uint32_t GetHeight() const;

  uint32_t GetGopSize() const;

  double GetFramerate() const;

  double GetTimebase() const;

  uint32_t GetVideoStreamIndex() const;

  AVPixelFormat GetPixelFormat() const;

  bool Demux(uint8_t *&pVideo, size_t &rVideoBytes,
             PacketData &rCtx, uint8_t **ppSEI = nullptr,
             size_t *pSEIBytes = nullptr);

  bool Seek(VPF::SeekContext *p_ctx);

  static int ReadPacket(void *opaque, uint8_t *pBuf, int nBuf);
};

inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
  switch (id) {
  case AV_CODEC_ID_MPEG1VIDEO:
    return cudaVideoCodec_MPEG1;
  case AV_CODEC_ID_MPEG2VIDEO:
    return cudaVideoCodec_MPEG2;
  case AV_CODEC_ID_MPEG4:
    return cudaVideoCodec_MPEG4;
  case AV_CODEC_ID_VC1:
    return cudaVideoCodec_VC1;
  case AV_CODEC_ID_H264:
    return cudaVideoCodec_H264;
  case AV_CODEC_ID_HEVC:
    return cudaVideoCodec_HEVC;
  case AV_CODEC_ID_VP8:
    return cudaVideoCodec_VP8;
  case AV_CODEC_ID_VP9:
    return cudaVideoCodec_VP9;
  case AV_CODEC_ID_MJPEG:
    return cudaVideoCodec_JPEG;
  default:
    return cudaVideoCodec_NumCodecs;
  }
}
