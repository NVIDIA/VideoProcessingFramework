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
#include <stdexcept>

namespace VPF {
enum SeekMode {
  /* Seek for exact frame number.
   * Suited for standalone demuxer seek. */
  EXACT_FRAME = 0,
  /* Seek for previous key frame in past.
   * Suitable for seek & decode.  */
  PREV_KEY_FRAME = 1
};

struct SeekContext {
  /* Will be set to false for default ctor, true otherwise;
   */
  bool use_seek;

  /* Frame we want to get. Set by user. */
  int64_t seek_frame;

  /* Mode in which we seek. */
  SeekMode mode;

  /* PTS of frame found after seek. */
  int64_t out_frame_pts;

  /* Duration of frame found after seek. */
  int64_t out_frame_duration;

  /* Number of frames that were decoded during seek. */
  uint64_t num_frames_decoded;

  SeekContext()
      : use_seek(false), seek_frame(0), mode(PREV_KEY_FRAME), out_frame_pts(0),
        out_frame_duration(0), num_frames_decoded(0U) {}

  SeekContext(int64_t frame_num)
      : use_seek(true), seek_frame(frame_num), mode(PREV_KEY_FRAME),
        out_frame_pts(0), out_frame_duration(0), num_frames_decoded(0U) {
    if (frame_num < 0) {
      throw std::runtime_error("Negative frame number is not allowed");
    }
  }

  SeekContext(int64_t frame_num, SeekMode seek_mode)
      : use_seek(true), seek_frame(frame_num), mode(seek_mode),
        out_frame_pts(0), out_frame_duration(0), num_frames_decoded(0U) {
    if (frame_num < 0) {
      throw std::runtime_error("Negative frame number is not allowed");
    }
  }

  SeekContext(const SeekContext &other)
      : use_seek(other.use_seek), seek_frame(other.seek_frame),
        mode(other.mode), out_frame_pts(other.out_frame_pts),
        out_frame_duration(other.out_frame_duration),
        num_frames_decoded(other.num_frames_decoded) {}

  SeekContext &operator=(const SeekContext &other) {
    use_seek = other.use_seek;
    seek_frame = other.seek_frame;
    mode = other.mode;
    out_frame_pts = other.out_frame_pts;
    out_frame_duration = other.out_frame_duration;
    num_frames_decoded = other.num_frames_decoded;
    return *this;
  }
};

} 

class DataProvider;

class DllExport FFmpegDemuxer {
  AVIOContext *avioc = nullptr;
  AVBSFContext *bsfc_annexb = nullptr, *bsfc_sei = nullptr;
  AVFormatContext *fmtc = nullptr;

  AVPacket pktSrc, pktDst, pktSei;
  AVCodecID eVideoCodec = AV_CODEC_ID_NONE;
  AVPixelFormat eChromaFormat;

  uint32_t width;
  uint32_t height;
  uint32_t gop_size;
  uint64_t nb_frames;
  double framerate;
  double timebase;

  int videoStream = -1;

  bool is_seekable;
  bool is_mp4H264;
  bool is_mp4HEVC;
  bool is_VP9;
  bool is_EOF = false;

  PacketData last_packet_data;

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

  uint32_t GetNumFrames() const;

  double GetFramerate() const;

  double GetTimebase() const;

  uint32_t GetVideoStreamIndex() const;

  AVPixelFormat GetPixelFormat() const;

  bool Demux(uint8_t *&pVideo, size_t &rVideoBytes, PacketData &pktData,
             uint8_t **ppSEI = nullptr, size_t *pSEIBytes = nullptr);

  bool Seek(VPF::SeekContext &seek_ctx, uint8_t *&pVideo,
            size_t &rVideoBytes, PacketData &pktData, uint8_t **ppSEI = nullptr,
            size_t *pSEIBytes = nullptr);

  void Flush();

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
