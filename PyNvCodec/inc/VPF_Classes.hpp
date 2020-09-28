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

#include "MemoryInterfaces.hpp"
#include "NvCodecCLIOptions.h"
#include "TC_CORE.hpp"
#include "Tasks.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/motion_vector.h>
}

using namespace VPF;

struct VpfFrameUploaderContext {
  uint32_t gpuID;
  uint32_t width;
  uint32_t height;

  Pixel_Format format;

  VpfFrameUploaderContext()
      : gpuID(0U), width(0U), height(0U), format(UNDEFINED) {}

  VpfFrameUploaderContext(uint32_t new_gpuID, uint32_t new_width,
                          uint32_t new_height, Pixel_Format new_format)
      : gpuID(new_gpuID), width(new_width), height(new_height),
        format(new_format) {}
};

struct VpfFrameUploaderArgs {
  const uint8_t *frame;
  const size_t frameSize;

  std::shared_ptr<Surface> surface;
  std::string errorMessage;

  VpfFrameUploaderArgs(const uint8_t *new_frame, const size_t new_frameSize)
      : frame(new_frame), frameSize(new_frameSize), surface(nullptr) {}
};

class VpfFrameUploader {
  std::unique_ptr<CudaUploadFrame> upUploader = nullptr;
  std::unique_ptr<VpfFrameUploaderContext> upCtx = nullptr;

public:
  explicit VpfFrameUploader(const VpfFrameUploaderContext &ctx);

  inline uint32_t GetWidth() const { return upCtx->width; }
  inline uint32_t GetHeight() const { return upCtx->height; }
  inline Pixel_Format GetFormat() const { return upCtx->format; }

  bool UploadSingleFrame(VpfFrameUploaderArgs &args);
};

typedef VpfFrameUploaderContext VpfSurfaceDownloaderContext;

struct VpfSurfaceDownloaderArgs {
  std::shared_ptr<uint8_t> frame;
  size_t frameSize;

  std::shared_ptr<Surface> surface;
  std::string errorMessage;

  VpfSurfaceDownloaderArgs()
      : frame(nullptr), frameSize(0U), surface(nullptr) {}

  VpfSurfaceDownloaderArgs(std::shared_ptr<Surface> new_surface)
      : frame(nullptr), frameSize(0U), surface(new_surface) {}
};

class VpfSurfaceDownloader {
  std::unique_ptr<CudaDownloadSurface> upDownloader = nullptr;
  std::unique_ptr<VpfSurfaceDownloaderContext> upCtx = nullptr;

public:
  explicit VpfSurfaceDownloader(const VpfSurfaceDownloaderContext &ctx);

  inline uint32_t GetWidth() const { return upCtx->width; }
  inline uint32_t GetHeight() const { return upCtx->height; }
  inline Pixel_Format GetFormat() const { return upCtx->format; }

  bool DownloadSingleSurface(VpfSurfaceDownloaderArgs &args);
};

struct VpfSurfaceConverterContext {
  uint32_t gpuID;
  uint32_t width;
  uint32_t height;

  Pixel_Format srcFormat;
  Pixel_Format dstFormat;

  VpfSurfaceConverterContext(uint32_t new_gpuID, uint32_t new_width,
                             uint32_t new_height, Pixel_Format new_srcFormat,
                             Pixel_Format new_dstFormat)
      : gpuID(new_gpuID), width(new_width), height(new_height),
        srcFormat(new_srcFormat), dstFormat(new_dstFormat) {}
};

struct VpfSurfaceConverterArgs {
  std::shared_ptr<Surface> srcSurface;
  std::shared_ptr<Surface> dstSurface;

  std::string errorMessage;

  VpfSurfaceConverterArgs(std::shared_ptr<Surface> new_srcSurface)
      : srcSurface(new_srcSurface), dstSurface(nullptr) {}
};

class VpfSurfaceConverter {
  std::unique_ptr<ConvertSurface> upConverter = nullptr;
  std::unique_ptr<VpfSurfaceConverterContext> upCtx = nullptr;

public:
  explicit VpfSurfaceConverter(const VpfSurfaceConverterContext &ctx);

  inline uint32_t GetWidth() const { return upCtx->width; }
  inline uint32_t GetHeight() const { return upCtx->height; }
  inline Pixel_Format GetSrcFormat() const { return upCtx->srcFormat; }
  inline Pixel_Format GetDstFormat() const { return upCtx->dstFormat; }

  bool ConvertSingleSurface(VpfSurfaceConverterArgs &args);
};

typedef VpfFrameUploaderContext VpfSurfaceResizerContext;
typedef VpfSurfaceConverterArgs VpfSurfaceResizerArgs;

class VpfSurfaceResizer {
  std::unique_ptr<ResizeSurface> upResizer = nullptr;
  std::unique_ptr<VpfSurfaceResizerContext> upCtx = nullptr;

public:
  explicit VpfSurfaceResizer(const VpfSurfaceResizerContext &ctx);

  inline uint32_t GetWidth() const { return upCtx->width; }
  inline uint32_t GetHeight() const { return upCtx->height; }
  inline Pixel_Format GetFormat() const { return upCtx->format; }

  bool ResizeSingleSurface(VpfSurfaceResizerArgs &args);
};

struct MotionVector {
  int source;
  int w, h;
  int src_x, src_y;
  int dst_x, dst_y;
  int motion_x, motion_y;
  int motion_scale;
};

struct VpfFfmpegDecoderContext {
  std::string pathToFile;
  std::map<std::string, std::string> ffmpegOptions;

  VpfFfmpegDecoderContext(
      const std::string &new_pathToFile,
      const std::map<std::string, std::string> &new_ffmpegOptions)
      : pathToFile(new_pathToFile), ffmpegOptions(new_ffmpegOptions) {}
};

struct VpfFfmpegDecoderArgs {
  std::shared_ptr<uint8_t> frame;
  size_t frameSize;

  std::shared_ptr<MotionVector> motionVectors;
  size_t motionVectorsSize;
  bool needMotionVectors;

  std::string errorMessage;

  VpfFfmpegDecoderArgs()
      : frame(nullptr), frameSize(0U), motionVectors(nullptr),
        motionVectorsSize(0U), needMotionVectors(false) {}
};

class VpfFfmpegDecoder {
  std::unique_ptr<FfmpegDecodeFrame> upDecoder = nullptr;
  std::unique_ptr<VpfFfmpegDecoderContext> upCtx = nullptr;
  void *GetSideData(AVFrameSideDataType data_type, size_t &raw_size);
  bool GetMotionVectors(VpfFfmpegDecoderArgs &args);

public:
  explicit VpfFfmpegDecoder(const VpfFfmpegDecoderContext &ctx);
  bool DecodeSingleFrame(VpfFfmpegDecoderArgs &args);
};

class HwResetException : public std::runtime_error {
public:
  HwResetException(std::string &str) : std::runtime_error(str) {}
  HwResetException() : std::runtime_error("HW reset") {}
};

struct VpfNvDecoderContext {
  uint32_t gpuID;
  std::string pathToFile;
  std::map<std::string, std::string> ffmpegOptions;
};

struct VpfNvDecoderArgs {
  std::shared_ptr<uint8_t> decodedFrame;
  size_t decodedFrameSize;

  std::shared_ptr<uint8_t> sei;
  size_t seiSize;
  bool needSei;

  std::shared_ptr<Surface> decodedSurface;
  std::string errorMessage;

  bool decoderHwReset;
};

class VpfNvDecoder {
  std::unique_ptr<DemuxFrame> upDemuxer = nullptr;
  std::unique_ptr<NvdecDecodeFrame> upDecoder = nullptr;
  std::unique_ptr<VpfSurfaceDownloader> upDownloader = nullptr;
  std::unique_ptr<VpfNvDecoderContext> upCtx = nullptr;

  static uint32_t const poolFrameSize = 4U;

  Buffer *GetVideoPacket();
  bool DecodeSurfaceInternal(VpfNvDecoderArgs &args);
  bool FlushSingleSurface(VpfNvDecoderArgs &args);

public:
  VpfNvDecoder(const VpfNvDecoderContext &ctx);

  uint32_t Width() const;
  uint32_t Height() const;
  double Framerate() const;
  double Timebase() const;
  Pixel_Format GetPixelFormat() const;
  void LastPacketData(PacketData &packetData) const;

  bool DecodeSingleSurface(VpfNvDecoderArgs &args);
  bool DecodeSingleFrame(VpfNvDecoderArgs &args);
};

struct VpfNvEncoderContext {
  uint32_t gpuID;
  uint32_t width;
  uint32_t height;

  Pixel_Format format;
  std::map<std::string, std::string> options;

  bool verbose;
};

struct VpfNvEncoderReconfigureContext {
  std::map<std::string, std::string> options;
  bool forceIDR;
  bool reset;
  bool verbose;
};

struct VpfNvEncoderArgs {
  const std::shared_ptr<uint8_t> seiMessage;
  size_t seiMessageSize;

  std::shared_ptr<uint8_t> frame;
  size_t frameSize;

  std::shared_ptr<Surface> surface;

  std::shared_ptr<uint8_t> packet;
  size_t packetSize;

  bool sync;
  bool append;

  std::string errorMessage;
};

class VpfNvEncoder {
  std::unique_ptr<VpfFrameUploader> upUploader = nullptr;
  std::unique_ptr<NvencEncodeFrame> upEncoder = nullptr;
  std::unique_ptr<VpfNvEncoderContext> upCtx = nullptr;

public:
  VpfNvEncoder(const VpfNvEncoderContext &ctx);

  inline uint32_t Width() const { return upCtx->width; }
  inline uint32_t Height() const { return upCtx->height; }
  inline Pixel_Format GetPixelFormat() const { return upCtx->format; }

  bool Reconfigure(const VpfNvEncoderReconfigureContext &ctx);

  bool EncodeSingleSurface(VpfNvEncoderArgs &args);
  bool EncodeSingleFrame(VpfNvEncoderArgs &args);
  bool Flush(VpfNvEncoderArgs &args);
};

class CudaResMgr {
  CudaResMgr();

public:
  static CudaResMgr &Instance();
  CUcontext GetCtx(size_t idx);
  CUstream GetStream(size_t idx);
  ~CudaResMgr();
  static size_t GetNumGpus();

  std::vector<CUcontext> g_Contexts;
  std::vector<CUstream> g_Streams;
};