/*
 * Copyright 2020 NVIDIA Corporation
 * Copyright 2021 Kognia Sports Intelligence
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

#include "MemoryInterfaces.hpp"
#include "NvCodecCLIOptions.h"
#include "FFmpegDemuxer.h"
#include "NvDecoder.h"
#include "TC_CORE.hpp"
#include "Tasks.hpp"

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/motion_vector.h>
}

using namespace VPF;
namespace py = pybind11;

struct MotionVector {
  int source;
  int w, h;
  int src_x, src_y;
  int dst_x, dst_y;
  int motion_x, motion_y;
  int motion_scale;
};

class HwResetException : public std::runtime_error {
public:
  HwResetException(std::string &str) : std::runtime_error(str) {}
  HwResetException() : std::runtime_error("HW reset") {}
};

class CuvidParserException : public std::runtime_error {
public:
  CuvidParserException(std::string &str) : std::runtime_error(str) {}
  CuvidParserException() : std::runtime_error("HW reset") {}
};

class PyFrameUploader {
  std::unique_ptr<CudaUploadFrame> uploader;
  uint32_t surfaceWidth, surfaceHeight;
  Pixel_Format surfaceFormat;

public:
  PyFrameUploader(uint32_t width, uint32_t height, Pixel_Format format,
                  uint32_t gpu_ID);

  PyFrameUploader(uint32_t width, uint32_t height, Pixel_Format format,
                  CUcontext ctx, CUstream str);

  PyFrameUploader(uint32_t width, uint32_t height, Pixel_Format format,
                  size_t ctx, size_t str) : 
    PyFrameUploader(width, height, format, (CUcontext)ctx, (CUstream)str) {}

  Pixel_Format GetFormat();

  std::shared_ptr<Surface> UploadSingleFrame(py::array_t<uint8_t> &frame);
};

class PySurfaceDownloader {
  std::unique_ptr<CudaDownloadSurface> upDownloader;
  uint32_t surfaceWidth, surfaceHeight;
  Pixel_Format surfaceFormat;

public:
  PySurfaceDownloader(uint32_t width, uint32_t height, Pixel_Format format,
                      uint32_t gpu_ID);

  PySurfaceDownloader(uint32_t width, uint32_t height, Pixel_Format format,
                      CUcontext ctx, CUstream str);

  PySurfaceDownloader(uint32_t width, uint32_t height, Pixel_Format format,
                      size_t ctx, size_t str):
    PySurfaceDownloader(width, height, format, (CUcontext)ctx, (CUstream)str) {}

  Pixel_Format GetFormat();

  bool DownloadSingleSurface(std::shared_ptr<Surface> surface,
                             py::array_t<uint8_t> &frame);
};

class PySurfaceConverter {
  std::unique_ptr<ConvertSurface> upConverter;
  std::unique_ptr<Buffer> upCtxBuffer;
  Pixel_Format outputFormat;

public:
  PySurfaceConverter(uint32_t width, uint32_t height, Pixel_Format inFormat,
                     Pixel_Format outFormat, uint32_t gpuID);

  PySurfaceConverter(uint32_t width, uint32_t height, Pixel_Format inFormat,
                     Pixel_Format outFormat, CUcontext ctx, CUstream str);

  PySurfaceConverter(uint32_t width, uint32_t height, Pixel_Format inFormat,
                     Pixel_Format outFormat, size_t ctx, size_t str):
    PySurfaceConverter(width, height, inFormat, outFormat, (CUcontext)ctx, (CUstream)str) {}

  std::shared_ptr<Surface>
  Execute(std::shared_ptr<Surface> surface,
          std::shared_ptr<ColorspaceConversionContext> context);

  Pixel_Format GetFormat();
};

class PySurfaceResizer {
  std::unique_ptr<ResizeSurface> upResizer;
  Pixel_Format outputFormat;

public:
  PySurfaceResizer(uint32_t width, uint32_t height, Pixel_Format format,
                   uint32_t gpuID);

  PySurfaceResizer(uint32_t width, uint32_t height, Pixel_Format format,
                   CUcontext ctx, CUstream str);

  PySurfaceResizer(uint32_t width, uint32_t height, Pixel_Format format,
                   size_t ctx, size_t str):
    PySurfaceResizer(width, height, format, (CUcontext)ctx, (CUstream)str){}

  Pixel_Format GetFormat();

  std::shared_ptr<Surface> Execute(std::shared_ptr<Surface> surface);
};

class PyFFmpegDemuxer {
  std::unique_ptr<DemuxFrame> upDemuxer;

public:
  PyFFmpegDemuxer(const std::string &pathToFile);
  PyFFmpegDemuxer(const std::string &pathToFile,
                  const std::map<std::string, std::string> &ffmpeg_options);

  bool DemuxSinglePacket(py::array_t<uint8_t> &packet);

  void GetLastPacketData(PacketData &pkt_data);

  bool Seek(SeekContext &ctx, py::array_t<uint8_t> &packet);

  uint32_t Width() const;

  uint32_t Height() const;

  Pixel_Format Format() const;

  ColorSpace GetColorSpace() const;

  ColorRange GetColorRange() const;

  cudaVideoCodec Codec() const;

  double Framerate() const;

  double AvgFramerate() const;

  bool IsVFR() const;

  uint32_t Numframes() const;

  double Timebase() const;

};

class PyFfmpegDecoder {
  std::unique_ptr<FfmpegDecodeFrame> upDecoder = nullptr;

  void *GetSideData(AVFrameSideDataType data_type, size_t &raw_size);

public:
  PyFfmpegDecoder(const std::string &pathToFile,
                  const std::map<std::string, std::string> &ffmpeg_options);

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame);

  py::array_t<MotionVector> GetMotionVectors();
};

class PyNvDecoder {
  std::unique_ptr<DemuxFrame> upDemuxer;
  std::unique_ptr<NvdecDecodeFrame> upDecoder;
  std::unique_ptr<PySurfaceDownloader> upDownloader;
  uint32_t gpuID;
  static uint32_t const poolFrameSize = 4U;
  Pixel_Format format;

public:
  PyNvDecoder(uint32_t width, uint32_t height, Pixel_Format format,
              cudaVideoCodec codec, uint32_t gpuOrdinal);

  PyNvDecoder(const std::string &pathToFile, int gpuOrdinal);

  PyNvDecoder(const std::string &pathToFile, int gpuOrdinal,
              const std::map<std::string, std::string> &ffmpeg_options);

  PyNvDecoder(uint32_t width, uint32_t height, Pixel_Format format,
              cudaVideoCodec codec, CUcontext ctx, CUstream str);

  PyNvDecoder(uint32_t width, uint32_t height, Pixel_Format format,
              cudaVideoCodec codec, size_t ctx, size_t str):
    PyNvDecoder(width, height, format, codec, (CUcontext)ctx, (CUstream)str) {}

  PyNvDecoder(const std::string &pathToFile, CUcontext ctx, CUstream str);

  PyNvDecoder(const std::string &pathToFile, size_t ctx, size_t str):
    PyNvDecoder(pathToFile, (CUcontext)ctx, (CUstream)str){}

  PyNvDecoder(const std::string &pathToFile, CUcontext ctx, CUstream str,
              const std::map<std::string, std::string> &ffmpeg_options);

  PyNvDecoder(const std::string &pathToFile, size_t ctx, size_t str,
              const std::map<std::string, std::string> &ffmpeg_options):
    PyNvDecoder(pathToFile, (CUcontext)ctx, (CUstream)str, ffmpeg_options){}

  static Buffer *getElementaryVideo(DemuxFrame *demuxer,
                                    SeekContext &seek_ctx, bool needSEI);

  static Surface *getDecodedSurface(NvdecDecodeFrame *decoder,
                                    DemuxFrame *demuxer,
                                    SeekContext &seek_ctx, bool needSEI);

  uint32_t Width() const;

  ColorSpace GetColorSpace() const;

  ColorRange GetColorRange() const;

  void LastPacketData(PacketData &packetData) const;

  uint32_t Height() const;

  double Framerate() const;

  double AvgFramerate() const;

  bool IsVFR() const;

  uint32_t Numframes() const;

  double Timebase() const;

  uint32_t Framesize() const;

  Pixel_Format GetPixelFormat() const;

  std::shared_ptr<Surface> DecodeSurfaceFromPacket(py::array_t<uint8_t> &packet,
                                                   py::array_t<uint8_t> &sei);

  std::shared_ptr<Surface>
  DecodeSurfaceFromPacket(py::array_t<uint8_t> &packet);

  std::shared_ptr<Surface> DecodeSingleSurface(py::array_t<uint8_t> &sei);

  std::shared_ptr<Surface> DecodeSingleSurface(py::array_t<uint8_t> &sei,
                                               PacketData &pkt_data);

  std::shared_ptr<Surface> DecodeSingleSurface(py::array_t<uint8_t> &sei,
                                               SeekContext &ctx);

  std::shared_ptr<Surface> DecodeSingleSurface(py::array_t<uint8_t> &sei,
                                               SeekContext &ctx,
                                               PacketData &pkt_data);

  std::shared_ptr<Surface> DecodeSingleSurface();

  std::shared_ptr<Surface> DecodeSingleSurface(PacketData &pkt_data);

  std::shared_ptr<Surface> DecodeSingleSurface(SeekContext &ctx);

  std::shared_ptr<Surface> DecodeSingleSurface(SeekContext &ctx,
                                               PacketData &pkt_data);

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame,
                         py::array_t<uint8_t> &sei);

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame,
                         py::array_t<uint8_t> &sei,
                         PacketData &pkt_data);

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame,
                         py::array_t<uint8_t> &sei,
                         SeekContext &ctx);

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame,
                         py::array_t<uint8_t> &sei,
                         SeekContext &ctx,
                         PacketData &pkt_data);

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame);

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame,
                         PacketData &pkt_data);

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame,
                         SeekContext &ctx);

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame,
                         SeekContext &ctx,
                         PacketData &pkt_data);

  bool DecodeFrameFromPacket(py::array_t<uint8_t> &frame,
                             py::array_t<uint8_t> &packet,
                             py::array_t<uint8_t> &sei);

  bool DecodeFrameFromPacket(py::array_t<uint8_t> &frame,
                             py::array_t<uint8_t> &packet);

  bool FlushSingleFrame(py::array_t<uint8_t> &frame);

  std::shared_ptr<Surface> FlushSingleSurface();

private:
  bool DecodeSurface(struct DecodeContext &ctx);

  Surface *getDecodedSurfaceFromPacket(py::array_t<uint8_t> *pPacket,
                                       bool no_eos = false);
};

struct EncodeContext {
  std::shared_ptr<Surface> rawSurface;
  py::array_t<uint8_t> *pPacket;
  const py::array_t<uint8_t> *pMessageSEI;
  bool sync;
  bool append;

  EncodeContext(std::shared_ptr<Surface> spRawSurface,
                py::array_t<uint8_t> *packet,
                const py::array_t<uint8_t> *messageSEI, bool is_sync,
                bool is_append)
      : rawSurface(spRawSurface), pPacket(packet), pMessageSEI(messageSEI),
        sync(is_sync), append(is_append) {}
};

class PyNvEncoder {
  std::unique_ptr<PyFrameUploader> uploader;
  std::unique_ptr<NvencEncodeFrame> upEncoder;
  uint32_t encWidth, encHeight;
  Pixel_Format eFormat;
  std::map<std::string, std::string> options;
  bool verbose_ctor;
  CUcontext cuda_ctx;
  CUstream cuda_str;

public:
  uint32_t Width() const;
  uint32_t Height() const;
  Pixel_Format GetPixelFormat() const;
  bool Reconfigure(const std::map<std::string, std::string> &encodeOptions,
                   bool force_idr = false, bool reset_enc = false,
                   bool verbose = false);

  PyNvEncoder(const std::map<std::string, std::string> &encodeOptions,
              int gpuOrdinal, Pixel_Format format = NV12, bool verbose = false);

  PyNvEncoder(const std::map<std::string, std::string> &encodeOptions,
              CUcontext ctx, CUstream str, Pixel_Format format = NV12, 
              bool verbose = false);

  PyNvEncoder(const std::map<std::string, std::string> &encodeOptions,
              size_t ctx, size_t str, Pixel_Format format = NV12, 
              bool verbose = false):
    PyNvEncoder(encodeOptions, (CUcontext)ctx, (CUstream)str, format, verbose){}

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t> &packet,
                     const py::array_t<uint8_t> &messageSEI, bool sync,
                     bool append);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t> &packet,
                     const py::array_t<uint8_t> &messageSEI, bool sync);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t> &packet, bool sync);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t> &packet,
                     const py::array_t<uint8_t> &messageSEI);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t> &packet);

  bool EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                   py::array_t<uint8_t> &packet);

  bool EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                   py::array_t<uint8_t> &packet,
                   const py::array_t<uint8_t> &messageSEI);

  bool EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                   py::array_t<uint8_t> &packet, bool sync);

  bool EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                   py::array_t<uint8_t> &packet,
                   const py::array_t<uint8_t> &messageSEI, bool sync);

  bool EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                   py::array_t<uint8_t> &packet,
                   const py::array_t<uint8_t> &messageSEI, bool sync,
                   bool append);

  // Flush all the encoded frames (packets)
  bool Flush(py::array_t<uint8_t> &packets);
  // Flush only one encoded frame (packet)
  bool FlushSinglePacket(py::array_t<uint8_t> &packet);

private:
  bool EncodeSingleSurface(EncodeContext &ctx);
};
