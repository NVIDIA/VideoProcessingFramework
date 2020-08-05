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
#pragma once

#include "MemoryInterfaces.hpp"
#include "NvCodecCLIOptions.h"
#include "TC_CORE.hpp"
#include "Tasks.hpp"

#ifdef GENERATE_PYTHON_BINDINGS
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#else
#include <map>
#include <string>
#include <vector>
#endif

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/motion_vector.h>
}

#include <memory>
#include <stdexcept>

using namespace VPF;

class DllExport PyFrameUploader {
  std::unique_ptr<CudaUploadFrame> uploader;
  uint32_t gpuID = 0U, surfaceWidth, surfaceHeight;
  Pixel_Format surfaceFormat;

public:
  PyFrameUploader(uint32_t width, uint32_t height, Pixel_Format format,
                  uint32_t gpu_ID);
  Pixel_Format GetFormat();
#ifdef GENERATE_PYTHON_BINDINGS
  std::shared_ptr<Surface> UploadSingleFrame(py::array_t<uint8_t> &frame);
#else
  std::shared_ptr<Surface> UploadSingleFrame(std::vector<uint8_t> &frame);
#endif
};

class DllExport PySurfaceDownloader {
  std::unique_ptr<CudaDownloadSurface> upDownloader;
  uint32_t gpuID = 0U, surfaceWidth, surfaceHeight;
  Pixel_Format surfaceFormat;

public:
  PySurfaceDownloader(uint32_t width, uint32_t height, Pixel_Format format,
                      uint32_t gpu_ID);
  Pixel_Format GetFormat();
#ifdef GENERATE_PYTHON_BINDINGS
  bool DownloadSingleSurface(std::shared_ptr<Surface> surface,
                             py::array_t<uint8_t> &frame);
#else
  bool DownloadSingleSurface(std::shared_ptr<Surface> surface,
                             std::vector<uint8_t> &frame);
#endif
};

class DllExport PySurfaceConverter {
  std::unique_ptr<ConvertSurface> upConverter;
  Pixel_Format outputFormat;
  uint32_t gpuId;

public:
  PySurfaceConverter(uint32_t width, uint32_t height, Pixel_Format inFormat,
                     Pixel_Format outFormat, uint32_t gpuID);
  std::shared_ptr<Surface> Execute(std::shared_ptr<Surface> surface);
  Pixel_Format GetFormat();
};

class DllExport PySurfaceResizer {
  std::unique_ptr<ResizeSurface> upResizer;
  Pixel_Format outputFormat;
  uint32_t gpuId;

public:
  PySurfaceResizer(uint32_t width, uint32_t height, Pixel_Format format,
                   uint32_t gpuID);
  Pixel_Format GetFormat();
  std::shared_ptr<Surface> Execute(std::shared_ptr<Surface> surface);
};

struct DllExport MotionVector {
  int source;
  int w, h;
  int src_x, src_y;
  int dst_x, dst_y;
  int motion_x, motion_y;
  int motion_scale;
};

class DllExport PyFfmpegDecoder {
  std::unique_ptr<FfmpegDecodeFrame> upDecoder = nullptr;

public:
  PyFfmpegDecoder(const std::string &pathToFile,
                  const std::map<std::string, std::string> &ffmpeg_options);
#ifdef GENERATE_PYTHON_BINDINGS
  bool DecodeSingleFrame(py::array_t<uint8_t> &frame);
  py::array_t<MotionVector> GetMotionVectors();
#else
  bool DecodeSingleFrame(std::vector<uint8_t> &frame);
  std::vector<MotionVector> GetMotionVectors();
#endif
  void *GetSideData(AVFrameSideDataType data_type, size_t &raw_size);
};

class DllExport HwResetException : public std::runtime_error {
public:
  HwResetException(std::string &str) : std::runtime_error(str) {}
  HwResetException() : std::runtime_error("HW reset") {}
};

class DllExport PyNvDecoder {
  std::unique_ptr<DemuxFrame> upDemuxer;
  std::unique_ptr<NvdecDecodeFrame> upDecoder;
  std::unique_ptr<PySurfaceDownloader> upDownloader;
  uint32_t gpuId;
  static uint32_t const poolFrameSize = 4U;

public:
  PyNvDecoder(const std::string &pathToFile, int gpuOrdinal);

  PyNvDecoder(const std::string &pathToFile, int gpuOrdinal,
              const std::map<std::string, std::string> &ffmpeg_options);
  static Buffer *getElementaryVideo(DemuxFrame *demuxer);
  static Surface *getDecodedSurface(NvdecDecodeFrame *decoder,
                                    DemuxFrame *demuxer,
                                    bool &hw_decoder_failure);
  static bool getDecodedSurfaceFlush(NvdecDecodeFrame *decoder,
                                     DemuxFrame *demuxer, Surface *&output);
  uint32_t Width() const;
  void LastPacketData(PacketData &packetData) const;
  uint32_t Height() const;
  double Framerate() const;
  double Timebase() const;
  uint32_t Framesize() const;
  Pixel_Format GetPixelFormat() const;
  std::shared_ptr<Surface> DecodeSingleSurface();
#ifdef GENERATE_PYTHON_BINDINGS
  bool DecodeSingleFrame(py::array_t<uint8_t> &frame);
#else
  bool DecodeSingleFrame(std::vector<uint8_t> &frame);
#endif
};

class DllExport PyNvEncoder {
  std::unique_ptr<PyFrameUploader> uploader;
  std::unique_ptr<NvencEncodeFrame> upEncoder;
  uint32_t encWidth, encHeight, gpuId;
  Pixel_Format eFormat = NV12;
  std::map<std::string, std::string> options;
  bool verbose_ctor;

public:
  uint32_t Width() const;
  uint32_t Height() const;
  Pixel_Format GetPixelFormat() const;
  bool Reconfigure(const std::map<std::string, std::string> &encodeOptions,
                   bool force_idr = false, bool reset_enc = false,
                   bool verbose = false);
  PyNvEncoder(const std::map<std::string, std::string> &encodeOptions,
              int gpuOrdinal, bool verbose = false);
#ifdef GENERATE_PYTHON_BINDINGS
  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t> &packet, bool sync);
  bool EncodeSingleSurface(std::shared_ptr<Surface> rawSurface,
                           py::array_t<uint8_t> &packet, bool append,
                           bool sync);
  bool EncodeSingleFrame(py::array_t<uint8_t> &inRawFrame,
                         py::array_t<uint8_t> &packet, bool sync);
  bool Flush(py::array_t<uint8_t> &packets);
#else
  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     std::vector<uint8_t> &packet, bool sync);
  bool EncodeSingleSurface(std::shared_ptr<Surface> rawSurface,
                           std::vector<uint8_t> &packet, bool append,
                           bool sync);
  bool EncodeSingleFrame(std::vector<uint8_t> &inRawFrame,
                         std::vector<uint8_t> &packet, bool sync);
  bool Flush(std::vector<uint8_t> &packets);
#endif
};