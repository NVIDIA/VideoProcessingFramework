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

#include "MemoryInterfaces.hpp"
#include "NvCodecCLIOptions.h"
#include "TC_CORE.hpp"
#include "Tasks.hpp"
#include "VPF_Classes.hpp"

#include <chrono>
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

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

static auto ThrowOnCudaError = [](CUresult res, int lineNum = -1) {
  if (CUDA_SUCCESS != res) {
    stringstream ss;

    if (lineNum > 0) {
      ss << __FILE__ << ":";
      ss << lineNum << endl;
    }

    const char *errName = nullptr;
    if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {
      ss << "CUDA error with code " << res << endl;
    } else {
      ss << "CUDA error: " << errName << endl;
    }

    const char *errDesc = nullptr;
    if (CUDA_SUCCESS != cuGetErrorString(res, &errDesc)) {
      // Try CUDA runtime function then;
      errDesc = cudaGetErrorString((cudaError_t)res);
    }

    if (!errDesc) {
      ss << "No error string available" << endl;
    } else {
      ss << errDesc << endl;
    }

    throw runtime_error(ss.str());
  }
};

auto CopySurface = [](shared_ptr<Surface> self, shared_ptr<Surface> other,
                      int gpuID) {
  auto cudaCtx = CudaResMgr::Instance().GetCtx(gpuID);
  CUstream cudaStream = CudaResMgr::Instance().GetStream(gpuID);

  for (auto plane = 0U; plane < self->NumPlanes(); plane++) {
    auto srcPlanePtr = self->PlanePtr(plane);
    auto dstPlanePtr = other->PlanePtr(plane);

    if (!srcPlanePtr || !dstPlanePtr) {
      break;
    }

    CudaCtxPush ctxPush(cudaCtx);

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = srcPlanePtr;
    m.dstDevice = dstPlanePtr;
    m.srcPitch = self->Pitch(plane);
    m.dstPitch = other->Pitch(plane);
    m.Height = self->Height(plane);
    m.WidthInBytes = self->WidthInBytes(plane);

    ThrowOnCudaError(cuMemcpy2DAsync(&m, cudaStream), __LINE__);
  }

  ThrowOnCudaError(cuStreamSynchronize(cudaStream), __LINE__);
};

template <typename T>
void PyArrayFromPtr(py::array_t<T> &array, shared_ptr<T> ptr, size_t size) {
  array.resize({size}, false);
  memcpy(array.mutable_data(), ptr.get(), size);
};

class PyFrameUploader {
public:
  PyFrameUploader(uint32_t width, uint32_t height, Pixel_Format format,
                  uint32_t gpuID)
      : ctx(gpuID, width, height, format) {
    upUploader.reset(new VpfFrameUploader(ctx));
  }

  shared_ptr<Surface> UploadSingleFrame(py::array_t<uint8_t> const &frame) {
    VpfFrameUploaderArgs uploaderArgs(frame.data(), frame.size());
    if (!upUploader->UploadSingleFrame(uploaderArgs)) {
      throw runtime_error(uploaderArgs.errorMessage);
    } else {
      return shared_ptr<Surface>(uploaderArgs.surface->Clone());
    }
  }

  inline uint32_t GetWidth() const { return ctx.width; }
  inline uint32_t GetHeight() const { return ctx.height; }
  inline Pixel_Format GetFormat() const { return ctx.format; }

private:
  VpfFrameUploaderContext ctx;
  unique_ptr<VpfFrameUploader> upUploader;
};

class PySurfaceDownloader {
public:
  PySurfaceDownloader(uint32_t width, uint32_t height, Pixel_Format format,
                      uint32_t gpuID)
      : ctx(gpuID, width, height, format) {
    upDownloader.reset(new VpfSurfaceDownloader(ctx));
  }

  bool DownloadSingleSurface(shared_ptr<Surface> pSurface,
                             py::array_t<uint8_t> &frame) {
    VpfSurfaceDownloaderArgs args(pSurface);

    if (!upDownloader->DownloadSingleSurface(args)) {
      throw runtime_error(args.errorMessage);
    } else {
      PyArrayFromPtr<uint8_t>(frame, args.frame, args.frameSize);
    }

    return true;
  }

  inline uint32_t GetWidth() const { return ctx.width; }
  inline uint32_t GetHeight() const { return ctx.height; }
  inline Pixel_Format GetFormat() const { return ctx.format; }

private:
  VpfSurfaceDownloaderContext ctx;
  unique_ptr<VpfSurfaceDownloader> upDownloader;
};

class PySurfaceConverter {
public:
  PySurfaceConverter(uint32_t width, uint32_t height, Pixel_Format srcFormat,
                     Pixel_Format dstFormat, uint32_t gpuID)
      : ctx(gpuID, width, height, srcFormat, dstFormat) {
    upConverter.reset(new VpfSurfaceConverter(ctx));
  }

  inline uint32_t GetWidth() const { return ctx.width; }
  inline uint32_t GetHeight() const { return ctx.height; }
  inline Pixel_Format GetSrcFormat() const { return ctx.srcFormat; }
  inline Pixel_Format GetDstFormat() const { return ctx.dstFormat; }

  shared_ptr<Surface> ConvertSingleSurface(shared_ptr<Surface> pSurface) {
    VpfSurfaceConverterArgs args(pSurface);

    if (!upConverter->ConvertSingleSurface(args)) {
      throw runtime_error(args.errorMessage);
    } else {
      return shared_ptr<Surface>(args.dstSurface->Clone());
    }
  }

private:
  VpfSurfaceConverterContext ctx;
  unique_ptr<VpfSurfaceConverter> upConverter;
};

class PySurfaceResizer {
public:
  PySurfaceResizer(uint32_t width, uint32_t height, Pixel_Format format,
                   uint32_t gpuID)
      : ctx(gpuID, width, height, format) {
    upResizer.reset(new VpfSurfaceResizer(ctx));
  }

  inline uint32_t GetWidth() const { return ctx.width; }
  inline uint32_t GetHeight() const { return ctx.height; }
  inline Pixel_Format GetFormat() const { return ctx.format; }

  shared_ptr<Surface> ResizeSingleSurface(shared_ptr<Surface> pSurface) {
    VpfSurfaceResizerArgs args(pSurface);

    if (!upResizer->ResizeSingleSurface(args)) {
      throw runtime_error(args.errorMessage);
    } else {
      return shared_ptr<Surface>(args.dstSurface->Clone());
    }
  }

private:
  VpfSurfaceResizerContext ctx;
  unique_ptr<VpfSurfaceResizer> upResizer;
};

class PyFFmpegDecoder {
public:
  PyFFmpegDecoder(const string &pathToFile,
                  const map<string, string> &ffmpegOptions)
      : ctx(pathToFile, ffmpegOptions) {
    upDecoder.reset(new VpfFfmpegDecoder(ctx));
  }

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame) {
    VpfFfmpegDecoderArgs args;

    if (!upDecoder->DecodeSingleFrame(args)) {
      throw runtime_error(args.errorMessage);
    } else {
      PyArrayFromPtr(frame, args.frame, args.frameSize);
      return true;
    }
  }

  bool DecodeSingleFrame(py::array_t<uint8_t> &frame,
                         py::array_t<MotionVector> &mv) {
    VpfFfmpegDecoderArgs args;
    args.needMotionVectors = true;

    if (!upDecoder->DecodeSingleFrame(args)) {
      throw runtime_error(args.errorMessage);
    } else {
      PyArrayFromPtr<uint8_t>(frame, args.frame, args.frameSize);
      PyArrayFromPtr<MotionVector>(mv, args.motionVectors,
                                   args.motionVectorsSize);
      return true;
    }
  }

private:
  VpfFfmpegDecoderContext ctx;
  unique_ptr<VpfFfmpegDecoder> upDecoder;
};

PYBIND11_MODULE(PyNvCodec, m) {
  m.doc() = "Python bindings for Nvidia-accelerated video processing";

  PYBIND11_NUMPY_DTYPE_EX(MotionVector, source, "source", w, "w", h, "h", src_x,
                          "src_x", src_y, "src_y", dst_x, "dst_x", dst_y,
                          "dst_y", motion_x, "motion_x", motion_y, "motion_y",
                          motion_scale, "motion_scale");

  py::class_<MotionVector>(m, "MotionVector");

  py::register_exception<HwResetException>(m, "HwResetException");

  py::enum_<Pixel_Format>(m, "PixelFormat")
      .value("Y", Pixel_Format::Y)
      .value("RGB", Pixel_Format::RGB)
      .value("NV12", Pixel_Format::NV12)
      .value("YUV420", Pixel_Format::YUV420)
      .value("RGB_PLANAR", Pixel_Format::RGB_PLANAR)
      .value("BGR", Pixel_Format::BGR)
      .value("YCBCR", Pixel_Format::YCBCR)
      .value("YUV444", Pixel_Format::YUV444)
      .value("UNDEFINED", Pixel_Format::UNDEFINED)
      .export_values();

  py::class_<SurfacePlane, shared_ptr<SurfacePlane>>(m, "SurfacePlane")
      .def("Width", &SurfacePlane::Width)
      .def("Height", &SurfacePlane::Height)
      .def("Pitch", &SurfacePlane::Pitch)
      .def("GpuMem", &SurfacePlane::GpuMem)
      .def("ElemSize", &SurfacePlane::ElemSize)
      .def("HostFrameSize", &SurfacePlane::GetHostMemSize);

  py::class_<Surface, shared_ptr<Surface>>(m, "Surface")
      .def("Width", &Surface::Width, py::arg("planeNumber") = 0U)
      .def("Height", &Surface::Height, py::arg("planeNumber") = 0U)
      .def("Pitch", &Surface::Pitch, py::arg("planeNumber") = 0U)
      .def("Format", &Surface::PixelFormat)
      .def("Empty", &Surface::Empty)
      .def("NumPlanes", &Surface::NumPlanes)
      .def("HostSize", &Surface::HostMemSize)
      .def_static("Make",
                  [](Pixel_Format format, uint32_t newWidth, uint32_t newHeight,
                     int gpuID) {
                    auto pNewSurf = shared_ptr<Surface>(
                        Surface::Make(format, newWidth, newHeight,
                                      CudaResMgr::Instance().GetCtx(gpuID)));
                    return pNewSurf;
                  },
                  py::return_value_policy::take_ownership)
      .def("PlanePtr",
           [](shared_ptr<Surface> self, int planeNumber) {
             auto pPlane = self->GetSurfacePlane(planeNumber);
             return make_shared<SurfacePlane>(*pPlane);
           },
           // Integral part of Surface, only reference it;
           py::arg("planeNumber") = 0U, py::return_value_policy::reference)
      .def("CopyFrom",
           [](shared_ptr<Surface> self, shared_ptr<Surface> other, int gpuID) {
             if (self->PixelFormat() != other->PixelFormat()) {
               throw runtime_error("Surfaces have different pixel formats");
             }

             if (self->Width() != other->Width() ||
                 self->Height() != other->Height()) {
               throw runtime_error("Surfaces have different size");
             }

             CopySurface(self, other, gpuID);
           })
      .def("Clone",
           [](shared_ptr<Surface> self, int gpuID) {
             auto pNewSurf = shared_ptr<Surface>(Surface::Make(
                 self->PixelFormat(), self->Width(), self->Height(),
                 CudaResMgr::Instance().GetCtx(gpuID)));

             CopySurface(self, pNewSurf, gpuID);
             return pNewSurf;
           },
           py::return_value_policy::take_ownership);

  // py::class_<PyNvEncoder>(m, "PyNvEncoder")
  //    .def(py::init<const map<string, string> &, int, Pixel_Format, bool>(),
  //         py::arg("settings"), py::arg("gpu_id"), py::arg("format") = NV12,
  //         py::arg("verbose") = false)
  //    .def("Reconfigure", &PyNvEncoder::Reconfigure, py::arg("settings"),
  //         py::arg("force_idr") = false, py::arg("reset_encoder") = false,
  //         py::arg("verbose") = false)
  //    .def("Width", &PyNvEncoder::Width)
  //    .def("Height", &PyNvEncoder::Height)
  //    .def("Format", &PyNvEncoder::GetPixelFormat)
  //    .def("EncodeSingleSurface", &PyNvEncoder::EncodeSurface,
  //         py::arg("surface"), py::arg("packet"), py::arg("sei_usr_unreg"),
  //         py::arg("sync") = false)
  //    .def("EncodeSingleFrame", &PyNvEncoder::EncodeSingleFrame,
  //         py::arg("frame"), py::arg("packet"), py::arg("sei_usr_unreg"),
  //         py::arg("sync") = false)
  //    .def("Flush", &PyNvEncoder::Flush);

  py::class_<PyFFmpegDecoder>(m, "PyFFmpegDecoder")
      .def(py::init<const string &, const map<string, string> &>())
      .def("DecodeSingleFrame", py::overload_cast<py::array_t<uint8_t> &>(
                                    &PyFFmpegDecoder::DecodeSingleFrame))
      .def("DecodeSingleFrame", py::overload_cast<py::array_t<uint8_t> &,
                                                  py::array_t<MotionVector> &>(
                                    &PyFFmpegDecoder::DecodeSingleFrame));

  py::class_<PacketData>(m, "PacketData")
      .def(py::init<>())
      .def_readonly("pts", &PacketData::pts)
      .def_readonly("dts", &PacketData::dts)
      .def_readonly("pos", &PacketData::pos)
      .def_readonly("duration", &PacketData::duration);

  // py::class_<PyNvDecoder>(m, "PyNvDecoder")
  //    .def(py::init<const string &, int, const map<string, string> &>())
  //    .def(py::init<const string &, int>())
  //    .def("Width", &PyNvDecoder::Width)
  //    .def("Height", &PyNvDecoder::Height)
  //    .def("LastPacketData", &PyNvDecoder::LastPacketData)
  //    .def("Framerate", &PyNvDecoder::Framerate)
  //    .def("Timebase", &PyNvDecoder::Timebase)
  //    .def("Framesize", &PyNvDecoder::Framesize)
  //    .def("Format", &PyNvDecoder::GetPixelFormat)
  //    .def("DecodeSingleSurface", &PyNvDecoder::DecodeSingleSurface,
  //         py::arg("sei"), py::return_value_policy::take_ownership)
  //    .def("DecodeSingleFrame", &PyNvDecoder::DecodeSingleFrame,
  //         py::arg("frame"), py::arg("sei"));

  py::class_<PyFrameUploader>(m, "PyFrameUploader")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
      .def("Width", &PyFrameUploader::GetWidth)
      .def("Height", &PyFrameUploader::GetHeight)
      .def("Format", &PyFrameUploader::GetFormat)
      .def("UploadSingleFrame", &PyFrameUploader::UploadSingleFrame,
           py::return_value_policy::take_ownership);

  py::class_<PySurfaceDownloader>(m, "PySurfaceDownloader")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
      .def("Width", &PySurfaceDownloader::GetWidth)
      .def("Height", &PySurfaceDownloader::GetHeight)
      .def("Format", &PySurfaceDownloader::GetFormat)
      .def("DownloadSingleSurface",
           &PySurfaceDownloader::DownloadSingleSurface);

  // py::class_<PySurfaceConverter>(m, "PySurfaceConverter")
  //    .def(py::init<uint32_t, uint32_t, Pixel_Format, Pixel_Format,
  //    uint32_t>()) .def("Format", &PySurfaceConverter::GetFormat)
  //    .def("Execute", &PySurfaceConverter::Execute,
  //         py::return_value_policy::take_ownership);

  // py::class_<PySurfaceResizer>(m, "PySurfaceResizer")
  //    .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
  //    .def("Format", &PySurfaceResizer::GetFormat)
  //    .def("Execute", &PySurfaceResizer::Execute,
  //         py::return_value_policy::take_ownership);

  m.def("GetNumGpus", &CudaResMgr::GetNumGpus);
}
