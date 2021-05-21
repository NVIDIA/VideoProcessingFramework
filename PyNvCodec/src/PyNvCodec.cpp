/*
 * Copyright 2019 NVIDIA Corporation
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

#include "PyNvCodec.hpp"

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

CudaResMgr::CudaResMgr() {
  ThrowOnCudaError(cuInit(0), __LINE__);

  int nGpu;
  ThrowOnCudaError(cuDeviceGetCount(&nGpu), __LINE__);
  {
    lock_guard<mutex> lock(gContextsMutex);
    for (int i = 0; i < nGpu; i++) {
      CUcontext cuContext = nullptr;

      g_Contexts.push_back(cuContext);
    }
  }
  {
    lock_guard<mutex> lock(gStreamsMutex);
    for (int i = 0; i < nGpu; i++) {
      CUstream cuStream = nullptr;

      g_Streams.push_back(cuStream);
    }
  }
  return;
}

CudaResMgr &CudaResMgr::Instance() {
  static CudaResMgr instance;
  return instance;
}

CUcontext CudaResMgr::GetCtx(size_t idx) {
  if (idx >= GetNumGpus()) {
    return nullptr;
  }
  lock_guard<mutex> lock(gContextsMutex);
  auto &ctx = g_Contexts[idx];
  if (!ctx) {
    CUdevice cuDevice = 0;
    ThrowOnCudaError(cuDeviceGet(&cuDevice, idx), __LINE__);
    ThrowOnCudaError(cuCtxCreate(&ctx, 0, cuDevice), __LINE__);
  }

  return g_Contexts[idx];
}

CUstream CudaResMgr::GetStream(size_t idx) {
  if (idx >= GetNumGpus()) {
    return nullptr;
  }
  lock_guard<mutex> lock(gStreamsMutex);
  auto &str = g_Streams[idx];
  if (!str) {
    auto ctx = GetCtx(idx);
    CudaCtxPush push(ctx);
    ThrowOnCudaError(cuStreamCreate(&str, 0), __LINE__);
  }

  return g_Streams[idx];
}

/* Also a static function as we want to keep all the
 * CUDA stuff within one Python module;
 */
CudaResMgr::~CudaResMgr() {
  stringstream ss;
  try {
    {
      lock_guard<mutex> lock(gStreamsMutex);
      for (auto &cuStream : g_Streams) {
        if (cuStream) {
          ThrowOnCudaError(cuStreamDestroy(cuStream), __LINE__);
        }
      }
      g_Streams.clear();
    }
    {
      lock_guard<mutex> lock(gContextsMutex);
      for (auto &cuContext : g_Contexts) {
        if (cuContext) {
          ThrowOnCudaError(cuCtxDestroy(cuContext), __LINE__);
        }
      }
      g_Contexts.clear();
    }
  } catch (runtime_error &e) {
    cerr << e.what() << endl;
  }

#ifdef TRACK_TOKEN_ALLOCATIONS
  cout << "Checking token allocation counters: ";
  auto res = CheckAllocationCounters();
  cout << (res ? "No leaks dectected" : "Leaks detected") << endl;
#endif
}

size_t CudaResMgr::GetNumGpus() { return Instance().g_Contexts.size(); }

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

PYBIND11_MODULE(PyNvCodec, m) {
  m.doc() = "Python bindings for Nvidia-accelerated video processing";

  PYBIND11_NUMPY_DTYPE_EX(MotionVector, source, "source", w, "w", h, "h", src_x,
                          "src_x", src_y, "src_y", dst_x, "dst_x", dst_y,
                          "dst_y", motion_x, "motion_x", motion_y, "motion_y",
                          motion_scale, "motion_scale");

  py::class_<MotionVector>(m, "MotionVector");
  
  py::register_exception<HwResetException>(m, "HwResetException");

  py::register_exception<CuvidParserException>(m, "CuvidParserException");

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

    py::enum_<ColorSpace>(m, "ColorSpace")
      .value("BT_601", ColorSpace::BT_601)
      .value("BT_709", ColorSpace::BT_709)
      .value("UNSPEC", ColorSpace::UNSPEC)
      .export_values();

    py::enum_<ColorRange>(m, "ColorRange")
        .value("MPEG", ColorRange::MPEG)
        .value("JPEG", ColorRange::JPEG)
        .value("UDEF", ColorRange::UDEF)
        .export_values();

  py::enum_<cudaVideoCodec>(m, "CudaVideoCodec")
      .value("H264", cudaVideoCodec::cudaVideoCodec_H264)
      .value("HEVC", cudaVideoCodec::cudaVideoCodec_HEVC)
      .value("VP9", cudaVideoCodec::cudaVideoCodec_VP9)
      .export_values();

  py::enum_<SeekMode>(m, "SeekMode")
      .value("EXACT_FRAME", SeekMode::EXACT_FRAME)
      .value("PREV_KEY_FRAME", SeekMode::PREV_KEY_FRAME)
      .export_values();

  py::class_<SeekContext, shared_ptr<SeekContext>>(m, "SeekContext")
      .def(py::init<int64_t>(), py::arg("seek_frame"))
      .def(py::init<int64_t, SeekMode>(), py::arg("seek_frame"), py::arg("mode"))
      .def_readwrite("seek_frame", &SeekContext::seek_frame)
      .def_readwrite("mode", &SeekContext::mode)
      .def_readwrite("out_frame_pts", &SeekContext::out_frame_pts)
      .def_readonly("num_frames_decoded", &SeekContext::num_frames_decoded);

  py::class_<PacketData, shared_ptr<PacketData>>(m, "PacketData")
      .def(py::init<>())
      .def_readwrite("pts", &PacketData::pts)
      .def_readwrite("dts", &PacketData::dts)
      .def_readwrite("pos", &PacketData::pos)
      .def_readwrite("poc", &PacketData::poc)
      .def_readwrite("duration", &PacketData::duration);

    py::class_<ColorspaceConversionContext,
             shared_ptr<ColorspaceConversionContext>>(
      m, "ColorspaceConversionContext")
      .def(py::init<>())
      .def(py::init<ColorSpace, ColorRange>(), py::arg("color_space"),
           py::arg("color_range"))
      .def_readwrite("color_space", &ColorspaceConversionContext::color_space)
      .def_readwrite("color_range", &ColorspaceConversionContext::color_range);

  py::class_<SurfacePlane, shared_ptr<SurfacePlane>>(m, "SurfacePlane")
      .def("Width", &SurfacePlane::Width)
      .def("Height", &SurfacePlane::Height)
      .def("Pitch", &SurfacePlane::Pitch)
      .def("GpuMem", &SurfacePlane::GpuMem)
      .def("ElemSize", &SurfacePlane::ElemSize)
      .def("HostFrameSize", &SurfacePlane::GetHostMemSize)
      .def("Import",
           [](shared_ptr<SurfacePlane> self, CUdeviceptr src, uint32_t src_pitch,
              int gpuID) {
             self->Import(src, src_pitch, CudaResMgr::Instance().GetCtx(gpuID),
                          CudaResMgr::Instance().GetStream(gpuID));
           })
      .def("Export",
           [](shared_ptr<SurfacePlane> self, CUdeviceptr dst, uint32_t dst_pitch,
              int gpuID) {
             self->Export(dst, dst_pitch, CudaResMgr::Instance().GetCtx(gpuID),
                          CudaResMgr::Instance().GetStream(gpuID));
           });

  py::class_<Surface, shared_ptr<Surface>>(m, "Surface")
      .def("Width", &Surface::Width, py::arg("planeNumber") = 0U)
      .def("Height", &Surface::Height, py::arg("planeNumber") = 0U)
      .def("Pitch", &Surface::Pitch, py::arg("planeNumber") = 0U)
      .def("Format", &Surface::PixelFormat)
      .def("Empty", &Surface::Empty)
      .def("NumPlanes", &Surface::NumPlanes)
      .def("HostSize", &Surface::HostMemSize)
      .def_static(
          "Make",
          [](Pixel_Format format, uint32_t newWidth, uint32_t newHeight,
             int gpuID) {
            auto pNewSurf = shared_ptr<Surface>(
                Surface::Make(format, newWidth, newHeight,
                              CudaResMgr::Instance().GetCtx(gpuID)));
            return pNewSurf;
          },
          py::return_value_policy::take_ownership)
      .def(
          "PlanePtr",
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
      .def(
          "Clone",
          [](shared_ptr<Surface> self, int gpuID) {
            auto pNewSurf = shared_ptr<Surface>(Surface::Make(
                self->PixelFormat(), self->Width(), self->Height(),
                CudaResMgr::Instance().GetCtx(gpuID)));

            CopySurface(self, pNewSurf, gpuID);
            return pNewSurf;
          },
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>());

  py::class_<PyNvEncoder>(m, "PyNvEncoder")
      .def(py::init<const map<string, string> &, int, Pixel_Format, bool>(),
           py::arg("settings"), py::arg("gpu_id"), py::arg("format") = NV12,
           py::arg("verbose") = false)
      .def("Reconfigure", &PyNvEncoder::Reconfigure, py::arg("settings"),
           py::arg("force_idr") = false, py::arg("reset_encoder") = false,
           py::arg("verbose") = false)
      .def("Width", &PyNvEncoder::Width)
      .def("Height", &PyNvEncoder::Height)
      .def("Format", &PyNvEncoder::GetPixelFormat)
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &, bool, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::arg("sync"), py::arg("append"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::arg("sync"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sync"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &, bool, bool>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"),
           py::arg("append"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &, bool>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             bool>(&PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sync"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"),
           py::call_guard<py::gil_scoped_release>())
      .def("Flush", &PyNvEncoder::Flush, py::arg("packets"),
           py::call_guard<py::gil_scoped_release>())
      .def("FlushSinglePacket", &PyNvEncoder::FlushSinglePacket,
           py::arg("packets"),
           py::call_guard<py::gil_scoped_release>());

  py::class_<PyFfmpegDecoder>(m, "PyFfmpegDecoder")
      .def(py::init<const string &, const map<string, string> &>())
      .def("DecodeSingleFrame", &PyFfmpegDecoder::DecodeSingleFrame)
      .def("GetMotionVectors", &PyFfmpegDecoder::GetMotionVectors,
           py::return_value_policy::move);

  py::class_<PyFFmpegDemuxer>(m, "PyFFmpegDemuxer")
      .def(py::init<const string &>())
      .def(py::init<const string &, const map<string, string> &>())
      .def("DemuxSinglePacket", &PyFFmpegDemuxer::DemuxSinglePacket)
      .def("Width", &PyFFmpegDemuxer::Width)
      .def("Height", &PyFFmpegDemuxer::Height)
      .def("Format", &PyFFmpegDemuxer::Format)
      .def("Framerate", &PyFFmpegDemuxer::Framerate)
      .def("Timebase", &PyFFmpegDemuxer::Timebase)
      .def("Numframes", &PyFFmpegDemuxer::Numframes)
      .def("Codec", &PyFFmpegDemuxer::Codec)
      .def("LastPacketData", &PyFFmpegDemuxer::GetLastPacketData)
      .def("Seek", &PyFFmpegDemuxer::Seek)
      .def("ColorSpace", &PyFFmpegDemuxer::GetColorSpace)
      .def("ColorRange", &PyFFmpegDemuxer::GetColorRange);

  py::class_<PyNvDecoder>(m, "PyNvDecoder")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, cudaVideoCodec,
                    uint32_t>())
      .def(py::init<const string &, int, const map<string, string> &>())
      .def(py::init<const string &, int>())
      .def("Width", &PyNvDecoder::Width)
      .def("Height", &PyNvDecoder::Height)
      .def("ColorSpace", &PyNvDecoder::GetColorSpace)
      .def("ColorRange", &PyNvDecoder::GetColorRange)
      .def("LastPacketData", &PyNvDecoder::LastPacketData)
      .def("Framerate", &PyNvDecoder::Framerate)
      .def("Timebase", &PyNvDecoder::Timebase)
      .def("Framesize", &PyNvDecoder::Framesize)
      .def("Numframes", &PyNvDecoder::Numframes)
      .def("Format", &PyNvDecoder::GetPixelFormat)
      .def("DecodeSingleSurface",
           py::overload_cast<PacketData &>(
               &PyNvDecoder::DecodeSingleSurface),
           py::arg("packet_data"),
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleSurface",
           py::overload_cast<py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSingleSurface),
           py::arg("sei"), py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleSurface",
           py::overload_cast<py::array_t<uint8_t> &, PacketData &>(
               &PyNvDecoder::DecodeSingleSurface),
           py::arg("sei"), py::arg("packet_data"),
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleSurface",
           py::overload_cast<py::array_t<uint8_t> &, SeekContext &>(
               &PyNvDecoder::DecodeSingleSurface),
           py::arg("sei"), py::arg("seek_context"),
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleSurface",
           py::overload_cast<py::array_t<uint8_t> &, SeekContext &, PacketData &>(
               &PyNvDecoder::DecodeSingleSurface),
           py::arg("sei"), py::arg("seek_context"),
           py::arg("packet_data"),
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleSurface",
           py::overload_cast<>(&PyNvDecoder::DecodeSingleSurface),
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleSurface",
           py::overload_cast<SeekContext &>(&PyNvDecoder::DecodeSingleSurface),
           py::arg("seek_context"), py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleSurface",
           py::overload_cast<SeekContext &, PacketData &>(
               &PyNvDecoder::DecodeSingleSurface),
           py::arg("seek_context"), py::arg("packet_data"),
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSurfaceFromPacket",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSurfaceFromPacket),
           py::arg("packet"), py::arg("sei"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSurfaceFromPacket",
           py::overload_cast<py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSurfaceFromPacket),
           py::arg("packet"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"), py::arg("sei"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             PacketData &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"), py::arg("sei"), py::arg("packet_data"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             SeekContext &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"), py::arg("sei"), py::arg("seek_context"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             SeekContext &, PacketData &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"), py::arg("sei"), py::arg("seek_context"),
           py::arg("packet_data"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, PacketData &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"), py::arg("packet_data"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, SeekContext &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"), py::arg("seek_context"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, SeekContext &,
                             PacketData &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"), py::arg("seek_context"), py::arg("packet_data"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeFrameFromPacket",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeFrameFromPacket),
           py::arg("frame"), py::arg("packet"), py::arg("sei"),
           py::call_guard<py::gil_scoped_release>())
      .def("DecodeFrameFromPacket",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeFrameFromPacket),
           py::arg("frame"), py::arg("packet"),
           py::call_guard<py::gil_scoped_release>())
      .def("Numframes", &PyNvDecoder::Numframes)
      .def("FlushSingleSurface", &PyNvDecoder::FlushSingleSurface,
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("FlushSingleFrame", &PyNvDecoder::FlushSingleFrame,
           py::arg("frame"),
           py::call_guard<py::gil_scoped_release>());

  py::class_<PyFrameUploader>(m, "PyFrameUploader")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
      .def("Format", &PyFrameUploader::GetFormat)
      .def("UploadSingleFrame", &PyFrameUploader::UploadSingleFrame,
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>());

  py::class_<PySurfaceDownloader>(m, "PySurfaceDownloader")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
      .def("Format", &PySurfaceDownloader::GetFormat)
      .def("DownloadSingleSurface",
           &PySurfaceDownloader::DownloadSingleSurface,
           py::call_guard<py::gil_scoped_release>());

  py::class_<PySurfaceConverter>(m, "PySurfaceConverter")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, Pixel_Format, uint32_t>())
      .def("Format", &PySurfaceConverter::GetFormat)
      .def("Execute", &PySurfaceConverter::Execute,
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>());

  py::class_<PySurfaceResizer>(m, "PySurfaceResizer")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
      .def("Format", &PySurfaceResizer::GetFormat)
      .def("Execute", &PySurfaceResizer::Execute,
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>());

  m.def("GetNumGpus", &CudaResMgr::GetNumGpus);
}
