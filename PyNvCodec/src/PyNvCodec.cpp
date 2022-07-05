/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Kognia Sports Intelligence
 * Copyright 2021 Videonetics Technology Private Limited
 *
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

    const char* errName = nullptr;
    if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {
      ss << "CUDA error with code " << res << endl;
    } else {
      ss << "CUDA error: " << errName << endl;
    }

    const char* errDesc = nullptr;
    cuGetErrorString(res, &errDesc);

    if (!errDesc) {
      ss << "No error string available" << endl;
    } else {
      ss << errDesc << endl;
    }

    throw runtime_error(ss.str());
  }
};

CudaResMgr::CudaResMgr()
{
  lock_guard<mutex> lock_ctx(CudaResMgr::gInsMutex);

  ThrowOnCudaError(cuInit(0), __LINE__);

  int nGpu;
  ThrowOnCudaError(cuDeviceGetCount(&nGpu), __LINE__);

  for (int i = 0; i < nGpu; i++) {
    CUdevice cuDevice = 0;
    CUcontext cuContext = nullptr;
    g_Contexts.push_back(make_pair(cuDevice, cuContext));

    CUstream cuStream = nullptr;
    g_Streams.push_back(cuStream);
  }
  return;
}

CUcontext CudaResMgr::GetCtx(size_t idx)
{
  lock_guard<mutex> lock_ctx(CudaResMgr::gCtxMutex);

  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto& ctx = g_Contexts[idx];
  if (!ctx.second) {
    CUdevice cuDevice = 0;
    ThrowOnCudaError(cuDeviceGet(&cuDevice, idx), __LINE__);
    ThrowOnCudaError(cuDevicePrimaryCtxRetain(&ctx.second, cuDevice), __LINE__);
  }

  return g_Contexts[idx].second;
}

CUstream CudaResMgr::GetStream(size_t idx)
{
  lock_guard<mutex> lock_ctx(CudaResMgr::gStrMutex);

  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto& str = g_Streams[idx];
  if (!str) {
    auto ctx = GetCtx(idx);
    CudaCtxPush push(ctx);
    ThrowOnCudaError(cuStreamCreate(&str, CU_STREAM_NON_BLOCKING), __LINE__);
  }

  return g_Streams[idx];
}

CudaResMgr::~CudaResMgr()
{
  lock_guard<mutex> ins_lock(CudaResMgr::gInsMutex);
  lock_guard<mutex> ctx_lock(CudaResMgr::gCtxMutex);
  lock_guard<mutex> str_lock(CudaResMgr::gStrMutex);

  stringstream ss;
  try {
    {
      for (auto& cuStream : g_Streams) {
        if (cuStream) {
          cuStreamDestroy(cuStream); // Avoiding CUDA_ERROR_DEINITIALIZED while
                                     // destructing.
        }
      }
      g_Streams.clear();
    }

    {
      for (int i = 0; i < g_Contexts.size(); i++) {
        if (g_Contexts[i].second) {
          cuDevicePrimaryCtxRelease(
              g_Contexts[i].first); // Avoiding CUDA_ERROR_DEINITIALIZED while
                                    // destructing.
        }
      }
      g_Contexts.clear();
    }
  } catch (runtime_error& e) {
    cerr << e.what() << endl;
  }

#ifdef TRACK_TOKEN_ALLOCATIONS
  cout << "Checking token allocation counters: ";
  auto res = CheckAllocationCounters();
  cout << (res ? "No leaks dectected" : "Leaks detected") << endl;
#endif
}

CudaResMgr& CudaResMgr::Instance()
{
  static CudaResMgr instance;
  return instance;
}

size_t CudaResMgr::GetNumGpus() { return Instance().g_Contexts.size(); }

mutex CudaResMgr::gInsMutex;
mutex CudaResMgr::gCtxMutex;
mutex CudaResMgr::gStrMutex;

auto CopyBuffer_Ctx_Str = [](shared_ptr<CudaBuffer> dst,
                             shared_ptr<CudaBuffer> src, CUcontext cudaCtx,
                             CUstream cudaStream) {
  if (dst->GetRawMemSize() != src->GetRawMemSize()) {
    throw runtime_error("Can't copy: buffers have different size.");
  }

  CudaCtxPush ctxPush(cudaCtx);
  ThrowOnCudaError(cuMemcpyDtoDAsync(dst->GpuMem(), src->GpuMem(),
                                     src->GetRawMemSize(), cudaStream));
  ThrowOnCudaError(cuStreamSynchronize(cudaStream), __LINE__);
};

auto CopyBuffer = [](shared_ptr<CudaBuffer> dst, shared_ptr<CudaBuffer> src,
                     int gpuID) {
  auto ctx = CudaResMgr::Instance().GetCtx(gpuID);
  auto str = CudaResMgr::Instance().GetStream(gpuID);
  return CopyBuffer_Ctx_Str(dst, src, ctx, str);
};

auto CopySurface_Ctx_Str = [](shared_ptr<Surface> self,
                              shared_ptr<Surface> other, CUcontext cudaCtx,
                              CUstream cudaStream) {
  CudaCtxPush ctxPush(cudaCtx);

  for (auto plane = 0U; plane < self->NumPlanes(); plane++) {
    auto srcPlanePtr = self->PlanePtr(plane);
    auto dstPlanePtr = other->PlanePtr(plane);

    if (!srcPlanePtr || !dstPlanePtr) {
      break;
    }

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

auto CopySurface = [](shared_ptr<Surface> self, shared_ptr<Surface> other,
                      int gpuID) {
  auto ctx = CudaResMgr::Instance().GetCtx(gpuID);
  auto str = CudaResMgr::Instance().GetStream(gpuID);

  return CopySurface_Ctx_Str(self, other, ctx, str);
};

void Init_PyBufferUploader(py::module&);

void Init_PyCudaBufferDownloader(py::module&);

void Init_PyFrameUploader(py::module&);

void Init_PySurfaceConverter(py::module&);

void Init_PySurfaceDownloader(py::module&);

void Init_PySurfaceResizer(py::module&);

void Init_PySurfaceRemaper(py::module&);

void Init_PyFFMpegDecoder(py::module&);

void Init_PyFFMpegDemuxer(py::module&);

void Init_PyNvDecoder(py::module&);

void Init_PyNvEncoder(py::module&);

PYBIND11_MODULE(PyNvCodec, m)
{
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
      .value("RGB_32F", Pixel_Format::RGB_32F)
      .value("RGB_32F_PLANAR", Pixel_Format::RGB_32F_PLANAR)
      .value("YUV422", Pixel_Format::YUV422)
      .value("P10", Pixel_Format::P10)
      .value("P12", Pixel_Format::P12)
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

  py::enum_<SeekCriteria>(m, "SeekCriteria")
      .value("BY_NUMBER", SeekCriteria::BY_NUMBER)
      .value("BY_TIMESTAMP", SeekCriteria::BY_TIMESTAMP)
      .export_values();

  py::class_<SeekContext, shared_ptr<SeekContext>>(m, "SeekContext")
      .def(py::init<int64_t>(), py::arg("seek_frame"),
           R"pbdoc(
        Constructor method.

        :param seek_frame: number of frame to seek for, starts from 0
    )pbdoc")
      .def(py::init<int64_t, SeekCriteria>(), py::arg("seek_frame"),
           py::arg("seek_criteria"),
           R"pbdoc(
        Constructor method.

        :param seek_frame: number of frame to seek for, starts from 0
        :param seek_criteria: seek by frame number of timestamp
    )pbdoc")
      .def(py::init<int64_t, SeekMode>(), py::arg("seek_frame"),
           py::arg("mode"),
           R"pbdoc(
        Constructor method.

        :param seek_frame: number of frame to seek for, starts from 0
        :param mode: seek to exact frame number or to closest previous key frame
    )pbdoc")
      .def(py::init<int64_t, SeekMode, SeekCriteria>(), py::arg("seek_frame"),
           py::arg("mode"), py::arg("seek_criteria"),
           R"pbdoc(
        Constructor method.

        :param seek_frame: number of frame to seek for, starts from 0
        :param mode: seek to exact frame number or to closest previous key frame
        :param seek_criteria: seek by frame number of timestamp
    )pbdoc")
      .def_readwrite("seek_frame", &SeekContext::seek_frame,
                     R"pbdoc(
        Frame number or timestamp depending on mode.
    )pbdoc")
      .def_readwrite("mode", &SeekContext::mode,
                     R"pbdoc(
        Seek mode: by frame number or timestamp
    )pbdoc")
      .def_readwrite("out_frame_pts", &SeekContext::out_frame_pts,
                     R"pbdoc(
        PTS of frame decoded after seek.
    )pbdoc")
      .def_readonly("num_frames_decoded", &SeekContext::num_frames_decoded,
                    R"pbdoc(
        Number of frames, decoded if seek was done to closest previous key frame.
    )pbdoc");

  py::class_<PacketData, shared_ptr<PacketData>>(m, "PacketData")
      .def(py::init<>())
      .def_readwrite("pts", &PacketData::pts)
      .def_readwrite("dts", &PacketData::dts)
      .def_readwrite("pos", &PacketData::pos)
      .def_readwrite("bsl", &PacketData::bsl)
      .def_readwrite("duration", &PacketData::duration);

  py::class_<ColorspaceConversionContext,
             shared_ptr<ColorspaceConversionContext>>(
      m, "ColorspaceConversionContext")
      .def(py::init<>())
      .def(py::init<ColorSpace, ColorRange>(), py::arg("color_space"),
           py::arg("color_range"))
      .def_readwrite("color_space", &ColorspaceConversionContext::color_space)
      .def_readwrite("color_range", &ColorspaceConversionContext::color_range);

  py::class_<CudaBuffer, shared_ptr<CudaBuffer>>(m, "CudaBuffer")
      .def("GetRawMemSize", &CudaBuffer::GetRawMemSize,
           R"pbdoc(
        Get size of buffer in bytes
    )pbdoc")
      .def("GetNumElems", &CudaBuffer::GetNumElems,
           R"pbdoc(
        Get number of elements in buffer
    )pbdoc")
      .def("GetElemSize", &CudaBuffer::GetElemSize,
           R"pbdoc(
        Get size of single element in bytes
    )pbdoc")
      .def("GpuMem", &CudaBuffer::GpuMem,
           R"pbdoc(
        Get CUdeviceptr of memory allocation
    )pbdoc")
      .def("Clone", &CudaBuffer::Clone, py::return_value_policy::take_ownership,
           R"pbdoc(
        Deep copy = CUDA mem alloc + CUDA mem copy
    )pbdoc")
      .def(
          "CopyFrom",
          [](shared_ptr<CudaBuffer> self, shared_ptr<CudaBuffer> other,
             size_t ctx, size_t str) {
            CopyBuffer_Ctx_Str(self, other, (CUcontext)ctx, (CUstream)str);
          },
          py::arg("other"), py::arg("context"), py::arg("stream"),
          R"pbdoc(
        Copy content of another CudaBuffer into this CudaBuffer

        :param other: other CudaBuffer
        :param context: CUDA context to use
        :param stream: CUDA stream to use
    )pbdoc")
      .def(
          "CopyFrom",
          [](shared_ptr<CudaBuffer> self, shared_ptr<CudaBuffer> other,
             int gpuID) { CopyBuffer(self, other, gpuID); },
          py::arg("other"), py::arg("gpu_id"),
          R"pbdoc(
        Copy content of another CudaBuffer into this CudaBuffer

        :param other: other CudaBuffer
        :param gpu_id: GPU to use for memcopy
    )pbdoc")
      .def_static(
          "Make",
          [](uint32_t elem_size, uint32_t num_elems, int gpuID) {
            auto pNewBuf = shared_ptr<CudaBuffer>(CudaBuffer::Make(
                elem_size, num_elems, CudaResMgr::Instance().GetCtx(gpuID)));
            return pNewBuf;
          },
          py::arg("elem_size"), py::arg("num_elems"), py::arg("gpu_id"),
          py::return_value_policy::take_ownership,
          R"pbdoc(
        Constructor method

        :param elem_size: single buffer element size in bytes
        :param num_elems: number of elements in buffer
        :param gpu_id: GPU to use for memcopy
    )pbdoc");

  py::class_<SurfacePlane, shared_ptr<SurfacePlane>>(m, "SurfacePlane")
      .def("Width", &SurfacePlane::Width,
           R"pbdoc(
        Get width in pixels
    )pbdoc")
      .def("Height", &SurfacePlane::Height,
           R"pbdoc(
        Get height in pixels
    )pbdoc")
      .def("Pitch", &SurfacePlane::Pitch,
           R"pbdoc(
        Get pitch in bytes
    )pbdoc")
      .def("GpuMem", &SurfacePlane::GpuMem,
           R"pbdoc(
        Get CUdeviceptr of memory object
    )pbdoc")
      .def("ElemSize", &SurfacePlane::ElemSize,
           R"pbdoc(
        Get element size in bytes
    )pbdoc")
      .def("HostFrameSize", &SurfacePlane::GetHostMemSize,
           R"pbdoc(
        Get amount of host memory needed to store this SurfacePlane
    )pbdoc")
      .def(
          "Import",
          [](shared_ptr<SurfacePlane> self, CUdeviceptr src, uint32_t src_pitch,
             int gpuID) {
            self->Import(src, src_pitch, CudaResMgr::Instance().GetCtx(gpuID),
                         CudaResMgr::Instance().GetStream(gpuID));
          },
          py::arg("src"), py::arg("src_pitch"), py::arg("gpu_id"),
          R"pbdoc(
        Import from another SurfacePlane

        :param src: source SurfacePlane
        :param src_pitch: source SurfacePlane pitch in bytes
        :param gpu_id: GPU to use
    )pbdoc")
      .def(
          "Import",
          [](shared_ptr<SurfacePlane> self, CUdeviceptr src, uint32_t src_pitch,
             size_t ctx, size_t str) {
            self->Import(src, src_pitch, (CUcontext)ctx, (CUstream)str);
          },
          py::arg("src"), py::arg("src_pitch"), py::arg("context"),
          py::arg("stream"),
          R"pbdoc(
        Import from another SurfacePlane

        :param src: source SurfacePlane
        :param src_pitch: source SurfacePlane pitch in bytes
        :param context: CUDA context to use
        :param stream: CUDA stream to use
    )pbdoc")
      .def(
          "Export",
          [](shared_ptr<SurfacePlane> self, CUdeviceptr dst, uint32_t dst_pitch,
             int gpuID) {
            self->Export(dst, dst_pitch, CudaResMgr::Instance().GetCtx(gpuID),
                         CudaResMgr::Instance().GetStream(gpuID));
          },
          py::arg("dst"), py::arg("dst_pitch"), py::arg("gpu_id"),
          R"pbdoc(
        Export to another SurfacePlane

        :param dst: destination SurfacePlane
        :param dst_pitch: destination SurfacePlane pitch in bytes
        :param gpu_id: GPU to use
    )pbdoc")
      .def(
          "Export",
          [](shared_ptr<SurfacePlane> self, CUdeviceptr dst, uint32_t dst_pitch,
             size_t ctx, size_t str) {
            self->Export(dst, dst_pitch, (CUcontext)ctx, (CUstream)str);
          },
          py::arg("dst"), py::arg("dst_pitch"), py::arg("context"),
          py::arg("stream"),
          R"pbdoc(
        Export to another SurfacePlane

        :param dst: destination SurfacePlane
        :param dst_pitch: destination SurfacePlane pitch in bytes
        :param context: CUDA context to use
        :param stream: CUDA stream to use
    )pbdoc");

  py::class_<Surface, shared_ptr<Surface>>(m, "Surface")
      .def("Width", &Surface::Width, py::arg("plane") = 0U,
           R"pbdoc(
        Width in pixels.
        Please note that different SurfacePlane may have different dimensions
        depending on pixel format.

        :param plane: SurfacePlane index
    )pbdoc")
      .def("Height", &Surface::Height, py::arg("plane") = 0U,
           R"pbdoc(
        Height in pixels.
        Please note that different SurfacePlane may have different dimensions
        depending on pixel format.

        :param plane: SurfacePlane index
    )pbdoc")
      .def("Pitch", &Surface::Pitch, py::arg("plane") = 0U,
           R"pbdoc(
        Pitch in bytes.
        Please note that different SurfacePlane may have different dimensions
        depending on pixel format.

        :param plane: SurfacePlane index
    )pbdoc")
      .def("Format", &Surface::PixelFormat,
           R"pbdoc(
        Get pixel format
    )pbdoc")
      .def("Empty", &Surface::Empty,
           R"pbdoc(
        Tell if Surface plane has memory allocated or it's empty inside.
    )pbdoc")
      .def("NumPlanes", &Surface::NumPlanes,
           R"pbdoc(
        Number of SurfacePlanes
    )pbdoc")
      .def("HostSize", &Surface::HostMemSize,
           R"pbdoc(
        Amount of memory in bytes which is needed for DtoH memcopy.
    )pbdoc")
      .def_static(
          "Make",
          [](Pixel_Format format, uint32_t newWidth, uint32_t newHeight,
             int gpuID) {
            auto pNewSurf = shared_ptr<Surface>(
                Surface::Make(format, newWidth, newHeight,
                              CudaResMgr::Instance().GetCtx(gpuID)));
            return pNewSurf;
          },
          py::arg("format"), py::arg("width"), py::arg("height"),
          py::arg("gpu_id"), py::return_value_policy::take_ownership,
          R"pbdoc(
        Constructor method.

        :param format: target pixel format
        :param width: width in pixels
        :param height: height in pixels
        :param gpu_id: GPU to use
    )pbdoc")
      .def_static(
          "Make",
          [](Pixel_Format format, uint32_t newWidth, uint32_t newHeight,
             size_t ctx) {
            auto pNewSurf = shared_ptr<Surface>(
                Surface::Make(format, newWidth, newHeight, (CUcontext)ctx));
            return pNewSurf;
          },
          py::arg("format"), py::arg("width"), py::arg("height"),
          py::arg("context"), py::return_value_policy::take_ownership,
          R"pbdoc(
        Constructor method.

        :param format: target pixel format
        :param width: width in pixels
        :param height: height in pixels
        :param context: CUDA contet to use
    )pbdoc")
      .def(
          "PlanePtr",
          [](shared_ptr<Surface> self, int plane) {
            auto pPlane = self->GetSurfacePlane(plane);
            return make_shared<SurfacePlane>(*pPlane);
          },
          // Integral part of Surface, only reference it;
          py::arg("plane") = 0U, py::return_value_policy::reference,
          R"pbdoc(
        Get SurfacePlane reference

        :param plane: SurfacePlane index
    )pbdoc")
      .def(
          "CopyFrom",
          [](shared_ptr<Surface> self, shared_ptr<Surface> other, int gpuID) {
            if (self->PixelFormat() != other->PixelFormat()) {
              throw runtime_error("Surfaces have different pixel formats");
            }

            if (self->Width() != other->Width() ||
                self->Height() != other->Height()) {
              throw runtime_error("Surfaces have different size");
            }

            CopySurface(self, other, gpuID);
          },
          py::arg("other"), py::arg("gpu_id"), R"pbdoc(
        Perform DtoD memcopy

        :param other: other Surface
        :param gpu_id: GPU to use
    )pbdoc")
      .def(
          "CopyFrom",
          [](shared_ptr<Surface> self, shared_ptr<Surface> other, size_t ctx,
             size_t str) {
            if (self->PixelFormat() != other->PixelFormat()) {
              throw runtime_error("Surfaces have different pixel formats");
            }

            if (self->Width() != other->Width() ||
                self->Height() != other->Height()) {
              throw runtime_error("Surfaces have different size");
            }

            CopySurface_Ctx_Str(self, other, (CUcontext)ctx, (CUstream)str);
          },
          py::arg("other"), py::arg("context"), py::arg("stream"),
          R"pbdoc(
        Perform DtoD memcopy

        :param other: other Surface
        :param context: CUDA contet to use
        :param stream: desc
    )pbdoc")
      .def(
          "Clone",
          [](shared_ptr<Surface> self, int gpuID) {
            auto pNewSurf = shared_ptr<Surface>(Surface::Make(
                self->PixelFormat(), self->Width(), self->Height(),
                CudaResMgr::Instance().GetCtx(gpuID)));

            CopySurface(self, pNewSurf, gpuID);
            return pNewSurf;
          },
          py::arg("gpu_id"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Deep copy = CUDA mem alloc + CUDA mem copy

        :param gpu_id: GPU to use
    )pbdoc")
      .def(
          "Clone",
          [](shared_ptr<Surface> self, size_t ctx, size_t str) {
            auto pNewSurf = shared_ptr<Surface>(
                Surface::Make(self->PixelFormat(), self->Width(),
                              self->Height(), (CUcontext)ctx));

            CopySurface_Ctx_Str(self, pNewSurf, (CUcontext)ctx, (CUstream)str);
            return pNewSurf;
          },
          py::arg("context"), py::arg("stream"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Deep copy = CUDA mem alloc + CUDA mem copy

        :param context: CUDA contet to use
        :param stream: CUDA stream to use
    )pbdoc")
      .def(
          "Crop",
          [](shared_ptr<Surface> self, uint32_t x, uint32_t y, uint32_t w,
             uint32_t h, int gpuID) {
            auto ctx = CudaResMgr::Instance().GetCtx(gpuID);
            auto str = CudaResMgr::Instance().GetStream(gpuID);
            auto cropped_surf = Surface::Make(self->PixelFormat(), w, h, ctx);
            self->Export(*cropped_surf, ctx, str, x, y, w, h, 0U, 0U);
            return cropped_surf;
          },
          py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"),
          py::arg("gpu_id"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Crop = select ROI + CUDA mem alloc + CUDA mem copy

        :param x: ROI top left X coordinate
        :param y: ROI top left Y coordinate
        :param w: ROI width in pixels
        :param h: ROI height in pixels
        :param gpu_id: GPU to use
    )pbdoc")
      .def(
          "Crop",
          [](shared_ptr<Surface> self, uint32_t x, uint32_t y, uint32_t w,
             uint32_t h, size_t context, size_t stream) {
            auto ctx = (CUcontext)context;
            auto str = (CUstream)stream;
            auto cropped_surf = Surface::Make(self->PixelFormat(), w, h, ctx);
            self->Export(*cropped_surf, ctx, str, x, y, w, h, 0U, 0U);
            return cropped_surf;
          },
          py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"),
          py::arg("context"), py::arg("stream"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Crop = select ROI + CUDA mem alloc + CUDA mem copy

        :param x: ROI top left X coordinate
        :param y: ROI top left Y coordinate
        :param w: ROI width in pixels
        :param h: ROI height in pixels
        :param context: CUDA contet to use
        :param stream: CUDA stream to use
    )pbdoc");

  Init_PyFFMpegDecoder(m);

  Init_PyFFMpegDemuxer(m);

  Init_PyNvDecoder(m);

  Init_PyNvEncoder(m);

  Init_PyFrameUploader(m);

  Init_PyBufferUploader(m);

  Init_PySurfaceDownloader(m);

  Init_PyCudaBufferDownloader(m);

  Init_PySurfaceConverter(m);

  Init_PySurfaceResizer(m);

  Init_PySurfaceRemaper(m);

  m.def("GetNumGpus", &CudaResMgr::GetNumGpus, R"pbdoc(
        Get number of available GPUs.
    )pbdoc");

  m.def("GetNvencParams", &GetNvencInitParams, R"pbdoc(
        Get list of params PyNvEncoder can be initialized with.
    )pbdoc");

  m.doc() = R"pbdoc(
        PyNvCodec
        ----------
        .. currentmodule:: PyNvCodec
        .. autosummary::
           :toctree: _generate

           GetNumGpus
           GetNvencParams
           PySurfaceResizer
           PySurfaceRemaper
           PySurfaceDownloader
           PySurfaceConverter
           PyNvEncoder
           PyNvDecoder
           PyFrameUploader
           PyFFmpegDemuxer
           PyFfmpegDecoder
           PyCudaBufferDownloader
           PyBufferUploader
           SeekContext
           CudaBuffer
           SurfacePlane
           Surface

    )pbdoc";
}
