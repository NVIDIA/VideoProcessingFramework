/*
 * Copyright 2022 NVIDIA Corporation
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
#include <sstream>
#include <map>

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

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

string ToString(Pixel_Format fmt)
{
  static map<Pixel_Format, string> fmt_names = {
    {Y,               "Y"},
    {RGB,             "RGB"},
    {NV12,            "NV12"},
    {YUV420,          "YUV420"},
    {RGB_PLANAR,      "RGB_PLANAR"},
    {BGR,             "BGR"},
    {YCBCR,           "YCBCR"},
    {YUV444,          "YUV444"},
    {RGB_32F,         "RGB_32F"},
    {RGB_32F_PLANAR,  "RGB_32F_PLANAR"},
    {YUV422,          "YUV422"},
    {P10,             "P10"},
    {P12,             "P12"},
  };

  auto it = fmt_names.find(fmt);
  if (fmt_names.end() != it) {
    return it->second;
  } else {
    return string("UNDEFINED");
  }
};

string ToString(SurfacePlane* self, int space = 0)
{
  if (!self) {
    return string();
  }

  stringstream spacer;
  for (int i = 0; i < space; i++) {
    spacer << " ";
  }

  stringstream ss;
  ss << spacer.str() << "Owns mem:  " << self->OwnMemory() << "\n";
  ss << spacer.str() << "Width:     " << self->Width() << "\n";
  ss << spacer.str() << "Height:    " << self->Height() << "\n";
  ss << spacer.str() << "Pitch:     " << self->Pitch() << "\n";
  ss << spacer.str() << "Elem size: " << self->ElemSize() << "\n";
  ss << spacer.str() << "Cuda ctx:  " << self->GetContext() << "\n";
  ss << spacer.str() << "CUDA ptr:  " << self->GpuMem() << "\n";

  return ss.str();
}

string ToString(Surface* self)
{
  if (!self) {
    return string();
  }

  stringstream ss;
  ss << "Width:            " << self->Width() << "\n";
  ss << "Height:           " << self->Height() << "\n";
  ss << "Format:           " << ToString(self->PixelFormat()) << "\n";
  ss << "Pitch:            " << self->Pitch() << "\n";
  ss << "Elem size(bytes): " << self->ElemSize() << "\n";

  for (int i = 0; i < self->NumPlanes() && self->GetSurfacePlane(i); i++) {
    ss << "Plane " << i << "\n";
    ss << ToString(self->GetSurfacePlane(i), 2) << "\n";
  }

  return ss.str();
}

void Init_PySurface(py::module& m)
{
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
    )pbdoc")
      .def("__repr__",
           [](shared_ptr<SurfacePlane> self) { return ToString(self.get()); });

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
      .def("OwnMemory", &Surface::OwnMemory,
           R"pbdoc(
        Return True if Surface owns memory, False if it only references actual
        memory allocation but doesn't own it.
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
          [](shared_ptr<Surface> self) {
            auto pNewSurf = shared_ptr<Surface>(
                Surface::Make(self->PixelFormat(), self->Width(),
                              self->Height(), self->Context()));

            CopySurface_Ctx_Str(self, pNewSurf, self->Context(), CUstream(0));
            return pNewSurf;
          },
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Deep copy = CUDA mem alloc + CUDA mem copy.
        Will use CUDA context associated with CUDA memory object and default
        CUDA stream.
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
    )pbdoc")
      .def("__repr__",
           [](shared_ptr<Surface> self) { return ToString(self.get()); });
}