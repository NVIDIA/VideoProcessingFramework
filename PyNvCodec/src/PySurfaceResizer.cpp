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

PySurfaceResizer::PySurfaceResizer(uint32_t width, uint32_t height,
                                   Pixel_Format format, CUcontext ctx,
                                   CUstream str)
    : outputFormat(format)
{
  upResizer.reset(ResizeSurface::Make(width, height, format, ctx, str));
}

PySurfaceResizer::PySurfaceResizer(uint32_t width, uint32_t height,
                                   Pixel_Format format, uint32_t gpuID)
    : outputFormat(format)
{
  upResizer.reset(ResizeSurface::Make(width, height, format,
                                      CudaResMgr::Instance().GetCtx(gpuID),
                                      CudaResMgr::Instance().GetStream(gpuID)));
}

Pixel_Format PySurfaceResizer::GetFormat() { return outputFormat; }

shared_ptr<Surface> PySurfaceResizer::Execute(shared_ptr<Surface> surface)
{
  if (!surface) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  upResizer->SetInput(surface.get(), 0U);

  if (TASK_EXEC_SUCCESS != upResizer->Execute()) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  auto pSurface = (Surface*)upResizer->GetOutput(0U);
  return shared_ptr<Surface>(pSurface ? pSurface->Clone()
                                      : Surface::Make(outputFormat));
}

void Init_PySurfaceResizer(py::module& m)
{
  py::class_<PySurfaceResizer>(m, "PySurfaceResizer")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("gpu_id"),
           R"pbdoc(
        Constructor method.

        :param width: target Surface width
        :param height: target Surface height
        :param format: target Surface pixel format
        :param gpu_id: what GPU to run resize on
    )pbdoc")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, size_t, size_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("context"), py::arg("stream"),
           R"pbdoc(
        Constructor method.

        :param width: target Surface width
        :param height: target Surface height
        :param format: target Surface pixel format
        :param context: CUDA context to use for resize
        :param stream: CUDA stream to use for resize
    )pbdoc")
      .def("Format", &PySurfaceResizer::GetFormat, R"pbdoc(
        Get pixel format.
    )pbdoc")
      .def("Execute", &PySurfaceResizer::Execute, py::arg("src"),
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Resize input Surface.

        :param src: input Surface. Must be of same format class instance was created with.
        :return: Surface of dimensions equal to given to ctor
        :rtype: PyNvCodec.Surface
    )pbdoc");
}