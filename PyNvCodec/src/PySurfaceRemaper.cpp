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

PySurfaceRemaper::PySurfaceRemaper(py::array_t<float>& x_map,
                                   py::array_t<float>& y_map, uint32_t width,
                                   uint32_t height, Pixel_Format format,
                                   size_t ctx, size_t str)
    : outputFormat(format)
{
  upRemaper.reset(RemapSurface::Make(x_map.data(), y_map.data(), width, height,
                                     format, (CUcontext)ctx, (CUstream)str));
}

PySurfaceRemaper::PySurfaceRemaper(py::array_t<float>& x_map,
                                   py::array_t<float>& y_map, uint32_t width,
                                   uint32_t height, Pixel_Format format,
                                   uint32_t gpuID)
    : outputFormat(format)
{
  upRemaper.reset(RemapSurface::Make(x_map.data(), y_map.data(), width, height,
                                     format,
                                     CudaResMgr::Instance().GetCtx(gpuID),
                                     CudaResMgr::Instance().GetStream(gpuID)));
}

Pixel_Format PySurfaceRemaper::GetFormat() { return outputFormat; }

shared_ptr<Surface> PySurfaceRemaper::Execute(shared_ptr<Surface> surface)
{
  if (!surface) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  upRemaper->SetInput(surface.get(), 0U);

  if (TASK_EXEC_SUCCESS != upRemaper->Execute()) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  auto pSurface = (Surface*)upRemaper->GetOutput(0U);
  return shared_ptr<Surface>(pSurface ? pSurface->Clone()
                                      : Surface::Make(outputFormat));
}

void Init_PySurfaceRemaper(py::module& m)
{
  py::class_<PySurfaceRemaper>(m, "PySurfaceRemaper")
      .def(py::init<py::array_t<float>&, py::array_t<float>&, uint32_t,
                    uint32_t, Pixel_Format, uint32_t>())
      .def(py::init<py::array_t<float>&, py::array_t<float>&, uint32_t,
                    uint32_t, Pixel_Format, size_t, size_t>())
      .def("Format", &PySurfaceRemaper::GetFormat)
      .def("Execute", &PySurfaceRemaper::Execute,
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>());
}
