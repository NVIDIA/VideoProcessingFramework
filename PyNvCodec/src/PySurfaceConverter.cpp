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

PySurfaceConverter::PySurfaceConverter(uint32_t width, uint32_t height,
                                       Pixel_Format inFormat,
                                       Pixel_Format outFormat, uint32_t gpuID)
    : outputFormat(outFormat)
{
  upConverter.reset(ConvertSurface::Make(
      width, height, inFormat, outFormat, CudaResMgr::Instance().GetCtx(gpuID),
      CudaResMgr::Instance().GetStream(gpuID)));
  upCtxBuffer.reset(Buffer::MakeOwnMem(sizeof(ColorspaceConversionContext)));
}

PySurfaceConverter::PySurfaceConverter(uint32_t width, uint32_t height,
                                       Pixel_Format inFormat,
                                       Pixel_Format outFormat, CUcontext ctx,
                                       CUstream str)
    : outputFormat(outFormat)
{
  upConverter.reset(
      ConvertSurface::Make(width, height, inFormat, outFormat, ctx, str));
  upCtxBuffer.reset(Buffer::MakeOwnMem(sizeof(ColorspaceConversionContext)));
}

shared_ptr<Surface>
PySurfaceConverter::Execute(shared_ptr<Surface> surface,
                            shared_ptr<ColorspaceConversionContext> context)
{
  if (!surface) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  upConverter->ClearInputs();

  upConverter->SetInput(surface.get(), 0U);

  if (context) {
    upCtxBuffer->CopyFrom(sizeof(ColorspaceConversionContext), context.get());
    upConverter->SetInput((Token*)upCtxBuffer.get(), 1U);
  }

  if (TASK_EXEC_SUCCESS != upConverter->Execute()) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  auto pSurface = (Surface*)upConverter->GetOutput(0U);
  return shared_ptr<Surface>(pSurface ? pSurface->Clone()
                                      : Surface::Make(outputFormat));
}

Pixel_Format PySurfaceConverter::GetFormat() { return outputFormat; }

PySurfaceResizer::PySurfaceResizer(uint32_t width, uint32_t height,
                                   Pixel_Format format, uint32_t gpuID)
    : outputFormat(format)
{
  upResizer.reset(ResizeSurface::Make(width, height, format,
                                      CudaResMgr::Instance().GetCtx(gpuID),
                                      CudaResMgr::Instance().GetStream(gpuID)));
}

void Init_PySurfaceConverter(py::module& m)
{
  py::class_<PySurfaceConverter>(m, "PySurfaceConverter")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, Pixel_Format, uint32_t>())
      .def(py::init<uint32_t, uint32_t, Pixel_Format, Pixel_Format, size_t,
                    size_t>())
      .def("Format", &PySurfaceConverter::GetFormat)
      .def("Execute", &PySurfaceConverter::Execute,
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>());
}