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

PyFrameUploader::PyFrameUploader(uint32_t width, uint32_t height,
                                 Pixel_Format format, uint32_t gpu_ID)
{
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  uploader.reset(CudaUploadFrame::Make(CudaResMgr::Instance().GetStream(gpu_ID),
                                       CudaResMgr::Instance().GetCtx(gpu_ID),
                                       surfaceWidth, surfaceHeight,
                                       surfaceFormat));
}

PyFrameUploader::PyFrameUploader(uint32_t width, uint32_t height,
                                 Pixel_Format format, CUcontext ctx,
                                 CUstream str)
{
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  uploader.reset(CudaUploadFrame::Make(str, ctx, surfaceWidth, surfaceHeight,
                                       surfaceFormat));
}

Pixel_Format PyFrameUploader::GetFormat() { return surfaceFormat; }

shared_ptr<Surface>
PyFrameUploader::UploadSingleFrame(py::array_t<uint8_t>& frame)
{
  /* Upload to GPU;
   */
  auto pRawFrame = Buffer::Make(frame.size(), frame.mutable_data());
  uploader->SetInput(pRawFrame, 0U);
  auto res = uploader->Execute();
  delete pRawFrame;

  if (TASK_EXEC_FAIL == res) {
    throw runtime_error("Error uploading frame to GPU");
  }

  /* Get surface;
   */
  auto pSurface = (Surface*)uploader->GetOutput(0U);
  if (!pSurface) {
    throw runtime_error("Error uploading frame to GPU");
  }

  return shared_ptr<Surface>(pSurface->Clone());
}

shared_ptr<Surface>
PyFrameUploader::UploadSingleFrame(py::array_t<float>& frame)
{
  /* Upload to GPU;
   */
  auto pRawFrame =
      Buffer::Make(frame.size() * sizeof(float), frame.mutable_data());
  uploader->SetInput(pRawFrame, 0U);
  auto res = uploader->Execute();
  delete pRawFrame;

  if (TASK_EXEC_FAIL == res) {
    throw runtime_error("Error uploading frame to GPU");
  }

  /* Get surface;
   */
  auto pSurface = (Surface*)uploader->GetOutput(0U);
  if (!pSurface) {
    throw runtime_error("Error uploading frame to GPU");
  }

  return shared_ptr<Surface>(pSurface->Clone());
}

void Init_PyFrameUploader(py::module& m)
{
  py::class_<PyFrameUploader>(m, "PyFrameUploader")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
      .def(py::init<uint32_t, uint32_t, Pixel_Format, size_t, size_t>())
      .def("Format", &PyFrameUploader::GetFormat)
      .def("UploadSingleFrame",
           py::overload_cast<py::array_t<uint8_t>&>(
               &PyFrameUploader::UploadSingleFrame),
           py::arg("frame").noconvert(true),
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>())
      .def("UploadSingleFrame",
           py::overload_cast<py::array_t<float>&>(
               &PyFrameUploader::UploadSingleFrame),
           py::arg("frame").noconvert(true),
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>());
}