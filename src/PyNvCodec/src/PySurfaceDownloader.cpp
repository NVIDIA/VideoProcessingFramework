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

PySurfaceDownloader::PySurfaceDownloader(uint32_t width, uint32_t height,
                                         Pixel_Format format, uint32_t gpu_ID)
{
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  upDownloader.reset(
      CudaDownloadSurface::Make(CudaResMgr::Instance().GetStream(gpu_ID),
                                CudaResMgr::Instance().GetCtx(gpu_ID),
                                surfaceWidth, surfaceHeight, surfaceFormat));
}

PySurfaceDownloader::PySurfaceDownloader(uint32_t width, uint32_t height,
                                         Pixel_Format format, CUcontext ctx,
                                         CUstream str)
{
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  upDownloader.reset(CudaDownloadSurface::Make(str, ctx, surfaceWidth,
                                               surfaceHeight, surfaceFormat));
}

Pixel_Format PySurfaceDownloader::GetFormat() { return surfaceFormat; }

bool PySurfaceDownloader::DownloadSingleSurface(shared_ptr<Surface> surface,
                                                py::array_t<uint8_t>& frame)
{
  upDownloader->SetInput(surface.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  auto* pRawFrame = (Buffer*)upDownloader->GetOutput(0U);
  if (pRawFrame) {
    auto const downloadSize = pRawFrame->GetRawMemSize();
    if (downloadSize != frame.size()) {
      frame.resize({downloadSize}, false);
    }

    memcpy(frame.mutable_data(), pRawFrame->GetRawMemPtr(), downloadSize);
    return true;
  }

  return false;
}

bool PySurfaceDownloader::DownloadSingleSurface(shared_ptr<Surface> surface,
                                                py::array_t<float>& frame)
{
  upDownloader->SetInput(surface.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  auto* pRawFrame = (Buffer*)upDownloader->GetOutput(0U);
  if (pRawFrame) {
    auto const downloadSize = pRawFrame->GetRawMemSize();
    if (downloadSize != frame.size() * sizeof(float)) {
      frame.resize({downloadSize}, false);
    }
    memcpy(frame.mutable_data(), pRawFrame->GetRawMemPtr(), downloadSize);
    return true;
  }

  return false;
}

bool PySurfaceDownloader::DownloadSingleSurface(shared_ptr<Surface> surface,
                                                py::array_t<uint16_t>& frame)
{
  upDownloader->SetInput(surface.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  auto* pRawFrame = (Buffer*)upDownloader->GetOutput(0U);
  if (pRawFrame) {
    auto const downloadSize = pRawFrame->GetRawMemSize();
    if (downloadSize != frame.size() * sizeof(uint16_t)) {
      frame.resize({downloadSize / sizeof(uint16_t)}, false);
    }
    memcpy(frame.mutable_data(), pRawFrame->GetRawMemPtr(), downloadSize);
    return true;
  }

  return false;
}

void Init_PySurfaceDownloader(py::module& m)
{
  py::class_<PySurfaceDownloader>(m, "PySurfaceDownloader")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("gpu_id"),
           R"pbdoc(
        Constructor method.

        :param width: Surface width
        :param height: Surface height
        :param format: Surface pixel format
        :param gpu_id: what GPU does Surface belong to
    )pbdoc")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, size_t, size_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("context"), py::arg("stream"),
           R"pbdoc(
        Constructor method.

        :param width: Surface width
        :param height: Surface height
        :param format: Surface pixel format
        :param context: CUDA context to use for HtoD memcopy
        :param stream: CUDA stream to use for HtoD memcopy
    )pbdoc")
      .def("Format", &PySurfaceDownloader::GetFormat,
           R"pbdoc(
        Get pixel format.
    )pbdoc")
      .def("DownloadSingleSurface",
           py::overload_cast<std::shared_ptr<Surface>, py::array_t<uint8_t>&>(
               &PySurfaceDownloader::DownloadSingleSurface),
           py::arg("surface"), py::arg("frame").noconvert(true),
           R"pbdoc(
        Perform DtoH memcpy.

        :param src: input Surface
        :param frame: output numpy array
        :type frame: numpy.ndarray of type numpy.uint8
        :return: True in case of success False otherwise
        :rtype: Bool
    )pbdoc")
      .def("DownloadSingleSurface",
           py::overload_cast<std::shared_ptr<Surface>, py::array_t<float>&>(
               &PySurfaceDownloader::DownloadSingleSurface),
           py::arg("surface"), py::arg("frame").noconvert(true),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Perform DtoH memcpy.

        :param src: input Surface
        :param frame: output numpy array
        :type frame: numpy.ndarray of type numpy.f
        :return: True in case of success False otherwise
        :rtype: Bool
    )pbdoc")
    .def("DownloadSingleSurface",
           py::overload_cast<std::shared_ptr<Surface>, py::array_t<uint16_t>&>(
               &PySurfaceDownloader::DownloadSingleSurface),
           py::arg("surface"), py::arg("frame").noconvert(true),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Perform DtoH memcpy.

        :param src: input Surface
        :param frame: output numpy array
        :type frame: numpy.ndarray of type numpy.uint16
        :return: True in case of success False otherwise
        :rtype: Bool
    )pbdoc");
}