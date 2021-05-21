/*
 * Copyright 2021 NVIDIA Corporation
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

PySurfaceDownloader::PySurfaceDownloader(uint32_t width, uint32_t height,
                                         Pixel_Format format, uint32_t gpu_ID) {
  gpuID = gpu_ID;
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  upDownloader.reset(
      CudaDownloadSurface::Make(CudaResMgr::Instance().GetStream(gpuID),
                                CudaResMgr::Instance().GetCtx(gpuID),
                                surfaceWidth, surfaceHeight, surfaceFormat));
}

Pixel_Format PySurfaceDownloader::GetFormat() { return surfaceFormat; }

bool PySurfaceDownloader::DownloadSingleSurface(shared_ptr<Surface> surface,
                                                py::array_t<uint8_t> &frame) {
  upDownloader->SetInput(surface.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  auto *pRawFrame = (Buffer *)upDownloader->GetOutput(0U);
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