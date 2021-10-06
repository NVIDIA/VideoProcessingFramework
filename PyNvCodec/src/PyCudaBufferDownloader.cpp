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

PyCudaBufferDownloader::PyCudaBufferDownloader(uint32_t elemSize,
                                               uint32_t numElems,
                                               uint32_t gpu_ID)
{
  elem_size = elemSize;
  num_elems = numElems;

  upDownloader.reset(DownloadCudaBuffer::Make(
      CudaResMgr::Instance().GetStream(gpu_ID),
      CudaResMgr::Instance().GetCtx(gpu_ID), elem_size, num_elems));
}

PyCudaBufferDownloader::PyCudaBufferDownloader(uint32_t elemSize,
                                               uint32_t numElems, CUcontext ctx,
                                               CUstream str)
{
  elem_size = elemSize;
  num_elems = numElems;

  upDownloader.reset(DownloadCudaBuffer::Make(str, ctx, elem_size, num_elems));
}

bool PyCudaBufferDownloader::DownloadSingleCudaBuffer(
    std::shared_ptr<CudaBuffer> buffer, py::array_t<uint8_t>& np_array)
{
  upDownloader->SetInput(buffer.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  auto* pRawBuf = (Buffer*)upDownloader->GetOutput(0U);
  if (pRawBuf) {
    auto const downloadSize = pRawBuf->GetRawMemSize();
    if (downloadSize != np_array.size()) {
      np_array.resize({downloadSize}, false);
    }

    memcpy(np_array.mutable_data(), pRawBuf->GetRawMemPtr(), downloadSize);
    return true;
  }

  return false;
}

void Init_PyCudaBufferDownloader(py::module& m)
{
  py::class_<PyCudaBufferDownloader>(m, "PyCudaBufferDownloader")
      .def(py::init<uint32_t, uint32_t, uint32_t>())
      .def(py::init<uint32_t, uint32_t, size_t, size_t>())
      .def("DownloadSingleCudaBuffer",
           &PyCudaBufferDownloader::DownloadSingleCudaBuffer,
           py::call_guard<py::gil_scoped_release>());
}