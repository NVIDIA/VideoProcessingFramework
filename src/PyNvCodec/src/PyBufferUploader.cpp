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

PyBufferUploader::PyBufferUploader(uint32_t elemSize, uint32_t numElems,
                                   uint32_t gpu_ID)
{
  elem_size = elemSize;
  num_elems = numElems;

  uploader.reset(UploadBuffer::Make(CudaResMgr::Instance().GetStream(gpu_ID),
                                    CudaResMgr::Instance().GetCtx(gpu_ID),
                                    elem_size, num_elems));
}

PyBufferUploader::PyBufferUploader(uint32_t elemSize, uint32_t numElems,
                                   CUcontext ctx, CUstream str)
{
  elem_size = elemSize;
  num_elems = numElems;

  uploader.reset(UploadBuffer::Make(str, ctx, elem_size, num_elems));
}

shared_ptr<CudaBuffer>
PyBufferUploader::UploadSingleBuffer(py::array_t<uint8_t>& frame)
{
  auto pRawBuf = Buffer::Make(frame.size(), frame.mutable_data());
  uploader->SetInput(pRawBuf, 0U);
  auto res = uploader->Execute();
  delete pRawBuf;

  if (TASK_EXEC_FAIL == res)
    throw runtime_error("Error uploading frame to GPU");

  auto pCudaBuffer = (CudaBuffer*)uploader->GetOutput(0U);
  if (!pCudaBuffer)
    throw runtime_error("Error uploading frame to GPU");

  return shared_ptr<CudaBuffer>(pCudaBuffer->Clone());
}

void Init_PyBufferUploader(py::module& m)
{
  py::class_<PyBufferUploader>(m, "PyBufferUploader")
      .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("elem_size"),
           py::arg("num_elems"), py::arg("gpu_id"),
           R"pbdoc(
        Constructor method.

        :param elem_size: single buffer element size in bytes
        :param num_elems: number of elements in buffer
        :param gpu_id: GPU to use for memcopy
    )pbdoc")
      .def(py::init<uint32_t, uint32_t, size_t, size_t>(), py::arg("elem_size"),
           py::arg("num_elems"), py::arg("context"), py::arg("stream"),
           R"pbdoc(
        Constructor method.

        :param elem_size: single buffer element size in bytes
        :param num_elems: number of elements in buffer
        :param context: CUDA context to use
        :param stream: CUDA stream to use
    )pbdoc")
      .def("UploadSingleBuffer", &PyBufferUploader::UploadSingleBuffer,
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>(), py::arg("array"),
           R"pbdoc(
        Perform HtoD memcopy.

        :param array: output numpy array
        :return: True in case of success, False otherwise
    )pbdoc");
}