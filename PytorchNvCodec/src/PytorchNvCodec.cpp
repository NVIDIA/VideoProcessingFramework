/*
 * Copyright 2020 NVIDIA Corporation
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

torch::Tensor makefromDevicePtrUint8(CUdeviceptr ptr, uint32_t width,
                                     uint32_t height, uint32_t pitch,
                                     uint32_t elem_size_bytes) {
  if (elem_size_bytes != 1) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << ": only torch::kUInt8 data type is supported";
    throw std::runtime_error(ss.str());
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA);

  torch::Tensor tensor = torch::full({height, width}, 128, options);

  uint8_t *devMem = tensor.data_ptr<uint8_t>();
  if (!devMem) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << ": Pytorch tensor doesn't have data ptr.";

    throw std::runtime_error(ss.str());
  }

  if (!ptr) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << ": Video frame has void CUDA device ptr.";

    throw std::runtime_error(ss.str());
  }

  auto res = cudaMemcpy2D((void *)devMem, width, (const void *)ptr, pitch,
                          width, height, cudaMemcpyDeviceToDevice);
  if (cudaSuccess != res) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << ": failed to copy data to tensor. CUDA error code: ";
    ss << res;

    throw std::runtime_error(ss.str());
  }

  return tensor;
}

PYBIND11_MODULE(PytorchNvCodec, m) {
  m.def("makefromDevicePtrUint8", &makefromDevicePtrUint8,
        py::return_value_policy::move);
}
