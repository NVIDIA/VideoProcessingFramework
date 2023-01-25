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

static int get_device_id(const void* dptr)
{
  cudaPointerAttributes attr;
  memset(&attr, 0, sizeof(attr));

  auto res = cudaPointerGetAttributes(&attr, dptr);
  if (cudaSuccess != res) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << ": failed to get pointer attributes. CUDA error code: ";
    ss << res;

    throw std::runtime_error(ss.str());
  }

  return attr.device;
}

torch::Tensor makefromDevicePtrUint8(CUdeviceptr ptr, uint32_t width,
                                     uint32_t height, uint32_t pitch,
                                     uint32_t elem_size, size_t str = 0U)
{
  if (elem_size != 1) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << ": only torch::kUInt8 data type is supported";
    throw std::runtime_error(ss.str());
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA, get_device_id((void*)ptr));

  torch::Tensor tensor = torch::full({height, width}, 128, options);

  uint8_t* devMem = nullptr;
  try {
    devMem = tensor.data_ptr<uint8_t>();
  } catch (std::exception& e) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << " failed to obtain uint8_t data pointer: " << e.what();
    throw std::runtime_error(ss.str());
  }

  if (!ptr) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << ": Video frame has void CUDA device ptr.";

    throw std::runtime_error(ss.str());
  }

  auto res = str ? cudaMemcpy2DAsync(
                       (void*)devMem, width, (const void*)ptr, pitch, width,
                       height, cudaMemcpyDeviceToDevice, (cudaStream_t)str)
                 : cudaMemcpy2D((void*)devMem, width, (const void*)ptr, pitch,
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

void copytoDevicePtrUint8(torch::Tensor tensor, CUdeviceptr ptr, uint32_t width,
                          uint32_t height, uint32_t pitch, uint32_t elem_size,
                          size_t str = 0U)
{
  if (elem_size != 1) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << ": only torch::kUInt8 data type is supported";
    throw std::runtime_error(ss.str());
  }

  uint8_t* devMem = nullptr;
  try {
    devMem = tensor.data_ptr<uint8_t>();
  } catch (std::exception& e) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << " failed to obtain uint8_t data pointer: " << e.what();
    throw std::runtime_error(ss.str());
  }

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

  auto res = str ? cudaMemcpy2DAsync(
                       (void*)ptr, pitch, (const void*)devMem, width, width,
                       height, cudaMemcpyDeviceToDevice, (cudaStream_t)str)
                 : cudaMemcpy2D((void*)ptr, pitch, (const void*)devMem, width,
                                width, height, cudaMemcpyDeviceToDevice);
  if (cudaSuccess != res) {
    std::stringstream ss;
    ss << __FUNCTION__;
    ss << ": failed to copy data to tensor. CUDA error code: ";
    ss << res;

    throw std::runtime_error(ss.str());
  }
}

PYBIND11_MODULE(_PytorchNvCodec, m)
{
  m.def(
      "makefromDevicePtrUint8",
      [](CUdeviceptr ptr, uint32_t width, uint32_t height, uint32_t pitch,
         uint32_t elem_size, size_t stream) {
        return makefromDevicePtrUint8(ptr, width, height, pitch, elem_size,
                                      stream);
      },
      py::arg("ptr"), py::arg("width"), py::arg("height"), py::arg("pitch"),
      py::arg("elem_size"), py::arg("stream"), py::return_value_policy::move,
      R"pbdoc(
        Create torch tensor from CUDA memory allocation.
        Works with VPF Surface.

        :param ptr: CUDA device pointer
        :param width: width in pixels
        :param height: height in pixels
        :param pitch: pitch in bytes
        :param elem_size: size of single element in bytes
        :param stream: CUDA stream to use
    )pbdoc");
  m.def(
      "DptrToTensor",
      [](CUdeviceptr ptr, uint32_t width, uint32_t height, uint32_t pitch,
         uint32_t elem_size, size_t stream) {
        return makefromDevicePtrUint8(ptr, width, height, pitch, elem_size,
                                      stream);
      },
      py::arg("ptr"), py::arg("width"), py::arg("height"), py::arg("pitch"),
      py::arg("elem_size"), py::arg("stream"), py::return_value_policy::move,
      R"pbdoc(
        Create torch tensor from CUDA memory allocation.
        Works with VPF Surface.

        :param ptr: CUDA device pointer
        :param width: width in pixels
        :param height: height in pixels
        :param pitch: pitch in bytes
        :param elem_size: size of single element in bytes
        :param stream: CUDA stream to use
    )pbdoc");
  m.def(
      "makefromDevicePtrUint8",
      [](CUdeviceptr ptr, uint32_t width, uint32_t height, uint32_t pitch,
         uint32_t elem_size) {
        return makefromDevicePtrUint8(ptr, width, height, pitch, elem_size);
      },
      py::arg("ptr"), py::arg("width"), py::arg("height"), py::arg("pitch"),
      py::arg("elem_size"), py::return_value_policy::move,
      R"pbdoc(
        Create torch tensor from CUDA memory allocation.
        Works with VPF Surface.

        :param ptr: CUDA device pointer
        :param width: width in pixels
        :param height: height in pixels
        :param pitch: pitch in bytes
        :param elem_size: size of single element in bytes
    )pbdoc");
  m.def(
      "DptrToTensor",
      [](CUdeviceptr ptr, uint32_t width, uint32_t height, uint32_t pitch,
         uint32_t elem_size) {
        return makefromDevicePtrUint8(ptr, width, height, pitch, elem_size);
      },
      py::arg("ptr"), py::arg("width"), py::arg("height"), py::arg("pitch"),
      py::arg("elem_size"), py::return_value_policy::move,
      R"pbdoc(
        Create torch tensor from CUDA memory allocation.
        Works with VPF Surface.

        :param ptr: CUDA device pointer
        :param width: width in pixels
        :param height: height in pixels
        :param pitch: pitch in bytes
        :param elem_size: size of single element in bytes
    )pbdoc");
  m.def(
      "TensorToDptr",
      [](torch::Tensor& tensor, CUdeviceptr ptr, uint32_t width,
         uint32_t height, uint32_t pitch, uint32_t elem_size, size_t stream) {
        return copytoDevicePtrUint8(tensor, ptr, width, height, pitch,
                                    elem_size, stream);
      },
      py::arg("tensor"), py::arg("ptr"), py::arg("width"), py::arg("height"),
      py::arg("pitch"), py::arg("elem_size"), py::arg("stream"),
      R"pbdoc(
        Copy torch tensor to CUDA memory allocation.
        Works with VPF Surface.

        :param tensor: torch tensor
        :param ptr: CUDA device pointer
        :param width: width in pixels
        :param height: height in pixels
        :param pitch: pitch in bytes
        :param elem_size: size of single element in bytes
        :param stream: CUDA stream to use
    )pbdoc");
  m.def(
      "TensorToDptr",
      [](torch::Tensor& tensor, CUdeviceptr ptr, uint32_t width,
         uint32_t height, uint32_t pitch, uint32_t elem_size) {
        return copytoDevicePtrUint8(tensor, ptr, width, height, pitch,
                                    elem_size);
      },
      py::arg("tensor"), py::arg("ptr"), py::arg("width"), py::arg("height"),
      py::arg("pitch"), py::arg("elem_size"), R"pbdoc(
        Copy torch tensor to CUDA memory allocation.
        Works with VPF Surface.

        :param tensor: torch tensor
        :param ptr: CUDA device pointer
        :param width: width in pixels
        :param height: height in pixels
        :param pitch: pitch in bytes
        :param elem_size: size of single element in bytes
    )pbdoc");
  m.doc() = R"pbdoc(
        PytorchNvCodec
        ---------------
        .. currentmodule:: PytorchNvCodec
        .. autosummary::
           :toctree: _generate
    )pbdoc";
}
