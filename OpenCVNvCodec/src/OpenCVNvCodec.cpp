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

#include <pybind11/pybind11.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/cuda.hpp>

namespace py = pybind11;

cv::cuda::GpuMat fromDevicePtrUint8(CUdeviceptr ptr, uint32_t width,
                                    uint32_t height, uint32_t pitch,
                                    uint32_t elem_size_bytes) {
      cv::cuda::GpuMat mat;
      return mat;
}

CUdeviceptr fromGpuMatUint8(cv::cuda::GpuMat mat,  uint32_t width,
                            uint32_t height, uint32_t pitch) {
      CUdeviceptr ptr;
      return ptr;
}

PYBIND11_MODULE(OpenCVNvCodec, m) {
  m.def("fromDevicePtrUint8", &fromDevicePtrUint8,
        py::return_value_policy::move);
  m.def("fromGpuMatUint8", &fromGpuMatUint8,
        py::return_value_policy::move);
}
