/*
 * Copyright 2019 NVIDIA Corporation
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

#pragma once

#include "NvEncoder.h"
#include <cuda.h>
#include <mutex>
#include <stdint.h>
#include <vector>

#define CHECK_CUDA_CALL(call)                                                  \
  do {                                                                         \
    CUresult err__ = call;                                                     \
    if (err__ != CUDA_SUCCESS) {                                               \
      const char *szErrName = NULL;                                            \
      cuGetErrorName(err__, &szErrName);                                       \
      std::ostringstream errorLog;                                             \
      errorLog << "CUDA driver API error " << szErrName;                       \
      throw NVENCException::makeNVENCException(                                \
          errorLog.str(), NV_ENC_ERR_GENERIC, __FUNCTION__, __FILE__,          \
          __LINE__);                                                           \
    }                                                                          \
  } while (0)

/**
 *  @brief Encoder for CUDA device memory.
 */
class NvEncoderCuda final : public NvEncoder {
public:
  NvEncoderCuda(CUcontext cuContext, uint32_t nWidth, uint32_t nHeight,
                NV_ENC_BUFFER_FORMAT eBufferFormat,
                uint32_t nExtraOutputDelay = 3,
                bool bMotionEstimationOnly = false,
                bool bOPInVideoMemory = false);

  ~NvEncoderCuda() override;

  static void CopyToDeviceFrame(
      CUcontext device, CUstream stream, void *pSrcFrame, uint32_t nSrcPitch,
      CUdeviceptr pDstFrame, uint32_t dstPitch, int width, int height,
      CUmemorytype srcMemoryType, NV_ENC_BUFFER_FORMAT pixelFormat,
      const uint32_t dstChromaOffsets[], uint32_t numChromaPlanes);

  NV_ENCODE_API_FUNCTION_LIST GetApi() const;

  void* GetEncoder() const;

private:
  /**
   *  @brief This function is used to release the input buffers allocated for
   * encoding. This function is an override of virtual function
   * NvEncoder::ReleaseInputBuffers().
   */
  void ReleaseInputBuffers() override;

  /**
   *  @brief This function is used to allocate input buffers for encoding.
   *  This function is an override of virtual function
   * NvEncoder::AllocateInputBuffers().
   */
  void AllocateInputBuffers(int32_t numInputBuffers) override;

  /**
   *  @brief This is a private function to release CUDA device memory used for
   * encoding.
   */
  void ReleaseCudaResources();

  CUcontext m_cuContext;

  size_t m_cudaPitch = 0;
};
