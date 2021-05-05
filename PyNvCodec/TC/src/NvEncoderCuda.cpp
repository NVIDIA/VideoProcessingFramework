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

#include "MemoryInterfaces.hpp"
#include "NvEncoderCuda.h"
#include <iostream>
using namespace std;

NvEncoderCuda::NvEncoderCuda(CUcontext cuContext, uint32_t nWidth,
                             uint32_t nHeight,
                             NV_ENC_BUFFER_FORMAT eBufferFormat,
                             uint32_t nExtraOutputDelay,
                             bool bMotionEstimationOnly,
                             bool bOutputInVideoMemory)
    :

      NvEncoder(NV_ENC_DEVICE_TYPE_CUDA, cuContext, nWidth, nHeight,
                eBufferFormat, nExtraOutputDelay, bMotionEstimationOnly,
                bOutputInVideoMemory),

      m_cuContext(cuContext) {
  if (!m_hEncoder) {
    NVENC_THROW_ERROR("Encoder Initialization failed",
                      NV_ENC_ERR_INVALID_DEVICE);
  }

  if (!m_cuContext) {
    NVENC_THROW_ERROR("Invalid Cuda Context", NV_ENC_ERR_INVALID_DEVICE);
  }
}

NvEncoderCuda::~NvEncoderCuda() { ReleaseCudaResources(); }

void NvEncoderCuda::AllocateInputBuffers(int32_t numInputBuffers) {
  if (!IsHWEncoderInitialized()) {
    NVENC_THROW_ERROR("Encoder initialization failed",
                      NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
  }

  // for MEOnly mode we need to allocate separate set of buffers for reference
  // frame
  int numCount = m_bMotionEstimationOnly ? 2 : 1;

  for (int count = 0; count < numCount; count++) {
    CudaCtxPush lock(m_cuContext);
    vector<void *> inputFrames;

    for (int i = 0; i < numInputBuffers; i++) {
      CUdeviceptr pDeviceFrame;
      uint32_t chromaHeight =
          GetNumChromaPlanes(GetPixelFormat()) *
          GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());
      if (GetPixelFormat() == NV_ENC_BUFFER_FORMAT_YV12 ||
          GetPixelFormat() == NV_ENC_BUFFER_FORMAT_IYUV){
        chromaHeight = GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());
      }

      CHECK_CUDA_CALL(cuMemAllocPitch(
          &pDeviceFrame, &m_cudaPitch,
          GetWidthInBytes(GetPixelFormat(), GetMaxEncodeWidth()),
          GetMaxEncodeHeight() + chromaHeight, 16));
      inputFrames.push_back((void *)pDeviceFrame);
    }

    RegisterInputResources(inputFrames,
                           NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                           GetMaxEncodeWidth(), GetMaxEncodeHeight(),
                           (int)m_cudaPitch, GetPixelFormat(), count == 1);
  }
}

void NvEncoderCuda::ReleaseInputBuffers() { ReleaseCudaResources(); }

void NvEncoderCuda::ReleaseCudaResources() {
  if (!m_hEncoder) {
    return;
  }

  if (!m_cuContext) {
    return;
  }

  UnregisterInputResources();

  CudaCtxPush lock(m_cuContext);

  for (auto inputFrame : m_vInputFrames) {
    if (inputFrame.inputPtr) {
      cuMemFree(reinterpret_cast<CUdeviceptr>(inputFrame.inputPtr));
    }
  }
  m_vInputFrames.clear();

  for (auto referenceFrame : m_vReferenceFrames) {
    if (referenceFrame.inputPtr) {
      cuMemFree(reinterpret_cast<CUdeviceptr>(referenceFrame.inputPtr));
    }
  }
  m_vReferenceFrames.clear();

  m_cuContext = nullptr;
}

void NvEncoderCuda::CopyToDeviceFrame(
    CUcontext ctx, CUstream stream, void *pSrcFrame, uint32_t nSrcPitch,
    CUdeviceptr pDstFrame, uint32_t dstPitch, int width, int height,
    CUmemorytype srcMemoryType, NV_ENC_BUFFER_FORMAT pixelFormat,
    const uint32_t dstChromaOffsets[], uint32_t numChromaPlanes) {
  if (srcMemoryType != CU_MEMORYTYPE_HOST &&
      srcMemoryType != CU_MEMORYTYPE_DEVICE) {
    NVENC_THROW_ERROR("Invalid source memory type for copy",
                      NV_ENC_ERR_INVALID_PARAM);
  }

  CudaCtxPush lock(ctx);

  uint32_t srcPitch =
      nSrcPitch ? nSrcPitch : NvEncoder::GetWidthInBytes(pixelFormat, width);
  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = srcMemoryType;
  if (srcMemoryType == CU_MEMORYTYPE_HOST) {
    m.srcHost = pSrcFrame;
  } else {
    m.srcDevice = (CUdeviceptr)pSrcFrame;
  }
  m.srcPitch = srcPitch;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = pDstFrame;
  m.dstPitch = dstPitch;
  m.WidthInBytes = NvEncoder::GetWidthInBytes(pixelFormat, width);
  m.Height = height;

  CHECK_CUDA_CALL(cuMemcpy2DAsync(&m, stream));

  vector<uint32_t> srcChromaOffsets;
  NvEncoder::GetChromaSubPlaneOffsets(pixelFormat, srcPitch, height,
                                      srcChromaOffsets);
  uint32_t chromaHeight = NvEncoder::GetChromaHeight(pixelFormat, height);
  uint32_t destChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, dstPitch);
  uint32_t srcChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, srcPitch);
  uint32_t chromaWidthInBytes =
      NvEncoder::GetChromaWidthInBytes(pixelFormat, width);

  for (uint32_t i = 0; i < numChromaPlanes; ++i) {
    if (chromaHeight) {
      if (srcMemoryType == CU_MEMORYTYPE_HOST) {
        m.srcHost = ((uint8_t *)pSrcFrame + srcChromaOffsets[i]);
      } else {
        m.srcDevice = (CUdeviceptr)((uint8_t *)pSrcFrame + srcChromaOffsets[i]);
      }
      m.srcPitch = srcChromaPitch;

      m.dstDevice = (CUdeviceptr)((uint8_t *)pDstFrame + dstChromaOffsets[i]);
      m.dstPitch = destChromaPitch;
      m.WidthInBytes = chromaWidthInBytes;
      m.Height = chromaHeight;
      CHECK_CUDA_CALL(cuMemcpy2DAsync(&m, stream));
    }
  }
}

NV_ENCODE_API_FUNCTION_LIST NvEncoderCuda::GetApi() const
{
  return m_nvenc;
}

void * NvEncoderCuda::GetEncoder() const
{
  return m_hEncoder;
}
