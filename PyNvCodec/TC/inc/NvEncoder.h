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

#include "nvEncodeAPI.h"
#include <cuda.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>

class NVENCException final : public std::exception {
public:
  NVENCException(const std::string &errorStr, const NVENCSTATUS errorCode)
      :

        runtimeError(errorStr), m_errorCode(errorCode) {}

  ~NVENCException() noexcept override {}

  const char *what() const noexcept override { return runtimeError.what(); }

  static NVENCException makeNVENCException(const std::string &errorStr,
                                           NVENCSTATUS errorCode,
                                           const std::string &functionName,
                                           const std::string &fileName,
                                           int lineNo);

private:
  std::runtime_error runtimeError;
  NVENCSTATUS m_errorCode;
};

inline NVENCException NVENCException::makeNVENCException(
    const std::string &errorStr, NVENCSTATUS errorCode,
    const std::string &functionName, const std::string &fileName, int lineNo) {
  std::ostringstream errorLog;
  errorLog << functionName << " : " << errorStr << " at " << fileName << ":"
           << lineNo << std::endl;
  NVENCException exception(errorLog.str(), errorCode);
  return exception;
}

#define NVENC_THROW_ERROR(errorStr, errorCode)                                 \
  do {                                                                         \
    throw NVENCException::makeNVENCException(                                  \
        errorStr, errorCode, __FUNCTION__, __FILE__, __LINE__);                \
  } while (0)

#define NVENC_API_CALL(nvencAPI)                                               \
  do {                                                                         \
    NVENCSTATUS errorCode = nvencAPI;                                          \
    if (errorCode != NV_ENC_SUCCESS) {                                         \
      std::ostringstream errorLog;                                             \
      errorLog << #nvencAPI << " returned error " << errorCode;                \
      throw NVENCException::makeNVENCException(                                \
          errorLog.str(), errorCode, __FUNCTION__, __FILE__, __LINE__);        \
    }                                                                          \
  } while (0)

struct NvEncInputFrame {
  void *inputPtr = nullptr;
  uint32_t chromaOffsets[2] = {0};
  uint32_t numChromaPlanes = 0;
  uint32_t pitch = 0;
  uint32_t chromaPitch = 0;
  NV_ENC_BUFFER_FORMAT bufferFormat = NV_ENC_BUFFER_FORMAT_UNDEFINED;
  NV_ENC_INPUT_RESOURCE_TYPE resourceType =
      NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
};

class NvEncoder {
public:
  void CreateEncoder(const NV_ENC_INITIALIZE_PARAMS *pEncodeParams,
                     CUstream str);

  void DestroyEncoder();

  const NvEncInputFrame *GetNextInputFrame();

  void EncodeFrame(std::vector<std::vector<uint8_t>> &vPacket,
                   NV_ENC_PIC_PARAMS *pPicParams = nullptr);

  void EndEncode(std::vector<std::vector<uint8_t>> &vPacket);

  int GetCapabilityValue(GUID guidCodec, NV_ENC_CAPS capsToQuery);

  void *GetDevice() const;

  int GetEncodeWidth() const;

  int GetEncodeHeight() const;

  void CreateDefaultEncoderParams(NV_ENC_INITIALIZE_PARAMS *pIntializeParams,
                                  GUID codecGuid, GUID presetGuid);

  void GetInitializeParams(NV_ENC_INITIALIZE_PARAMS *pInitializeParams);

  virtual ~NvEncoder();

  static void GetChromaSubPlaneOffsets(NV_ENC_BUFFER_FORMAT bufferFormat,
                                       uint32_t pitch, uint32_t height,
                                       std::vector<uint32_t> &chromaOffsets);

  static uint32_t GetChromaPitch(NV_ENC_BUFFER_FORMAT bufferFormat,
                                 uint32_t lumaPitch);

  static uint32_t GetNumChromaPlanes(NV_ENC_BUFFER_FORMAT bufferFormat);

  static uint32_t GetChromaWidthInBytes(NV_ENC_BUFFER_FORMAT bufferFormat,
                                        uint32_t lumaWidth);

  static uint32_t GetChromaHeight(NV_ENC_BUFFER_FORMAT bufferFormat,
                                  uint32_t lumaHeight);

  static uint32_t GetWidthInBytes(NV_ENC_BUFFER_FORMAT bufferFormat,
                                  uint32_t width);

protected:
  NvEncoder(NV_ENC_DEVICE_TYPE eDeviceType, void *pDevice, uint32_t nWidth,
            uint32_t nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat,
            uint32_t nOutputDelay, bool bMotionEstimationOnly,
            bool bOutputInVideoMemory = false);

  bool IsHWEncoderInitialized() const;

  void RegisterInputResources(std::vector<void *> const &input_frames,
                              NV_ENC_INPUT_RESOURCE_TYPE eResourceType,
                              int width, int height, int pitch,
                              NV_ENC_BUFFER_FORMAT bufferFormat,
                              bool bReferenceFrame = false);

  void UnregisterInputResources();

  NV_ENC_REGISTERED_PTR
  RegisterResource(void *pBuffer, NV_ENC_INPUT_RESOURCE_TYPE eResourceType,
                   int width, int height, int pitch,
                   NV_ENC_BUFFER_FORMAT bufferFormat,
                   NV_ENC_BUFFER_USAGE bufferUsage = NV_ENC_INPUT_IMAGE);

  uint32_t GetMaxEncodeWidth() const;

  uint32_t GetMaxEncodeHeight() const;

  void *GetCompletionEvent(uint32_t eventIdx);

  NV_ENC_BUFFER_FORMAT
  GetPixelFormat() const;

  NVENCSTATUS
  DoEncode(NV_ENC_INPUT_PTR inputBuffer, NV_ENC_OUTPUT_PTR outputBuffer,
           NV_ENC_PIC_PARAMS *pPicParams);

  void MapResources(uint32_t bfrIdx);

  void WaitForCompletionEvent(int iEvent);

  void SendEOS();

private:
  void LoadNvEncApi();

  void GetEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR> const &vOutputBuffer,
                        std::vector<std::vector<uint8_t>> &vPacket,
                        bool bOutputDelay);

  void InitializeBitstreamBuffer();

  void DestroyBitstreamBuffer();

  void InitializeMVOutputBuffer();

  void DestroyMVOutputBuffer();

  void DestroyHWEncoder();

  void FlushEncoder();

  virtual void AllocateInputBuffers(int32_t numInputBuffers) = 0;

  virtual void ReleaseInputBuffers() = 0;

protected:
  bool m_bMotionEstimationOnly = false;
  bool m_bOutputInVideoMemory = false;
  bool m_bEncoderInitialized = false;

  void *m_hModule = nullptr;
  void *m_hEncoder = nullptr;
  void *m_pDevice = nullptr;

  NV_ENCODE_API_FUNCTION_LIST m_nvenc;

  std::vector<NvEncInputFrame> m_vInputFrames;
  std::vector<NvEncInputFrame> m_vReferenceFrames;
  std::vector<NV_ENC_REGISTERED_PTR> m_vRegisteredResources;
  std::vector<NV_ENC_REGISTERED_PTR> m_vRegisteredResourcesForReference;
  std::vector<NV_ENC_INPUT_PTR> m_vMappedInputBuffers;
  std::vector<NV_ENC_INPUT_PTR> m_vMappedRefBuffers;
  std::vector<NV_ENC_OUTPUT_PTR> m_vBitstreamOutputBuffer;
  std::vector<NV_ENC_OUTPUT_PTR> m_vMVDataOutputBuffer;

  std::vector<void *> m_vpCompletionEvents;

  int32_t m_iToSend = 0;
  int32_t m_iGot = 0;
  int32_t m_nEncoderBufferSize = 0;
  int32_t m_nOutputDelay = 0;
  uint32_t m_nExtraOutputDelay = 3;
  uint32_t m_nWidth = 0;
  uint32_t m_nHeight = 0;
  uint32_t m_nMaxEncodeWidth = 0;
  uint32_t m_nMaxEncodeHeight = 0;

  NV_ENC_BUFFER_FORMAT m_eBufferFormat;
  NV_ENC_DEVICE_TYPE m_eDeviceType;
  NV_ENC_INITIALIZE_PARAMS m_initializeParams = {};
  NV_ENC_CONFIG m_encodeConfig = {};
};
