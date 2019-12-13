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
#include "Logger.h"
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <thread>

extern simplelogger::Logger *logger;

#ifdef __cuda_cuda_h__
inline bool check(CUresult e, int iLine, const char *szFile) {
  if (e != CUDA_SUCCESS) {
    const char *szErrName = NULL;
    cuGetErrorName(e, &szErrName);
    LOG(FATAL) << "CUDA driver API error " << szErrName << " at line " << iLine
               << " in file " << szFile;
    return false;
  }
  return true;
}
#endif

#ifdef __CUDA_RUNTIME_H__
inline bool check(cudaError_t e, int iLine, const char *szFile) {
  if (e != cudaSuccess) {
    LOG(FATAL) << "CUDA runtime API error " << cudaGetErrorName(e)
               << " at line " << iLine << " in file " << szFile;
    return false;
  }
  return true;
}
#endif

#ifdef _NV_ENCODEAPI_H_
inline bool check(NVENCSTATUS e, int iLine, const char *szFile) {
  const char *aszErrName[] = {
      "NV_ENC_SUCCESS",
      "NV_ENC_ERR_NO_ENCODE_DEVICE",
      "NV_ENC_ERR_UNSUPPORTED_DEVICE",
      "NV_ENC_ERR_INVALID_ENCODERDEVICE",
      "NV_ENC_ERR_INVALID_DEVICE",
      "NV_ENC_ERR_DEVICE_NOT_EXIST",
      "NV_ENC_ERR_INVALID_PTR",
      "NV_ENC_ERR_INVALID_EVENT",
      "NV_ENC_ERR_INVALID_PARAM",
      "NV_ENC_ERR_INVALID_CALL",
      "NV_ENC_ERR_OUT_OF_MEMORY",
      "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
      "NV_ENC_ERR_UNSUPPORTED_PARAM",
      "NV_ENC_ERR_LOCK_BUSY",
      "NV_ENC_ERR_NOT_ENOUGH_BUFFER",
      "NV_ENC_ERR_INVALID_VERSION",
      "NV_ENC_ERR_MAP_FAILED",
      "NV_ENC_ERR_NEED_MORE_INPUT",
      "NV_ENC_ERR_ENCODER_BUSY",
      "NV_ENC_ERR_EVENT_NOT_REGISTERD",
      "NV_ENC_ERR_GENERIC",
      "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY",
      "NV_ENC_ERR_UNIMPLEMENTED",
      "NV_ENC_ERR_RESOURCE_REGISTER_FAILED",
      "NV_ENC_ERR_RESOURCE_NOT_REGISTERED",
      "NV_ENC_ERR_RESOURCE_NOT_MAPPED",
  };
  if (e != NV_ENC_SUCCESS) {
    LOG(FATAL) << "NVENC error " << aszErrName[e] << " at line " << iLine
               << " in file " << szFile;
    return false;
  }
  return true;
}
#endif

#ifdef _WINERROR_
inline bool check(HRESULT e, int iLine, const char *szFile) {
  if (e != S_OK) {
    LOG(FATAL) << "HRESULT error 0x" << e << " at line " << iLine << " in file "
               << szFile;
    return false;
  }
  return true;
}
#endif

#if defined(__gl_h_) || defined(__GL_H__)
inline bool check(GLenum e, int iLine, const char *szFile) {
  if (e != 0) {
    LOG(ERROR) << "GLenum error " << e << " at line " << iLine << " in file "
               << szFile;
    return false;
  }
  return true;
}
#endif

inline bool check(int e, int iLine, const char *szFile) {
  if (e < 0) {
    LOG(ERROR) << "General error " << e << " at line " << iLine << " in file "
               << szFile;
    return false;
  }
  return true;
}

#define ck(call) check(call, __LINE__, __FILE__)

#ifndef _WIN32
#define _stricmp strcasecmp
#endif

void ResizeNv12(unsigned char *dpDstNv12, int nDstPitch, int nDstWidth,
                int nDstHeight, unsigned char *dpSrcNv12, int nSrcPitch,
                int nSrcWidth, int nSrcHeight,
                unsigned char *dpDstNv12UV = nullptr, cudaStream_t S = 0);