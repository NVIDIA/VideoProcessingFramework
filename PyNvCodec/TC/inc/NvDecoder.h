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

#if defined(_WIN32)
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

#include "nvcuvid.h"
#include <stdint.h>

struct DecodedFrameContext {
  CUdeviceptr mem;
  uint64_t pts;
  uint64_t poc;

  DecodedFrameContext(CUdeviceptr new_ptr, uint64_t new_pts, uint64_t new_poc)
      : mem(new_ptr), pts(new_pts), poc(new_poc) {}

  DecodedFrameContext() : mem(0U), pts(0U), poc(0U) {}
};

unsigned long GetNumDecodeSurfaces(cudaVideoCodec eCodec, unsigned int nWidth,
                                   unsigned int nHeight);

class decoder_error : public std::runtime_error
{
public:
  decoder_error(const char *str) : std::runtime_error(str) {}
};

class cuvid_parser_error : public std::runtime_error
{
public:
  cuvid_parser_error(const char *str) : std::runtime_error(str) {}
};

namespace VPF {
class Buffer;
};

class DllExport NvDecoder {
public:
  NvDecoder() = delete;
  NvDecoder(const NvDecoder &other) = delete;
  NvDecoder &operator=(const NvDecoder &other) = delete;

  NvDecoder(CUstream cuStream, CUcontext cuContext, cudaVideoCodec eCodec,
            bool bLowLatency = false, int maxWidth = 0, int maxHeight = 0);

  ~NvDecoder();

  int GetWidth();

  int GetHeight();

  int GetChromaHeight();

  int GetFrameSize();

  int GetDeviceFramePitch();

  int GetBitDepth();

  bool DecodeLockSurface(VPF::Buffer const *encFrame,
                         uint64_t const &timestamp,
                         DecodedFrameContext &decCtx);

  void UnlockSurface(CUdeviceptr &lockedSurface);

  cudaVideoCodec GetCodec() const;

private:
  /* All the functions with Handle* prefix doesn't
   * throw as they are called from different thread;
   */
  static int CUDAAPI HandleVideoSequenceProc(
      void *pUserData, CUVIDEOFORMAT *pVideoFormat) noexcept {
    return ((NvDecoder *)pUserData)->HandleVideoSequence(pVideoFormat);
  }

  static int CUDAAPI HandlePictureDecodeProc(
      void *pUserData, CUVIDPICPARAMS *pPicParams) noexcept {
    return ((NvDecoder *)pUserData)->HandlePictureDecode(pPicParams);
  }

  static int CUDAAPI HandlePictureDisplayProc(
      void *pUserData, CUVIDPARSERDISPINFO *pDispInfo) noexcept {
    return ((NvDecoder *)pUserData)->HandlePictureDisplay(pDispInfo);
  }

  int HandleVideoSequence(CUVIDEOFORMAT *pVideoFormat) noexcept;

  int HandlePictureDecode(CUVIDPICPARAMS *pPicParams) noexcept;

  int HandlePictureDisplay(CUVIDPARSERDISPINFO *pDispInfo) noexcept;

  int ReconfigureDecoder(CUVIDEOFORMAT *pVideoFormat);

  struct NvDecoderImpl *p_impl;
};
