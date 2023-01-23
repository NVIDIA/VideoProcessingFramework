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

#include "CodecsSupport.hpp"
#include "tc_core_export.h"

#include "CodecsSupport.hpp"
#include "CuvidFunctions.h"
#include "nvcuvid.h"
#include <stdexcept>
#include <stdint.h>

struct DecodedFrameContext {
  CUdeviceptr mem;
  uint64_t pts;
  uint64_t bsl;
  PacketData out_pdata;

  // Set up this flag to feed decoder with empty input without setting up EOS
  // flag;
  bool no_eos;

  DecodedFrameContext(CUdeviceptr new_ptr, uint64_t new_pts, uint64_t new_poc)
      : mem(new_ptr), pts(new_pts), no_eos(false)
  {
  }

  DecodedFrameContext(CUdeviceptr new_ptr, uint64_t new_pts, uint64_t new_poc,
                      bool new_no_eos)
      : mem(new_ptr), pts(new_pts), no_eos(new_no_eos)
  {
  }

  DecodedFrameContext() : mem(0U), pts(0U), no_eos(false) {}
};

unsigned long GetNumDecodeSurfaces(cudaVideoCodec eCodec, unsigned int nWidth,
                                   unsigned int nHeight);

class decoder_error : public std::runtime_error
{
public:
  decoder_error(const char* str) : std::runtime_error(str) {}
};

class cuvid_parser_error : public std::runtime_error
{
public:
  cuvid_parser_error(const char* str) : std::runtime_error(str) {}
};

namespace VPF
{
class Buffer;
};

class TC_CORE_EXPORT NvDecoder
{
public:
  NvDecoder() = delete;
  NvDecoder(const NvDecoder& other) = delete;
  NvDecoder& operator=(const NvDecoder& other) = delete;

  NvDecoder(CUstream cuStream, CUcontext cuContext, cudaVideoCodec eCodec,
            bool bLowLatency = false, int maxWidth = 0, int maxHeight = 0);

  ~NvDecoder();

  int GetWidth();

  int GetHeight();

  int GetChromaHeight();

  int GetFrameSize();

  int GetDeviceFramePitch();

  int GetBitDepth();

  bool DecodeLockSurface(VPF::Buffer const* encFrame,
                         struct PacketData const& pdata,
                         DecodedFrameContext& decCtx);

  void UnlockSurface(CUdeviceptr& lockedSurface);

  void Init(CUVIDEOFORMAT* format) { HandleVideoSequence(format); }

  cudaVideoCodec GetCodec() const;
  cudaVideoChromaFormat GetChromaFormat() const;
  inline const CuvidFunctions& _api() { return m_api; };

private:
  /* All the functions with Handle* prefix doesn't
   * throw as they are called from different thread;
   */
  static int CUDAAPI
  HandleVideoSequenceProc(void* pUserData, CUVIDEOFORMAT* pVideoFormat) noexcept
  {
    return ((NvDecoder*)pUserData)->HandleVideoSequence(pVideoFormat);
  }

  static int CUDAAPI
  HandlePictureDecodeProc(void* pUserData, CUVIDPICPARAMS* pPicParams) noexcept
  {
    return ((NvDecoder*)pUserData)->HandlePictureDecode(pPicParams);
  }

  static int CUDAAPI HandlePictureDisplayProc(
      void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) noexcept
  {
    return ((NvDecoder*)pUserData)->HandlePictureDisplay(pDispInfo);
  }

  int HandleVideoSequence(CUVIDEOFORMAT* pVideoFormat) noexcept;

  int HandlePictureDecode(CUVIDPICPARAMS* pPicParams) noexcept;

  int HandlePictureDisplay(CUVIDPARSERDISPINFO* pDispInfo) noexcept;

  int ReconfigureDecoder(CUVIDEOFORMAT* pVideoFormat);

  struct NvDecoderImpl* p_impl;
  CuvidFunctions m_api{};
};
