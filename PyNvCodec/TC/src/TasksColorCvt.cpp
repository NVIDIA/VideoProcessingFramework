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

#include "CodecsSupport.hpp"
#include "MemoryInterfaces.hpp"
#include "NppCommon.hpp"
#include "Tasks.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace VPF;
using namespace std;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

namespace VPF {

struct NppConvertSurface_Impl {
  NppConvertSurface_Impl(CUcontext ctx, CUstream str)
      : cu_ctx(ctx), cu_str(str) {
    SetupNppContext(cu_ctx, cu_str, nppCtx);
  }
  virtual ~NppConvertSurface_Impl() = default;
  virtual Token *Run(Token *pInput) = 0;

  CUcontext cu_ctx;
  CUstream cu_str;
  NppStreamContext nppCtx;
};

struct nv12_bgr final : public NppConvertSurface_Impl {
  nv12_bgr(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(BGR, width, height, context);
  }

  ~nv12_bgr() { delete pSurface; }

  Token *Run(Token *pInputNV12) override {
    NvtxMark tick(__FUNCTION__);
    if (!pInputNV12) {
      return nullptr;
    }

    auto pInput = (Surface *)pInputNV12;
    const Npp8u *const pSrc[] = {(const Npp8u *const)pInput->PlanePtr(0U),
                                 (const Npp8u *const)pInput->PlanePtr(1U)};

    auto pDst = (Npp8u *)pSurface->PlanePtr();
    NppiSize oSizeRoi = {(int)pInput->Width(), (int)pInput->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiNV12ToBGR_8u_P2C3R_Ctx(
        pSrc, pInput->Pitch(), pDst, pSurface->Pitch(), oSizeRoi, nppCtx);
    if (NPP_NO_ERROR != err) {
      cerr << "Failed to convert surface. Error code: " << err << endl;
      return nullptr;
    }

    return pSurface;
  }

  Surface *pSurface = nullptr;
};

struct nv12_rgb final : public NppConvertSurface_Impl {
  nv12_rgb(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(RGB, width, height, context);
  }

  ~nv12_rgb() { delete pSurface; }

  Token *Run(Token *pInputNV12) override {
    NvtxMark tick(__FUNCTION__);
    if (!pInputNV12) {
      return nullptr;
    }

    auto pInput = (Surface *)pInputNV12;
    const Npp8u *const pSrc[] = {(const Npp8u *const)pInput->PlanePtr(0U),
                                 (const Npp8u *const)pInput->PlanePtr(1U)};

    auto pDst = (Npp8u *)pSurface->PlanePtr();
    NppiSize oSizeRoi = {(int)pInput->Width(), (int)pInput->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
        pSrc, pInput->Pitch(), pDst, pSurface->Pitch(), oSizeRoi, nppCtx);
    if (NPP_NO_ERROR != err) {
      cerr << "Failed to convert surface. Error code: " << err << endl;
      return nullptr;
    }

    return pSurface;
  }

  Surface *pSurface = nullptr;
};

struct nv12_yuv420 final : public NppConvertSurface_Impl {
  nv12_yuv420(uint32_t width, uint32_t height, CUcontext context,
              CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(YUV420, width, height, context);
  }

  ~nv12_yuv420() { delete pSurface; }

  Token *Run(Token *pInputNV12) override {
    NvtxMark tick(__FUNCTION__);
    if (!pInputNV12) {
      return nullptr;
    }

    auto pInput_NV12 = (Surface *)pInputNV12;
    const Npp8u *const pSrc[] = {(const Npp8u *)pInput_NV12->PlanePtr(0U),
                                 (const Npp8u *)pInput_NV12->PlanePtr(1U)};

    Npp8u *pDst[] = {(Npp8u *)pSurface->PlanePtr(0U),
                     (Npp8u *)pSurface->PlanePtr(1U),
                     (Npp8u *)pSurface->PlanePtr(2U)};

    int dstStep[] = {(int)pSurface->Pitch(0U), (int)pSurface->Pitch(1U),
                     (int)pSurface->Pitch(2U)};
    NppiSize roi = {(int)pInput_NV12->Width(), (int)pInput_NV12->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiYCbCr420_8u_P2P3R_Ctx(pSrc[0], pInput_NV12->Pitch(0U),
                                         pSrc[1], pInput_NV12->Pitch(1U), pDst,
                                         dstStep, roi, nppCtx);
    if (NPP_NO_ERROR != err) {
      cerr << "Failed to convert surface. Error code: " << err << endl;
      return nullptr;
    }

    return pSurface;
  }

  Surface *pSurface = nullptr;
};

struct yuv420_rgb final : public NppConvertSurface_Impl {
  yuv420_rgb(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(RGB, width, height, context);
  }

  ~yuv420_rgb() { delete pSurface; }

  Token *Run(Token *pInputYUV420) override {
    NvtxMark tick(__FUNCTION__);
    if (!pInputYUV420) {
      return nullptr;
    }

    auto pInput_YUV420 = (SurfaceYUV420 *)pInputYUV420;
    const Npp8u *const pSrc[] = {(const Npp8u *)pInput_YUV420->PlanePtr(0U),
                                 (const Npp8u *)pInput_YUV420->PlanePtr(1U),
                                 (const Npp8u *)pInput_YUV420->PlanePtr(2U)};
    Npp8u *pDst = (Npp8u *)pSurface->PlanePtr();
    int srcStep[] = {(int)pInput_YUV420->Pitch(0U),
                     (int)pInput_YUV420->Pitch(1U),
                     (int)pInput_YUV420->Pitch(2U)};
    int dstStep = (int)pSurface->Pitch();
    NppiSize roi = {(int)pSurface->Width(), (int)pSurface->Height()};
    CudaCtxPush ctxPush(cu_ctx);
    auto err =
        nppiYUV420ToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi, nppCtx);
    if (NPP_NO_ERROR != err) {
      cerr << "Failed to convert surface. Error code: " << err << endl;
      return nullptr;
    }

    return pSurface;
  }

  Surface *pSurface = nullptr;
};

struct bgr_ycbcr final : public NppConvertSurface_Impl {
  bgr_ycbcr(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(YCBCR, width, height, context);
  }

  ~bgr_ycbcr() { delete pSurface; }

  Token *Run(Token *pInput) override {
    NvtxMark tick(__FUNCTION__);
    auto pInputBGR = (SurfaceRGB *)pInput;

    if (BGR != pInputBGR->PixelFormat()) {
      cerr << "Input surface isn't BGR" << endl;
      return nullptr;
    }

    if (YCBCR != pSurface->PixelFormat()) {
      cerr << "Output surface isn't YCbCr" << endl;
      return nullptr;
    }

    const Npp8u *pSrc = (const Npp8u *)pInputBGR->PlanePtr();
    int srcStep = pInputBGR->Pitch();
    Npp8u *pDst[] = {(Npp8u *)pSurface->PlanePtr(0U),
                     (Npp8u *)pSurface->PlanePtr(1U),
                     (Npp8u *)pSurface->PlanePtr(2U)};
    int dstStep[] = {(int)pSurface->Pitch(0U), (int)pSurface->Pitch(1U),
                     (int)pSurface->Pitch(2U)};
    NppiSize roi = {(int)pSurface->Width(), (int)pSurface->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiBGRToYCbCr420_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                              nppCtx);
    if (NPP_NO_ERROR != err) {
      cerr << "Failed to convert surface. Error code: " << err << endl;
      return nullptr;
    }

    return pSurface;
  }

  Surface *pSurface = nullptr;
};

struct rgb_yuv420 final : public NppConvertSurface_Impl {
  rgb_yuv420(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(YUV420, width, height, context);
  }

  ~rgb_yuv420() { delete pSurface; }

  Token *Run(Token *pInput) override {
    NvtxMark tick(__FUNCTION__);
    auto pInputRGB8 = (SurfaceRGB *)pInput;

    if (RGB != pInputRGB8->PixelFormat()) {
      return nullptr;
    }

    const Npp8u *pSrc = (const Npp8u *)pInputRGB8->PlanePtr();
    int srcStep = pInputRGB8->Pitch();
    Npp8u *pDst[] = {(Npp8u *)pSurface->PlanePtr(0U),
                     (Npp8u *)pSurface->PlanePtr(1U),
                     (Npp8u *)pSurface->PlanePtr(2U)};
    int dstStep[] = {(int)pSurface->Pitch(0U), (int)pSurface->Pitch(1U),
                     (int)pSurface->Pitch(2U)};
    NppiSize roi = {(int)pSurface->Width(), (int)pSurface->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err =
        nppiRGBToYUV420_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi, nppCtx);
    if (NPP_NO_ERROR != err) {
      cerr << "Failed to convert surface. Error code: " << err << endl;
      return nullptr;
    }

    return pSurface;
  }

  Surface *pSurface = nullptr;
};

struct yuv420_nv12 final : public NppConvertSurface_Impl {
  yuv420_nv12(uint32_t width, uint32_t height, CUcontext context,
              CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(NV12, width, height, context);
  }

  ~yuv420_nv12() { delete pSurface; }

  Token *Run(Token *pInputYUV420) override {
    NvtxMark tick(__FUNCTION__);
    if (!pInputYUV420) {
      return nullptr;
    }

    auto pInput_YUV420 = (Surface *)pInputYUV420;
    const Npp8u *const pSrc[] = {(const Npp8u *)pInput_YUV420->PlanePtr(0U),
                                 (const Npp8u *)pInput_YUV420->PlanePtr(1U),
                                 (const Npp8u *)pInput_YUV420->PlanePtr(2U)};

    Npp8u *pDst[] = {(Npp8u *)pSurface->PlanePtr(0U),
                     (Npp8u *)pSurface->PlanePtr(1U)};

    int srcStep[] = {(int)pInput_YUV420->Pitch(0U),
                     (int)pInput_YUV420->Pitch(1U),
                     (int)pInput_YUV420->Pitch(2U)};
    int dstStep[] = {(int)pSurface->Pitch(0U), (int)pSurface->Pitch(1U)};
    NppiSize roi = {(int)pInput_YUV420->Width(), (int)pInput_YUV420->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiYCbCr420_8u_P3P2R_Ctx(pSrc, srcStep, pDst[0], dstStep[0],
                                         pDst[1], dstStep[1], roi, nppCtx);
    if (NPP_NO_ERROR != err) {
      cerr << "Failed to convert surface. Error code: " << err << endl;
      return nullptr;
    }

    return pSurface;
  }

  Surface *pSurface = nullptr;
};

struct rgb8_deinterleave final : public NppConvertSurface_Impl {
  rgb8_deinterleave(uint32_t width, uint32_t height, CUcontext context,
                    CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(RGB_PLANAR, width, height, context);
  }

  ~rgb8_deinterleave() { delete pSurface; }

  Token *Run(Token *pInput) override {
    NvtxMark tick(__FUNCTION__);
    auto pInputRGB8 = (SurfaceRGB *)pInput;

    if (RGB != pInputRGB8->PixelFormat()) {
      return nullptr;
    }

    const Npp8u *pSrc = (const Npp8u *)pInputRGB8->PlanePtr();
    int nSrcStep = pInputRGB8->Pitch();
    Npp8u *aDst[] = {(Npp8u *)pSurface->PlanePtr(),
                     (Npp8u *)pSurface->PlanePtr() +
                         pSurface->Height() * pSurface->Pitch(),
                     (Npp8u *)pSurface->PlanePtr() +
                         pSurface->Height() * pSurface->Pitch() * 2};
    int nDstStep = pSurface->Pitch();
    NppiSize oSizeRoi = {0};
    oSizeRoi.height = pSurface->Height();
    oSizeRoi.width = pSurface->Width();

    CudaCtxPush ctxPush(cu_ctx);
    auto err =
        nppiCopy_8u_C3P3R_Ctx(pSrc, nSrcStep, aDst, nDstStep, oSizeRoi, nppCtx);
    if (NPP_NO_ERROR != err) {
      cerr << "Failed to convert surface. Error code: " << err << endl;
      return nullptr;
    }

    return pSurface;
  }

  Surface *pSurface = nullptr;
};

struct rbg8_swapchannel final : public NppConvertSurface_Impl {
  rbg8_swapchannel(uint32_t width, uint32_t height, CUcontext context,
                   CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(RGB, width, height, context);
  }

  ~rbg8_swapchannel() { delete pSurface; }

  Token *Run(Token *pInput) override {
    NvtxMark tick(__FUNCTION__);
    if (!pInput) {
      return nullptr;
    }

    auto pInputRGB8 = (SurfaceRGB *)pInput;
    if (RGB != pInputRGB8->PixelFormat()) {
      return nullptr;
    }

    const Npp8u *pSrc = (const Npp8u *)pInputRGB8->PlanePtr();

    int nSrcStep = pInputRGB8->Pitch();
    Npp8u *pDst = (Npp8u *)pSurface->PlanePtr();
    int nDstStep = pSurface->Pitch();
    NppiSize oSizeRoi = {0};
    oSizeRoi.height = pSurface->Height();
    oSizeRoi.width = pSurface->Width();
    // rgb to brg
    const int aDstOrder[3] = {2, 1, 0};
    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiSwapChannels_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                           oSizeRoi, aDstOrder, nppCtx);
    if (NPP_NO_ERROR != err) {
      cerr << "Failed to convert surface. Error code: " << err << endl;
      return nullptr;
    }

    return pSurface;
  }
  Surface *pSurface = nullptr;
};
} // namespace VPF

ConvertSurface::ConvertSurface(uint32_t width, uint32_t height,
                               Pixel_Format inFormat, Pixel_Format outFormat,
                               CUcontext ctx, CUstream str)
    : Task("NppConvertSurface", ConvertSurface::numInputs,
           ConvertSurface::numOutputs) {
  if (NV12 == inFormat && YUV420 == outFormat) {
    pImpl = new nv12_yuv420(width, height, ctx, str);
  } else if (YUV420 == inFormat && NV12 == outFormat) {
    pImpl = new yuv420_nv12(width, height, ctx, str);
  } else if (NV12 == inFormat && RGB == outFormat) {
    pImpl = new nv12_rgb(width, height, ctx, str);
  } else if (NV12 == inFormat && BGR == outFormat) {
    pImpl = new nv12_bgr(width, height, ctx, str);
  } else if (RGB == inFormat && RGB_PLANAR == outFormat) {
    pImpl = new rgb8_deinterleave(width, height, ctx, str);
  } else if (YUV420 == inFormat && RGB == outFormat) {
    pImpl = new yuv420_rgb(width, height, ctx, str);
  } else if (RGB == inFormat && YUV420 == outFormat) {
    pImpl = new rgb_yuv420(width, height, ctx, str);
  } else if (BGR == inFormat && YCBCR == outFormat) {
    pImpl = new bgr_ycbcr(width, height, ctx, str);
  }else if (RGB == inFormat && BGR == outFormat) {
      pImpl = new rbg8_swapchannel(width, height, ctx, str);
  }else {
    stringstream ss;
    ss << "Unsupported pixel format conversion: " << inFormat << " to "
       << outFormat;
    throw invalid_argument(ss.str());
  }
}

ConvertSurface::~ConvertSurface() { delete pImpl; }

ConvertSurface *ConvertSurface::Make(uint32_t width, uint32_t height,
                                     Pixel_Format inFormat,
                                     Pixel_Format outFormat, CUcontext ctx,
                                     CUstream str) {
  return new ConvertSurface(width, height, inFormat, outFormat, ctx, str);
}

TaskExecStatus ConvertSurface::Run() {
  ClearOutputs();
  auto pOutput = pImpl->Run(GetInput(0));
  SetOutput(pOutput, 0U);
  return TASK_EXEC_SUCCESS;
}
