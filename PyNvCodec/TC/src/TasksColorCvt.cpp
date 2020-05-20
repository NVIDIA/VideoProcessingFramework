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
  virtual Token *Execute(Token *pInput) = 0;

  CUcontext cu_ctx;
  CUstream cu_str;
  NppStreamContext nppCtx;
};

struct nv12_rgb final : public NppConvertSurface_Impl {
  nv12_rgb(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(RGB, width, height, context);
  }

  ~nv12_rgb() { delete pSurface; }

  Token *Execute(Token *pInputNV12) override {
    if (!pInputNV12) {
      return nullptr;
    }

    auto pInput = (Surface *)pInputNV12;
    const Npp8u *const pSrc[] = {(const Npp8u *const)pInput->PlanePtr(0U),
                                 (const Npp8u *const)pInput->PlanePtr(1U)};

    auto pDst = (Npp8u *)pSurface->PlanePtr();
    NppiSize oSizeRoi = {(int)pInput->Width(), (int)pInput->Height()};

    NppLock lock(nppCtx);
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

  Token *Execute(Token *pInputNV12) override {
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

    NppLock lock(nppCtx);
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

  Token *Execute(Token *pInputYUV420) override {
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
    NppLock lock(nppCtx);
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

struct rgb_yuv420 final : public NppConvertSurface_Impl {
  rgb_yuv420(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream) {
    pSurface = Surface::Make(YUV420, width, height, context);
  }

  ~rgb_yuv420() { delete pSurface; }

  Token *Execute(Token *pInput) override {
    auto pInputRGB8 = (SurfaceRGB *)pInput;

    if (RGB != pInputRGB8->PixelFormat()) {
      return nullptr;
    }

    const Npp8u *pSrc = (const Npp8u *)pInputRGB8->PlanePtr();
    int srcStep = pInputRGB8->Pitch();
    Npp8u *pDst[] = {(Npp8u *)pSurface->PlanePtr(0U),
                     (Npp8u *)pSurface->PlanePtr(1U),
                     (Npp8u *)pSurface->PlanePtr(2U)};
    int dstStep[] = {(int)pSurface->Pitch(0U),
                     (int)pSurface->Pitch(1U),
                     (int)pSurface->Pitch(2U)};
    NppiSize roi = {(int)pSurface->Width(), (int)pSurface->Height()};

    NppLock lock(nppCtx);
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

  Token *Execute(Token *pInputYUV420) override {
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

    NppLock lock(nppCtx);
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

  Token *Execute(Token *pInput) override {
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

    NppLock lock(nppCtx);
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
  } else if (RGB == inFormat && RGB_PLANAR == outFormat) {
    pImpl = new rgb8_deinterleave(width, height, ctx, str);
  } else if (YUV420 == inFormat && RGB == outFormat) {
    pImpl = new yuv420_rgb(width, height, ctx, str);
  } else if (RGB == inFormat && YUV420 == outFormat) {
    pImpl = new rgb_yuv420(width, height, ctx, str);
  } else {
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

TaskExecStatus ConvertSurface::Execute() {
  ClearOutputs();
  auto pOutput = pImpl->Execute(GetInput(0));
  SetOutput(pOutput, 0U);
  return TASK_EXEC_SUCCESS;
}
