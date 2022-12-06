#pragma once

#include "nvcuvid.h"
#include "tc_dlopen.h"
#include <stdint.h>

typedef struct CuvidFunctions {
  TC_LIB lib;
  CUresult (*cuvidGetDecoderCaps)(CUVIDDECODECAPS* pdc);
  CUresult (*cuvidCreateDecoder)(CUvideodecoder* phDecoder,
                                 CUVIDDECODECREATEINFO* pdci);
  CUresult (*cuvidDestroyDecoder)(CUvideodecoder hDecoder);
  CUresult (*cuvidDecodePicture)(CUvideodecoder hDecoder,
                                 CUVIDPICPARAMS* pPicParams);
  CUresult (*cuvidGetDecodeStatus)(CUvideodecoder hDecoder, int nPicIdx,
                                   CUVIDGETDECODESTATUS* pDecodeStatus);
  CUresult (*cuvidReconfigureDecoder)(
      CUvideodecoder hDecoder, CUVIDRECONFIGUREDECODERINFO* pDecReconfigParams);
  // CUresult (*cuvidMapVideoFrame)(CUvideodecoder hDecoder, int nPicIdx,
  // unsigned int* pDevPtr, unsigned int* pPitch,
  // CUVIDPROCPARAMS* pVPP);
  // CUresult (*cuvidUnmapVideoFrame)(CUvideodecoder hDecoder,
  // unsigned int DevPtr);
  CUresult (*cuvidMapVideoFrame64)(CUvideodecoder hDecoder, int nPicIdx,
                                   unsigned int* pDevPtr, unsigned int* pPitch,
                                   CUVIDPROCPARAMS* pVPP);
  CUresult (*cuvidUnmapVideoFrame64)(CUvideodecoder hDecoder,
                                     unsigned long long DevPtr);
  CUresult (*cuvidCtxLockCreate)(CUvideoctxlock* pLock, CUcontext ctx);
  CUresult (*cuvidCtxLockDestroy)(CUvideoctxlock lck);
  CUresult (*cuvidCtxLock)(CUvideoctxlock lck, unsigned int reserved_flags);
  CUresult (*cuvidCtxUnlock)(CUvideoctxlock lck, unsigned int reserved_flags);
} CuvidFunctions;

#define CUVID_LOAD_STRINGIFY(s) _CUVID_LOAD_STRINGIFY(s)
#define _CUVID_LOAD_STRINGIFY(s) #s

#define CUVID_LOAD_LIBRARY(api, symbol)                                        \
  (api).symbol =                                                               \
      (typeof((api).symbol))tc_dlsym((api).lib, CUVID_LOAD_STRINGIFY(symbol)); \
  if (!(api).symbol) {                                                         \
    err = "Could not load function \"" CUVID_LOAD_STRINGIFY(symbol) "\"";      \
    goto err;                                                                  \
  }
#define CUVID_UNLOAD_LIBRARY(api, symbol) (api).symbol = NULL;

static const char* unloadCuvidSymbols(CuvidFunctions* cuvidApi)
{
  const char* err = NULL;
  if (!cuvidApi) {
    return NULL;
  }

  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidGetDecoderCaps);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCreateDecoder);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidDestroyDecoder);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidDecodePicture);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidGetDecodeStatus);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidReconfigureDecoder);
  // CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame64);
  // CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame64);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLockCreate);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLockDestroy);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLock);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxUnlock);
  if (tc_dlclose(cuvidApi->lib) != 0) {
    return "Failed to close library handle";
  };
  cuvidApi->lib = 0;
  return NULL;
}

static const char* loadCuvidSymbols(CuvidFunctions* cuvidApi, const char* path)
{
  const char* err = NULL;
  cuvidApi->lib = tc_dlopen(path);
  if (!cuvidApi->lib) {
    return "Failed to open dynamic library";
  }
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidGetDecoderCaps);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCreateDecoder);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidDestroyDecoder);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidDecodePicture);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidGetDecodeStatus);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidReconfigureDecoder);
  // CUVID_LOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame64);
  // CUVID_LOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame64);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCtxLockCreate);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCtxLockDestroy);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCtxLock);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCtxUnlock);

  return NULL;

err:
  unloadCuvidSymbols(cuvidApi);
  return err;
}
