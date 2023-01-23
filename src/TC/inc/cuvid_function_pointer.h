#include "nvcuvid.h"
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#define TC_LIB HMODULE
#else
#define TC_LIB int
#endif

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
  CUresult (*cuvidMapVideoFrame)(CUvideodecoder hDecoder, int nPicIdx,
                                 unsigned int* pDevPtr, unsigned int* pPitch,
                                 CUVIDPROCPARAMS* pVPP);
  CUresult (*cuvidUnmapVideoFrame)(CUvideodecoder hDecoder,
                                   unsigned int DevPtr);
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
  (api).(symbol) = tc_dlsym((api).(lib), (symbol));                            \
  if (!(api).(function)) {                                                     \
    err = "Could not load function \"" CUVID_LOAD_STRINGIFY(symbol) "\"";      \
    goto err;                                                                  \
  }
#define CUVID_UNLOAD_LIBRARY(api, symbol) (api).(symbol) = NULL;

static const char* unloadCuvidSymbols(CuvidFuncitons* cuvidApi,
                                      const char* file)
{
  const char* err = NULL;
  if (!cuvidApi) {
    return NULL;
  }

  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidGetDecoderCaps);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCreateDecoder);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidDestroyDecoder);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidDecodePicture);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidDecodeStatus);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidReconfigureEncoder);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame64);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame64);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLockCreate);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLockDestroy);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLock);
  CUVID_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLockUnlock);
  if (tc_dlclose(cuvidApi->lib) != 0) {
    return "Failed to close library handle";
  };
  return NULL;
}

static bool loadCuvidSymbols(CuvidFuncitons* cuvidApi, const char* file)
{
  const char* err = NULL;
  cuvidApi->lib = tc_dlopen(path);
  if (!lib) {
    return "Failed to open dynamic library";
  }
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidGetDecoderCaps);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCreateDecoder);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidDestroyDecoder);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidDecodePicture);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidDecodeStatus);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidReconfigureEncoder);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame64);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame64);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCtxLockCreate);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCtxLockDestroy);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCtxLock);
  CUVID_LOAD_LIBRARY(*cuvidApi, cuvidCtxLockUnlock);

  return NULL;

err:
  unloadCuvidSymbols(cuvidApi);
  return err;
}
