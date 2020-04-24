#include "NppCommon.hpp"
#include <cstring>

void SetupNppContext(CUcontext context, CUstream stream,
                     NppStreamContext &nppCtx) {
  memset(&nppCtx, 0, sizeof(nppCtx));

  cuCtxPushCurrent(context);
  CUdevice device;
  cuCtxGetDevice(&device);

  cudaDeviceProp properties = {0};
  cudaGetDeviceProperties(&properties, device);
  cuCtxPopCurrent(nullptr);

  nppCtx.hStream = stream;
  nppCtx.nCudaDeviceId = (int)device;
  nppCtx.nMultiProcessorCount = properties.multiProcessorCount;
  nppCtx.nMaxThreadsPerBlock = properties.maxThreadsPerBlock;
  nppCtx.nSharedMemPerBlock = properties.sharedMemPerBlock;
  nppCtx.nCudaDevAttrComputeCapabilityMajor = properties.major;
  nppCtx.nCudaDevAttrComputeCapabilityMinor = properties.minor;
}