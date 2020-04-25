#include "NppCommon.hpp"
#include <cstring>
#include <iostream>
#include <mutex>

using namespace std;

void SetupNppContext(CUcontext context, CUstream stream,
                     NppStreamContext &nppCtx) {
  memset(&nppCtx, 0, sizeof(nppCtx));

  cuCtxPushCurrent(context);
  CUdevice device;
  auto res = cuCtxGetDevice(&device);
  if (CUDA_SUCCESS != res) {
    cerr << "Failed to get CUDA device. Error code: " << res << endl;
  }

  cudaDeviceProp properties = {0};
  auto ret = cudaGetDeviceProperties(&properties, device);
  if (CUDA_SUCCESS != ret) {
    cerr << "Failed to get CUDA device properties. Error code: " << ret << endl;
  }
  cuCtxPopCurrent(nullptr);

  nppCtx.hStream = stream;
  nppCtx.nCudaDeviceId = (int)device;
  nppCtx.nMultiProcessorCount = properties.multiProcessorCount;
  nppCtx.nMaxThreadsPerBlock = properties.maxThreadsPerBlock;
  nppCtx.nSharedMemPerBlock = properties.sharedMemPerBlock;
  nppCtx.nCudaDevAttrComputeCapabilityMajor = properties.major;
  nppCtx.nCudaDevAttrComputeCapabilityMinor = properties.minor;
}

static mutex gNppMutex;

NppLock::NppLock(NppStreamContext &nppCtx) : ctx(nppCtx) { gNppMutex.lock(); }

NppLock::~NppLock() {
  cudaStreamSynchronize(ctx.hStream);
  gNppMutex.unlock();
}