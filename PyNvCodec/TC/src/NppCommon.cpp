#include "NppCommon.hpp"
#include <cstring>
#include <iostream>
#include <mutex>

using namespace std;

static mutex gNppMutex;

void SetupNppContext(CUcontext context, CUstream stream,
                     NppStreamContext &nppCtx) {
  memset(&nppCtx, 0, sizeof(nppCtx));

  gNppMutex.lock();
  cuCtxPushCurrent(context);
  CUdevice device;
  auto res = cuCtxGetDevice(&device);
  if (CUDA_SUCCESS != res) {
    cerr << "Failed to get CUDA device. Error code: " << res << endl;
  }

  cudaDeviceProp properties = {0};
  auto ret = cudaGetDeviceProperties(&properties, device);
  if (cudaSuccess != ret) {
    cerr << "Failed to get CUDA device properties. Error code: " << ret << endl;
    cerr << "Error description: " << cudaGetErrorString(ret) << endl;
  }
  cuCtxPopCurrent(nullptr);

  gNppMutex.unlock();

  nppCtx.hStream = stream;
  nppCtx.nCudaDeviceId = (int)device;
  nppCtx.nMultiProcessorCount = properties.multiProcessorCount;
  nppCtx.nMaxThreadsPerBlock = properties.maxThreadsPerBlock;
  nppCtx.nSharedMemPerBlock = properties.sharedMemPerBlock;
  nppCtx.nCudaDevAttrComputeCapabilityMajor = properties.major;
  nppCtx.nCudaDevAttrComputeCapabilityMinor = properties.minor;
}