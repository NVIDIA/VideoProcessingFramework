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

#include "MemoryInterfaces.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <new>
#include <sstream>
#include <stdexcept>

using namespace VPF;
using namespace VPF;
using namespace std;

#ifdef TRACK_TOKEN_ALLOCATIONS
#include <algorithm>
#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace VPF {

struct AllocInfo {
  uint64_t id;
  uint64_t size;

  bool operator==(const AllocInfo &other) {
    /* Buffer size may change during the lifetime so we check id only;
     */
    return id == other.id;
  }

  explicit AllocInfo(decltype(id) const &newId, decltype(size) const &newSize)
      : id(newId), size(newSize) {}
};

struct AllocRegister {
  vector<AllocInfo> instances;
  mutex guard;
  uint64_t ID = 1U;

  decltype(AllocInfo::id) AddNote(decltype(AllocInfo::size) const &size) {
    unique_lock<decltype(guard)> lock;
    auto id = ID++;
    AllocInfo info(id, size);
    instances.push_back(info);
    return id;
  }

  void DeleteNote(AllocInfo const &allocInfo) {
    unique_lock<decltype(guard)> lock;
    instances.erase(remove(instances.begin(), instances.end(), allocInfo),
                    instances.end());
  }

  /* Call this after you're done releasing mem objects in your app;
   */
  size_t GetSize() const { return instances.size(); }

  /* Call this after you're done releasing mem objects in your app;
   */
  AllocInfo const *GetNoteByIndex(uint64_t idx) {
    return idx < instances.size() ? instances.data() + idx : nullptr;
  }
};

AllocRegister BuffersRegister, HWSurfaceRegister;

bool CheckAllocationCounters() {
  auto numLeakedBuffers = BuffersRegister.GetSize();
  auto numLeakedSurfaces = HWSurfaceRegister.GetSize();

  if (numLeakedBuffers) {
    cerr << "Leaked buffers (id : size): " << endl;
    for (auto i = 0; i < numLeakedBuffers; i++) {
      auto pNote = BuffersRegister.GetNoteByIndex(i);
      cerr << "\t" << pNote->id << "\t: " << pNote->size << endl;
    }
  }

  if (numLeakedSurfaces) {
    cerr << "Leaked surfaces (id : size): " << endl;
    for (auto i = 0; i < numLeakedSurfaces; i++) {
      auto pNote = HWSurfaceRegister.GetNoteByIndex(i);
      cerr << "\t" << pNote->id << "\t: " << pNote->size << endl;
    }
  }

  return (0U == numLeakedBuffers) && (0U == numLeakedSurfaces);
}

} // namespace VPF
#endif

Buffer *Buffer::Make(size_t bufferSize) {
  return new Buffer(bufferSize, false, nullptr);
}

Buffer *Buffer::Make(size_t bufferSize, void *pCopyFrom) {
  return new Buffer(bufferSize, pCopyFrom, false, nullptr);
}

Buffer::Buffer(size_t bufferSize, bool ownMemory, CUcontext ctx)
    : mem_size(bufferSize), own_memory(ownMemory), context(ctx) {
  if (own_memory) {
    if (!Allocate()) {
      throw bad_alloc();
    }
  }
#ifdef TRACK_TOKEN_ALLOCATIONS
  id = BuffersRegister.AddNote(mem_size);
#endif
}

Buffer::Buffer(size_t bufferSize, void *pCopyFrom, bool ownMemory,
               CUcontext ctx)
    : mem_size(bufferSize), own_memory(ownMemory), context(ctx) {
  if (own_memory) {
    if (Allocate()) {
      memcpy(this->GetRawMemPtr(), pCopyFrom, bufferSize);
    } else {
      throw bad_alloc();
    }
  } else {
    pRawData = pCopyFrom;
  }
#ifdef TRACK_TOKEN_ALLOCATIONS
  id = BuffersRegister.AddNote(mem_size);
#endif
}

Buffer::Buffer(size_t bufferSize, const void *pCopyFrom, CUcontext ctx)
    : mem_size(bufferSize), own_memory(true), context(ctx) {
  if (Allocate()) {
    memcpy(this->GetRawMemPtr(), pCopyFrom, bufferSize);
  } else {
    throw bad_alloc();
  }
#ifdef TRACK_TOKEN_ALLOCATIONS
  id = BuffersRegister.AddNote(mem_size);
#endif
}

Buffer::~Buffer() {
  Deallocate();
#ifdef TRACK_TOKEN_ALLOCATIONS
  AllocInfo info(id, mem_size);
  BuffersRegister.DeleteNote(info);
#endif
}

size_t Buffer::GetRawMemSize() const { return mem_size; }

static void ThrowOnCudaError(CUresult res, int lineNum = -1) {
  if (CUDA_SUCCESS != res) {
    stringstream ss;

    if (lineNum > 0) {
      ss << __FILE__ << ":";
      ss << lineNum << endl;
    }

    const char *errName = nullptr;
    if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {
      ss << "CUDA error with code " << res << endl;
    } else {
      ss << "CUDA error: " << errName << endl;
    }

    const char *errDesc = nullptr;
    if (CUDA_SUCCESS != cuGetErrorString(res, &errDesc)) {
      // Try CUDA runtime function then;
      errDesc = cudaGetErrorString((cudaError_t)res);
    }

    if (!errDesc) {
      ss << "No error string available" << endl;
    } else {
      ss << errDesc << endl;
    }

    throw runtime_error(ss.str());
  }
};

bool Buffer::Allocate() {
  if (GetRawMemSize()) {
    if (context) {
      CudaCtxPush lock(context);
      auto res = cuMemAllocHost(&pRawData, GetRawMemSize());
      ThrowOnCudaError(res, __LINE__);
    } else {
      pRawData = calloc(GetRawMemSize(), sizeof(uint8_t));
    }

    return (nullptr != pRawData);
  }
  return true;
}

void Buffer::Deallocate() {
  if (own_memory) {
    if (context) {
      auto const res = cuMemFreeHost(pRawData);
      ThrowOnCudaError(res, __LINE__);
    } else {
      free(pRawData);
    }
  }
  pRawData = nullptr;
}

void *Buffer::GetRawMemPtr() { return pRawData; }

const void *Buffer::GetRawMemPtr() const { return pRawData; }

void Buffer::Update(size_t newSize, void *newPtr) {
  Deallocate();

  mem_size = newSize;
  if (own_memory) {
    Allocate();
    if (newPtr) {
      memcpy(GetRawMemPtr(), newPtr, newSize);
    }
  } else {
    pRawData = newPtr;
  }
}

Buffer *Buffer::MakeOwnMem(size_t bufferSize, CUcontext ctx) {
  return new Buffer(bufferSize, true, ctx);
}

Buffer *Buffer::MakeOwnMem(size_t bufferSize, const void *pCopyFrom,
                           CUcontext ctx) {
  return new Buffer(bufferSize, pCopyFrom, ctx);
}

SurfacePlane::SurfacePlane() = default;

SurfacePlane &SurfacePlane::operator=(const SurfacePlane &other) {
  Deallocate();

  ownMem = false;
  gpuMem = other.gpuMem;
  width = other.width;
  height = other.height;
  pitch = other.pitch;
  elemSize = other.elemSize;

#ifdef TRACK_TOKEN_ALLOCATIONS
  id = other.id;
#endif

  return *this;
}

SurfacePlane::SurfacePlane(const SurfacePlane &other)
    : ownMem(false), gpuMem(other.gpuMem), width(other.width),
      height(other.height), pitch(other.pitch), elemSize(other.elemSize) {}

SurfacePlane::SurfacePlane(uint32_t newWidth, uint32_t newHeight,
                           uint32_t newPitch, uint32_t newElemSize,
                           CUdeviceptr pNewPtr)
    : ownMem(false), gpuMem(pNewPtr), width(newWidth), height(newHeight),
      pitch(newPitch), elemSize(newElemSize) {}

SurfacePlane::SurfacePlane(uint32_t newWidth, uint32_t newHeight,
                           uint32_t newElemSize, CUcontext context)
    : ownMem(true), width(newWidth), height(newHeight), elemSize(newElemSize),
      ctx(context) {
  Allocate();
}

SurfacePlane::~SurfacePlane() { Deallocate(); }

void SurfacePlane::Import(SurfacePlane &src, CUcontext ctx, CUstream str){
  bool same_size = Width() == src.Width();
  same_size |= Height() == src.Height();
  same_size |= Pitch() == src.Pitch();
  same_size |= ElemSize() == src.ElemSize();

  if(!same_size){
    return;
  }

  Import(src.GpuMem(), src.Pitch(), ctx, str);
}

void SurfacePlane::Export(SurfacePlane &dst, CUcontext ctx, CUstream str){
  bool same_size = Width() == dst.Width();
  same_size |= Height() == dst.Height();
  same_size |= Pitch() == dst.Pitch();
  same_size |= ElemSize() == dst.ElemSize();

  if(!same_size){
    return;
  }

  Export(dst.GpuMem(), dst.Pitch(), ctx, str);
}

void SurfacePlane::Import(CUdeviceptr src, uint32_t src_pitch, CUcontext ctx,
                          CUstream str)
{
  auto srcPlanePtr = src;
  auto dstPlanePtr = GpuMem();

  if (!srcPlanePtr || !dstPlanePtr) {
    return;
  }

  CudaCtxPush ctxPush(ctx);

  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.srcDevice = srcPlanePtr;
  m.dstDevice = dstPlanePtr;
  m.srcPitch = src_pitch;
  m.dstPitch = Pitch();
  m.Height = Height();
  m.WidthInBytes = Width() * ElemSize();

  ThrowOnCudaError(cuMemcpy2DAsync(&m, str), __LINE__);
  ThrowOnCudaError(cuStreamSynchronize(str), __LINE__);
}

void SurfacePlane::Export(CUdeviceptr dst, uint32_t dst_pitch, CUcontext ctx,
                          CUstream str)
{
  auto srcPlanePtr = GpuMem();
  auto dstPlanePtr = dst;

  if (!srcPlanePtr || !dstPlanePtr) {
    return;
  }

  CudaCtxPush ctxPush(ctx);

  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.srcDevice = srcPlanePtr;
  m.dstDevice = dstPlanePtr;
  m.srcPitch = Pitch();
  m.dstPitch = dst_pitch;
  m.Height = Height();
  m.WidthInBytes = Width() * ElemSize();

  ThrowOnCudaError(cuMemcpy2DAsync(&m, str), __LINE__);
  ThrowOnCudaError(cuStreamSynchronize(str), __LINE__);
}

SurfacePlane::SurfacePlane(uint32_t newWidth, uint32_t newHeight,
                           uint32_t newElemSize, uint32_t srcPitch, 
                           CUdeviceptr src, CUcontext context, CUstream str) 
  : SurfacePlane(newWidth, newHeight, newElemSize, context) {
  Import(src, srcPitch, context, str);
}

void SurfacePlane::Allocate() {
  if (!OwnMemory()) {
    return;
  }

  size_t newPitch;
  CudaCtxPush ctxPush(ctx);
  auto res = cuMemAllocPitch(&gpuMem, &newPitch, width * elemSize, height, 16);
  ThrowOnCudaError(res, __LINE__);
  pitch = newPitch;

#ifdef TRACK_TOKEN_ALLOCATIONS
  id = HWSurfaceRegister.AddNote(GpuMem());
#endif
}

void SurfacePlane::Deallocate() {
  if (!OwnMemory()) {
    return;
  }

#ifdef TRACK_TOKEN_ALLOCATIONS
  AllocInfo info(id, GpuMem());
  HWSurfaceRegister.DeleteNote(info);
#endif

  CudaCtxPush ctxPush(ctx);
  cuMemFree(gpuMem);
}

Surface::Surface() = default;

Surface::~Surface() = default;

Surface *Surface::Make(Pixel_Format format) {
  switch (format) {
  case Y:
    return new SurfaceY;
  case RGB:
    return new SurfaceRGB;
  case NV12:
    return new SurfaceNV12;
  case YUV420:
    return new SurfaceYUV420;
  case RGB_PLANAR:
    return new SurfaceRGBPlanar;
  case YCBCR:
    return new SurfaceYCbCr;
  case YUV444:
    return new SurfaceYUV444;
  default:
    return nullptr;
  }
}

Surface *Surface::Make(Pixel_Format format, uint32_t newWidth,
                       uint32_t newHeight, CUcontext context) {
  switch (format) {
  case Y:
    return new SurfaceY(newWidth, newHeight, context);
  case NV12:
    return new SurfaceNV12(newWidth, newHeight, context);
  case YUV420:
    return new SurfaceYUV420(newWidth, newHeight, context);
  case RGB:
    return new SurfaceRGB(newWidth, newHeight, context);
  case BGR:
    return new SurfaceBGR(newWidth, newHeight, context);
  case RGB_PLANAR:
    return new SurfaceRGBPlanar(newWidth, newHeight, context);
  case YCBCR:
    return new SurfaceYCbCr(newWidth, newHeight, context);
  case YUV444:
    return new SurfaceYUV444(newWidth, newHeight, context);
  default:
    return nullptr;
  }
}

SurfaceY::~SurfaceY() = default;

SurfaceY::SurfaceY() = default;

Surface *SurfaceY::Clone() { return new SurfaceY(*this); }

Surface *SurfaceY::Create() { return new SurfaceY; }

SurfaceY::SurfaceY(const SurfaceY &other) : plane(other.plane) {}

SurfaceY::SurfaceY(uint32_t width, uint32_t height, CUcontext context)
    : plane(width, height, ElemSize(), context) {}

SurfaceY &SurfaceY::operator=(const SurfaceY &other) {
  plane = other.plane;
  return *this;
}

bool SurfaceY::Update(SurfacePlane *pPlanes, size_t planesNum) {
  if (pPlanes && 1 == planesNum && !plane.OwnMemory()) {
    plane = *pPlanes;
    return true;
  }

  return false;
}

uint32_t SurfaceY::Width(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceY::WidthInBytes(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() * plane.ElemSize();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceY::Height(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Height();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceY::Pitch(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceY::HostMemSize() const { return plane.GetHostMemSize(); }

CUdeviceptr SurfaceY::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    return plane.GpuMem();
  }

  throw invalid_argument("Invalid plane number");
}

void SurfaceY::Update(const SurfacePlane &newPlane) { plane = newPlane; }

SurfacePlane *SurfaceY::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceNV12::~SurfaceNV12() = default;

SurfaceNV12::SurfaceNV12() = default;

SurfaceNV12::SurfaceNV12(const SurfaceNV12 &other) : plane(other.plane) {}

SurfaceNV12::SurfaceNV12(uint32_t width, uint32_t height, CUcontext context)
    : plane(width, height * 3 / 2, ElemSize(), context) {}

SurfaceNV12 &SurfaceNV12::operator=(const SurfaceNV12 &other) {
  plane = other.plane;
  return *this;
}

Surface *SurfaceNV12::Clone() { return new SurfaceNV12(*this); }

Surface *SurfaceNV12::Create() { return new SurfaceNV12; }

uint32_t SurfaceNV12::Width(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
  case 1:
    return plane.Width();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceNV12::WidthInBytes(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
  case 1:
    return plane.Width() * plane.ElemSize();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceNV12::Height(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
    return plane.Height() * 2 / 3;
  case 1:
    return plane.Height() / 3;
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceNV12::Pitch(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
  case 1:
    return plane.Pitch();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceNV12::HostMemSize() const { return plane.GetHostMemSize(); }

CUdeviceptr SurfaceNV12::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    return plane.GpuMem() + planeNumber * Height() * plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

void SurfaceNV12::Update(const SurfacePlane &newPlane) { plane = newPlane; }

bool SurfaceNV12::Update(SurfacePlane *pPlanes, size_t planesNum) {
  if (pPlanes && 1 == planesNum && !plane.OwnMemory()) {
    plane = *pPlanes;
    return true;
  }

  return false;
}

SurfacePlane *SurfaceNV12::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceYUV420::~SurfaceYUV420() = default;

SurfaceYUV420::SurfaceYUV420() = default;

SurfaceYUV420::SurfaceYUV420(const SurfaceYUV420 &other)
    : planeY(other.planeY), planeU(other.planeU), planeV(other.planeV) {}

SurfaceYUV420::SurfaceYUV420(uint32_t width, uint32_t height, CUcontext context)
    : planeY(width, height, ElemSize(), context),
      planeU(width / 2, height / 2, ElemSize(), context),
      planeV(width / 2, height / 2, ElemSize(), context) {}

SurfaceYUV420 &SurfaceYUV420::operator=(const SurfaceYUV420 &other) {
  planeY = other.planeY;
  planeU = other.planeU;
  planeV = other.planeV;

  return *this;
}

Surface *SurfaceYUV420::Clone() { return new SurfaceYUV420(*this); }

Surface *SurfaceYUV420::Create() { return new SurfaceYUV420; }

uint32_t SurfaceYUV420::Width(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
    return planeY.Width();
  case 1:
    return planeU.Width();
  case 2:
    return planeV.Width();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceYUV420::WidthInBytes(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
    return planeY.Width() * planeY.ElemSize();
  case 1:
    return planeU.Width() * planeU.ElemSize();
  case 2:
    return planeV.Width() * planeV.ElemSize();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceYUV420::Height(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
    return planeY.Height();
  case 1:
    return planeU.Height();
  case 2:
    return planeV.Height();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceYUV420::Pitch(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
    return planeY.Pitch();
  case 1:
    return planeU.Pitch();
  case 2:
    return planeV.Pitch();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceYUV420::HostMemSize() const {
  return planeY.GetHostMemSize() + planeU.GetHostMemSize() +
         planeV.GetHostMemSize();
}

CUdeviceptr SurfaceYUV420::PlanePtr(uint32_t planeNumber) {
  switch (planeNumber) {
  case 0:
    return planeY.GpuMem();
  case 1:
    return planeU.GpuMem();
  case 2:
    return planeV.GpuMem();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

void SurfaceYUV420::Update(const SurfacePlane &newPlaneY,
                           const SurfacePlane &newPlaneU,
                           const SurfacePlane &newPlaneV) {
  planeY = newPlaneY;
  planeU = newPlaneY;
  planeV = newPlaneV;
}

bool SurfaceYUV420::Update(SurfacePlane *pPlanes, size_t planesNum) {
  bool ownMemory =
      planeY.OwnMemory() || planeU.OwnMemory() || planeV.OwnMemory();

  if (pPlanes && 3 == planesNum && !ownMemory) {
    planeY = pPlanes[0];
    planeU = pPlanes[1];
    planeV = pPlanes[2];

    return true;
  }

  return false;
}

SurfacePlane *SurfaceYUV420::GetSurfacePlane(uint32_t planeNumber) {
  switch (planeNumber) {
  case 0U:
    return &planeY;
  case 1U:
    return &planeU;
  case 2U:
    return &planeV;
  default:
    return nullptr;
  }
}

SurfaceYCbCr::SurfaceYCbCr() : SurfaceYUV420() {}

SurfaceYCbCr::SurfaceYCbCr(const SurfaceYCbCr &other) : SurfaceYUV420(other) {}

SurfaceYCbCr::SurfaceYCbCr(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceYUV420(width, height, context) {}

Surface *VPF::SurfaceYCbCr::Clone() { return new SurfaceYCbCr(*this); }

Surface *VPF::SurfaceYCbCr::Create() { return new SurfaceYCbCr; }

SurfaceRGB::~SurfaceRGB() = default;

SurfaceRGB::SurfaceRGB() = default;

SurfaceRGB::SurfaceRGB(const SurfaceRGB &other) : plane(other.plane) {}

SurfaceRGB::SurfaceRGB(uint32_t width, uint32_t height, CUcontext context)
    : plane(width * 3, height, ElemSize(), context) {}

SurfaceRGB &SurfaceRGB::operator=(const SurfaceRGB &other) {
  plane = other.plane;
  return *this;
}

Surface *SurfaceRGB::Clone() { return new SurfaceRGB(*this); }

Surface *SurfaceRGB::Create() { return new SurfaceRGB; }

uint32_t SurfaceRGB::Width(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() / 3;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB::WidthInBytes(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() * plane.ElemSize();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB::Height(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Height();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB::Pitch(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB::HostMemSize() const { return plane.GetHostMemSize(); }

CUdeviceptr SurfaceRGB::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    return plane.GpuMem();
  }

  throw invalid_argument("Invalid plane number");
}

void SurfaceRGB::Update(const SurfacePlane &newPlane) { plane = newPlane; }

bool SurfaceRGB::Update(SurfacePlane *pPlanes, size_t planesNum) {
  if (pPlanes && 1 == planesNum && !plane.OwnMemory()) {
    plane = *pPlanes;
    return true;
  }

  return false;
}

SurfacePlane *SurfaceRGB::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceBGR::~SurfaceBGR() = default;

SurfaceBGR::SurfaceBGR() = default;

SurfaceBGR::SurfaceBGR(const SurfaceBGR &other) : plane(other.plane) {}

SurfaceBGR::SurfaceBGR(uint32_t width, uint32_t height, CUcontext context)
    : plane(width * 3, height, ElemSize(), context) {}

SurfaceBGR &SurfaceBGR::operator=(const SurfaceBGR &other) {
  plane = other.plane;
  return *this;
}

Surface *SurfaceBGR::Clone() { return new SurfaceBGR(*this); }

Surface *SurfaceBGR::Create() { return new SurfaceBGR; }

uint32_t SurfaceBGR::Width(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() / 3;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceBGR::WidthInBytes(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() * plane.ElemSize();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceBGR::Height(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Height();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceBGR::Pitch(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceBGR::HostMemSize() const { return plane.GetHostMemSize(); }

CUdeviceptr SurfaceBGR::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    return plane.GpuMem();
  }

  throw invalid_argument("Invalid plane number");
}

void SurfaceBGR::Update(const SurfacePlane &newPlane) { plane = newPlane; }

SurfacePlane *SurfaceBGR::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceRGBPlanar::~SurfaceRGBPlanar() = default;

SurfaceRGBPlanar::SurfaceRGBPlanar() = default;

SurfaceRGBPlanar::SurfaceRGBPlanar(const SurfaceRGBPlanar &other)
    : plane(other.plane) {}

SurfaceRGBPlanar::SurfaceRGBPlanar(uint32_t width, uint32_t height,
                                   CUcontext context)
    : plane(width, height * 3, ElemSize(), context) {}

SurfaceRGBPlanar &SurfaceRGBPlanar::operator=(const SurfaceRGBPlanar &other) {
  plane = other.plane;
  return *this;
}

Surface *SurfaceRGBPlanar::Clone() { return new SurfaceRGBPlanar(*this); }

Surface *SurfaceRGBPlanar::Create() { return new SurfaceRGBPlanar; }

uint32_t SurfaceRGBPlanar::Width(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGBPlanar::WidthInBytes(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() * plane.ElemSize();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGBPlanar::Height(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Height() / 3;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGBPlanar::Pitch(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGBPlanar::HostMemSize() const {
  return plane.GetHostMemSize();
}

CUdeviceptr SurfaceRGBPlanar::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    return plane.GpuMem() + planeNumber * Height() * plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

void SurfaceRGBPlanar::Update(const SurfacePlane &newPlane) {
  plane = newPlane;
}

bool SurfaceRGBPlanar::Update(SurfacePlane *pPlanes, size_t planesNum) {
  if (pPlanes && 1 == planesNum && !plane.OwnMemory()) {
    plane = *pPlanes;
    return true;
  }

  return false;
}

SurfacePlane *SurfaceRGBPlanar::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceYUV444::SurfaceYUV444() : SurfaceRGBPlanar() {}

SurfaceYUV444::SurfaceYUV444(const SurfaceYUV444 &other)
    : SurfaceRGBPlanar(other) {}

SurfaceYUV444::SurfaceYUV444(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceRGBPlanar(width, height, context) {}

Surface *VPF::SurfaceYUV444::Clone() { return new SurfaceYUV444(*this); }

Surface *VPF::SurfaceYUV444::Create() { return new SurfaceYUV444; }