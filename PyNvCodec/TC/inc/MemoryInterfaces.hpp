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

#include "TC_CORE.hpp"
#include "nvEncodeAPI.h"
#include <cuda.h>

using namespace VPF;

namespace VPF {

enum Pixel_Format {
  UNDEFINED = 0,
  Y = 1,
  RGB = 2,
  NV12 = 3,
  YUV420 = 4,
};

/* Represents CPU-side memory.
 * May own the memory or be a wrapper around existing ponter;
 */
class DllExport Buffer final : public Token {
public:
  Buffer() = delete;
  Buffer(const Buffer &other) = delete;
  Buffer &operator=(Buffer &other) = delete;

  ~Buffer() final;
  void *GetRawMemPtr();
  size_t GetRawMemSize();
  void Update(size_t newSize, void *newPtr = nullptr);
  template <typename T> T *GetDataAs() { return (T *)GetRawMemPtr(); }

  static Buffer *Make(size_t bufferSize);
  static Buffer *Make(size_t bufferSize, void *pCopyFrom);
  static Buffer *MakeOwnMem(size_t bufferSize);

private:
  explicit Buffer(size_t bufferSize, bool ownMemory = true);
  Buffer(size_t bufferSize, void *pCopyFrom, bool ownMemory = true);
  bool Allocate();
  void Deallocate();

  bool own_memory = true;
  size_t mem_size = 0UL;
  void *pRawData = nullptr;
#ifdef TRACK_TOKEN_ALLOCATIONS
  uint32_t id;
#endif
};

/* RAII-style CUDA Context (un)lock;
 */
class DllExport CudaCtxLock final {
public:
  CudaCtxLock(CUcontext ctx) { cuCtxPushCurrent(ctx); }
  ~CudaCtxLock() { cuCtxPopCurrent(nullptr); }
};

/* Represents GPU-side memory.
 * Pure interface class, see ancestors;
 */
class DllExport Surface : public Token {
public:
  virtual ~Surface();

  /* Returns width in pixels;
   */
  virtual uint32_t Width(uint32_t planeNumber = 0U) const = 0;

  /* Returns width in bytes;
   */
  virtual uint32_t WidthInBytes(uint32_t planeNumber = 0U) const = 0;

  /* Returns height in pixels;
   */
  virtual uint32_t Height(uint32_t planeNumber = 0U) const = 0;

  /* Returns pitch in bytes;
   */
  virtual uint32_t Pitch(uint32_t planeNumber = 0U) const = 0;

  virtual uint32_t ElemSize() const = 0;

  virtual uint32_t NumPlanes() const = 0;

  virtual CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) = 0;

  virtual Pixel_Format PixelFormat() const = 0;

  virtual bool Empty() const = 0;

  /* Virtual copy constructor;
   */
  virtual Surface *Clone() = 0;

  /* Virtual default constructor;
   */
  virtual Surface *Create() = 0;

  /* Make empty;
   */
  static Surface *Make(Pixel_Format format);

  /* Make & own memory;
   */
  static Surface *Make(Pixel_Format format, uint32_t newWidth,
                       uint32_t newHeight);

protected:
  Surface();
};

/* Surface plane class;
 * 2-dimensional GPU memory;
 * Doesn't have any format, just storafe for bytes;
 * Size in pixels are raw sizes.
 * E. g. RGB image will have single SurfacePlane which is 3x wide.
 */
struct DllExport SurfacePlane {
  CUdeviceptr gpuMem = 0UL;

  uint32_t width = 0U;
  uint32_t height = 0U;
  uint32_t pitch = 0U;
  uint32_t elemSize = 0U;

  bool ownMem = false;

  /* Blank plane, zero size;
   */
  SurfacePlane();

  /* Update from existing, don't own memory;
   */
  SurfacePlane &operator=(const SurfacePlane &other);

  /* Construct from another, don't own memory;
   */
  SurfacePlane(const SurfacePlane &other);

  /* Construct from ptr & dimensions, don't own memory;
   */
  SurfacePlane(uint32_t newWidth, uint32_t newHeight, uint32_t newPitch,
               uint32_t newElemSize, CUdeviceptr pNewPtr);

  /* Construct & own memory;
   */
  SurfacePlane(uint32_t newWidth, uint32_t newHeight, uint32_t newElemSize);

  /* Destruct, free memory if we own it;
   */
  ~SurfacePlane();

  /* Allocate memory if we own it;
   */
  void Allocate();

  /* Deallocate memory if we own it;
   */
  void Deallocate();

  /* Returns true if class owns the memory, false otherwise;
   */
  inline bool OwnMemory() const { return ownMem; }

  /* Returns pointer to GPU memory object;
   */
  inline CUdeviceptr GpuMem() const { return gpuMem; }

  /* Get plane width in pixels;
   */
  inline uint32_t Width() const { return width; }

  /* Get plane height in pixels;
   */
  inline uint32_t Height() const { return height; }

  /* Get plane pitch in pixels;
   */
  inline uint32_t Pitch() const { return pitch; }

  /* Get element size in bytes;
   */
  inline uint32_t ElemSize() const { return elemSize; }

#ifdef TRACK_TOKEN_ALLOCATIONS
  uint64_t id;
#endif
};

/* 8-bit single plane image.
 */
class DllExport SurfaceY final : public Surface {
public:
  ~SurfaceY();

  SurfaceY();
  SurfaceY(const SurfaceY &other);
  SurfaceY(uint32_t width, uint32_t height);
  SurfaceY &operator=(const SurfaceY &other);

  Surface *Clone();
  Surface *Create();

  uint32_t Width(uint32_t planeNumber = 0U) const;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const;
  uint32_t Height(uint32_t planeNumber = 0U) const;
  uint32_t Pitch(uint32_t planeNumber = 0U) const;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U);
  Pixel_Format PixelFormat() const { return Y; };
  uint32_t NumPlanes() const { return 1U; };
  uint32_t ElemSize() const { return sizeof(uint8_t); }
  bool Empty() const { return 0UL == plane.GpuMem(); }

  void Update(SurfacePlane &newPlane);

private:
  SurfacePlane plane;
};

/* 8-bit NV12 image;
 */
class DllExport SurfaceNV12 final : public Surface {
public:
  ~SurfaceNV12();

  SurfaceNV12();
  SurfaceNV12(const SurfaceNV12 &other);
  SurfaceNV12(uint32_t width, uint32_t height);
  SurfaceNV12 &operator=(const SurfaceNV12 &other);

  Surface *Clone();
  Surface *Create();

  uint32_t Width(uint32_t planeNumber = 0U) const;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const;
  uint32_t Height(uint32_t planeNumber = 0U) const;
  uint32_t Pitch(uint32_t planeNumber = 0U) const;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U);
  Pixel_Format PixelFormat() const { return NV12; }
  uint32_t NumPlanes() const { return 2; }
  uint32_t ElemSize() const { return sizeof(uint8_t); }
  bool Empty() const { return 0UL == plane.GpuMem(); }

  void Update(SurfacePlane &newPlane);

private:
  SurfacePlane plane;
};

/* 8-bit YUV420P image;
 */
class DllExport SurfaceYUV420 final : public Surface {
public:
  ~SurfaceYUV420();

  SurfaceYUV420();
  SurfaceYUV420(const SurfaceYUV420 &other);
  SurfaceYUV420(uint32_t width, uint32_t height);
  SurfaceYUV420 &operator=(const SurfaceYUV420 &other);

  Surface *Clone();
  Surface *Create();

  uint32_t Width(uint32_t planeNumber = 0U) const;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const;
  uint32_t Height(uint32_t planeNumber = 0U) const;
  uint32_t Pitch(uint32_t planeNumber = 0U) const;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U);
  Pixel_Format PixelFormat() const { return YUV420; }
  uint32_t NumPlanes() const { return 3; }
  uint32_t ElemSize() const { return sizeof(uint8_t); }
  bool Empty() const {
    return 0UL == planeY.GpuMem() && 0UL == planeU.GpuMem() &&
           0UL == planeV.GpuMem();
  }

  void Update(SurfacePlane &newPlaneY, SurfacePlane &newPlaneU,
              SurfacePlane &newPlaneV);

private:
  SurfacePlane planeY;
  SurfacePlane planeU;
  SurfacePlane planeV;
};

/* 8-bit RGB image;
 */
class DllExport SurfaceRGB final : public Surface {
public:
  ~SurfaceRGB();

  SurfaceRGB();
  SurfaceRGB(const SurfaceRGB &other);
  SurfaceRGB(uint32_t width, uint32_t height);
  SurfaceRGB &operator=(const SurfaceRGB &other);

  Surface *Clone();
  Surface *Create();

  uint32_t Width(uint32_t planeNumber = 0U) const;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const;
  uint32_t Height(uint32_t planeNumber = 0U) const;
  uint32_t Pitch(uint32_t planeNumber = 0U) const;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U);
  Pixel_Format PixelFormat() const { return RGB; }
  uint32_t NumPlanes() const { return 1; }
  uint32_t ElemSize() const { return sizeof(uint8_t); }
  bool Empty() const { return 0UL == plane.GpuMem(); }

  void Update(SurfacePlane &newPlane);

private:
  SurfacePlane plane;
};

#ifdef TRACK_TOKEN_ALLOCATIONS
/* Returns true if allocation counters are equal to zero, false otherwise;
 * If you want to check for dangling pointers, call this function at exit;
 */
bool DllExport CheckAllocationCounters();
#endif

} // namespace VPF