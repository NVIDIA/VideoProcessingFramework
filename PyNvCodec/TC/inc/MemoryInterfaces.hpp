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
  RGB_PLANAR = 5,
  BGR = 6,
  YCBCR = 7,
  YUV444 = 8,
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
  const void *GetRawMemPtr() const;
  size_t GetRawMemSize() const;
  void Update(size_t newSize, void *newPtr = nullptr);
  template <typename T> T *GetDataAs() { return (T *)GetRawMemPtr(); }
  template <typename T> T const *GetDataAs() const {
    return (T const *)GetRawMemPtr();
  }

  static Buffer *Make(size_t bufferSize);
  static Buffer *Make(size_t bufferSize, void *pCopyFrom);

  static Buffer *MakeOwnMem(size_t bufferSize, CUcontext ctx = nullptr);
  static Buffer *MakeOwnMem(size_t bufferSize, const void *pCopyFrom,
                            CUcontext ctx = nullptr);

private:
  explicit Buffer(size_t bufferSize, bool ownMemory = true,
                  CUcontext ctx = nullptr);
  Buffer(size_t bufferSize, void *pCopyFrom, bool ownMemory,
         CUcontext ctx = nullptr);
  Buffer(size_t bufferSize, const void *pCopyFrom, CUcontext ctx = nullptr);
  bool Allocate();
  void Deallocate();

  bool own_memory = false;
  size_t mem_size = 0UL;
  void *pRawData = nullptr;
  CUcontext context = nullptr;
#ifdef TRACK_TOKEN_ALLOCATIONS
  uint32_t id;
#endif
};

/* RAII-style CUDA Context (un)lock;
 */
class DllExport CudaCtxPush final {
public:
  explicit CudaCtxPush(CUcontext ctx) { cuCtxPushCurrent(ctx); }
  ~CudaCtxPush() { cuCtxPopCurrent(nullptr); }
};

/* Surface plane class;
 * 2-dimensional GPU memory;
 * Doesn't have any format, just storafe for bytes;
 * Size in pixels are raw sizes.
 * E. g. RGB image will have single SurfacePlane which is 3x wide.
 */
struct DllExport SurfacePlane {
  CUdeviceptr gpuMem = 0UL;
  CUcontext ctx = nullptr;

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
  SurfacePlane(uint32_t newWidth, uint32_t newHeight, uint32_t newElemSize,
               CUcontext context);

  /* Construct & own memory. Copy from given pointer.
   */
  SurfacePlane(uint32_t newWidth, uint32_t newHeight, uint32_t newElemSize,
               uint32_t srcPitch, CUdeviceptr src, CUcontext context, CUstream str);

  /* Destruct, free memory if we own it;
   */
  ~SurfacePlane();

  /* Allocate memory if we own it;
   */
  void Allocate();

  /* Deallocate memory if we own it;
   */
  void Deallocate();

  /* Copy from SurfacePlane memory to given pointer.
   * User must check that memory allocation referenced by ptr is enough.
   */
  void Export(CUdeviceptr dst, uint32_t dst_pitch, CUcontext ctx, CUstream str);

  /* Copy to SurfacePlane memory from given pointer.
   * User must check that memory allocation referenced by ptr is enough.
   */
  void Import(CUdeviceptr src, uint32_t src_pitch, CUcontext ctx, CUstream str);

  /* Copy from SurfacePlane;
   */
  void Export(SurfacePlane &dst, CUcontext ctx, CUstream str);

  /* Copy to SurfacePlane;
   */
  void Import(SurfacePlane &src, CUcontext ctx, CUstream str);

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

  /* Get plane pitch in bytes;
   */
  inline uint32_t Pitch() const { return pitch; }

  /* Get element size in bytes;
   */
  inline uint32_t ElemSize() const { return elemSize; }

  /* Get amount of bytes in Host memory that is needed
   * to store image plane; */
  inline uint32_t GetHostMemSize() const { return width * height * elemSize; }

#ifdef TRACK_TOKEN_ALLOCATIONS
  uint64_t id = 0U;
#endif
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

  /* Returns element size in bytes;
   */
  virtual uint32_t ElemSize() const = 0;

  /* Returns total amount of memory in bytes needed
   * to store all pixels of Surface in Host memory;
   */
  virtual uint32_t HostMemSize() const = 0;

  /* Returns number of image planes;
   */
  virtual uint32_t NumPlanes() const = 0;

  virtual CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) = 0;

  virtual Pixel_Format PixelFormat() const = 0;

  virtual bool Empty() const = 0;

  virtual SurfacePlane *GetSurfacePlane(uint32_t planeNumber = 0U) = 0;

  /* Update from set of image planes, don't own the memory;
   */
  virtual bool Update(SurfacePlane *pPlanes, size_t planesNum) = 0;

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
                       uint32_t newHeight, CUcontext context);

protected:
  Surface();
};

/* 8-bit single plane image.
 */
class DllExport SurfaceY final : public Surface {
public:
  ~SurfaceY();

  SurfaceY();
  SurfaceY(const SurfaceY &other);
  SurfaceY(uint32_t width, uint32_t height, CUcontext context);
  SurfaceY &operator=(const SurfaceY &other);

  Surface *Clone() override;
  Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return Y; };
  uint32_t NumPlanes() const override { return 1U; };
  uint32_t ElemSize() const override { return sizeof(uint8_t); }
  bool Empty() const override { return 0UL == plane.GpuMem(); }

  void Update(const SurfacePlane &newPlane);
  bool Update(SurfacePlane *pPlanes, size_t planesNum) override;
  SurfacePlane *GetSurfacePlane(uint32_t planeNumber = 0U) override;

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
  SurfaceNV12(uint32_t width, uint32_t height, CUcontext context);
  SurfaceNV12 &operator=(const SurfaceNV12 &other);

  Surface *Clone() override;
  Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return NV12; }
  uint32_t NumPlanes() const override { return 2; }
  uint32_t ElemSize() const override { return sizeof(uint8_t); }
  bool Empty() const override { return 0UL == plane.GpuMem(); }

  void Update(const SurfacePlane &newPlane);
  bool Update(SurfacePlane *pPlanes, size_t planesNum) override;

  SurfacePlane *GetSurfacePlane(uint32_t planeNumber = 0U) override;

private:
  SurfacePlane plane;
};

/* 8-bit YUV420P image;
 */
class DllExport SurfaceYUV420 : public Surface {
public:
  ~SurfaceYUV420();

  SurfaceYUV420();
  SurfaceYUV420(const SurfaceYUV420 &other);
  SurfaceYUV420(uint32_t width, uint32_t height, CUcontext context);
  SurfaceYUV420 &operator=(const SurfaceYUV420 &other);

  virtual Surface *Clone() override;
  virtual Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  virtual Pixel_Format PixelFormat() const override { return YUV420; }
  uint32_t NumPlanes() const override { return 3; }
  uint32_t ElemSize() const override { return sizeof(uint8_t); }
  bool Empty() const override {
    return 0UL == planeY.GpuMem() && 0UL == planeU.GpuMem() &&
           0UL == planeV.GpuMem();
  }

  void Update(const SurfacePlane &newPlaneY, const SurfacePlane &newPlaneU,
              const SurfacePlane &newPlaneV);
  bool Update(SurfacePlane *pPlanes, size_t planesNum) override;
  SurfacePlane *GetSurfacePlane(uint32_t planeNumber = 0U) override;

private:
  SurfacePlane planeY;
  SurfacePlane planeU;
  SurfacePlane planeV;
};

class DllExport SurfaceYCbCr final : public SurfaceYUV420 {
public:
  Pixel_Format PixelFormat() const override { return YCBCR; }

  SurfaceYCbCr();
  SurfaceYCbCr(const SurfaceYCbCr &other);
  SurfaceYCbCr(uint32_t width, uint32_t height, CUcontext context);

  Surface *Clone() override;
  Surface *Create() override;
};

/* 8-bit RGB image;
 */
class DllExport SurfaceRGB : public Surface {
public:
  ~SurfaceRGB();

  SurfaceRGB();
  SurfaceRGB(const SurfaceRGB &other);
  SurfaceRGB(uint32_t width, uint32_t height, CUcontext context);
  SurfaceRGB &operator=(const SurfaceRGB &other);

  Surface *Clone() override;
  Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return RGB; }
  uint32_t NumPlanes() const override { return 1; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }
  bool Empty() const override { return 0UL == plane.GpuMem(); }

  void Update(const SurfacePlane &newPlane);
  bool Update(SurfacePlane *pPlanes, size_t planesNum) override;
  SurfacePlane *GetSurfacePlane(uint32_t planeNumber = 0U) override;

protected:
  SurfacePlane plane;
};

/* 8-bit BGR image;
 */
class DllExport SurfaceBGR : public SurfaceRGB {
public:
  ~SurfaceBGR();

  SurfaceBGR();
  SurfaceBGR(const SurfaceBGR &other);
  SurfaceBGR(uint32_t width, uint32_t height, CUcontext context);
  SurfaceBGR &operator=(const SurfaceBGR &other);

  Surface *Clone() override;
  Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return BGR; }
  uint32_t NumPlanes() const override { return 1; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }
  bool Empty() const override { return 0UL == plane.GpuMem(); }

  void Update(const SurfacePlane &newPlane);
  SurfacePlane *GetSurfacePlane(uint32_t planeNumber = 0U) override;

protected:
  SurfacePlane plane;
};

/* 8-bit planar RGB image;
 */
class DllExport SurfaceRGBPlanar : public Surface {
public:
  ~SurfaceRGBPlanar();

  SurfaceRGBPlanar();
  SurfaceRGBPlanar(const SurfaceRGBPlanar &other);
  SurfaceRGBPlanar(uint32_t width, uint32_t height, CUcontext context);
  SurfaceRGBPlanar &operator=(const SurfaceRGBPlanar &other);

  virtual Surface *Clone() override;
  virtual Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return RGB_PLANAR; }
  uint32_t NumPlanes() const override { return 3; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }
  bool Empty() const override { return 0UL == plane.GpuMem(); }

  void Update(const SurfacePlane &newPlane);
  bool Update(SurfacePlane *pPlanes, size_t planesNum) override;
  SurfacePlane *GetSurfacePlane(uint32_t planeNumber = 0U) override;

protected:
  SurfacePlane plane;
};

class DllExport SurfaceYUV444 : public SurfaceRGBPlanar {
public:
  Pixel_Format PixelFormat() const override { return YUV444; }

  SurfaceYUV444();
  SurfaceYUV444(const SurfaceYUV444 &other);
  SurfaceYUV444(uint32_t width, uint32_t height, CUcontext context);
  SurfaceYUV444 &operator=(const SurfaceYUV444 &other);

  Surface *Clone() override;
  Surface *Create() override;
};

#ifdef TRACK_TOKEN_ALLOCATIONS
/* Returns true if allocation counters are equal to zero, false otherwise;
 * If you want to check for dangling pointers, call this function at exit;
 */
bool DllExport CheckAllocationCounters();
#endif

} // namespace VPF