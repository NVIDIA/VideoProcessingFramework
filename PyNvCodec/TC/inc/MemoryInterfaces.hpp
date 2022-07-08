/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Videonetics Technology Private Limited
 *
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
  RGB_32F = 9,
  RGB_32F_PLANAR = 10,
  YUV422 = 11,
  P10 = 12,
  P12 = 13,
};

enum ColorSpace {
  BT_601 = 0,
  BT_709 = 1,
  UNSPEC = 2,
};

enum ColorRange {
  MPEG = 0, /* Narrow range.*/
  JPEG = 1, /* Full range. */
  UDEF = 2,
};

struct ColorspaceConversionContext {
  ColorSpace color_space;
  ColorRange color_range;

  ColorspaceConversionContext() : color_space(UNSPEC), color_range(UDEF) {}

  ColorspaceConversionContext(ColorSpace cspace, ColorRange crange)
      : color_space(cspace), color_range(crange) {}
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
  bool CopyFrom(size_t size, void const *ptr);
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

  bool own_memory = true;
  size_t mem_size = 0UL;
  void *pRawData = nullptr;
  CUcontext context = nullptr;
#ifdef TRACK_TOKEN_ALLOCATIONS
  uint32_t id;
#endif
};

class DllExport CudaBuffer final : public Token {
public:
  CudaBuffer() = delete;
  CudaBuffer(const CudaBuffer& other) = delete;
  CudaBuffer& operator=(CudaBuffer& other) = delete;

  static CudaBuffer* Make(size_t elemSize, size_t numElems, CUcontext context);
  static CudaBuffer* Make(const void* ptr, size_t elemSize, size_t numElems,
                          CUcontext context, CUstream str);
  CudaBuffer* Clone();

  size_t GetRawMemSize() const { return elem_size * num_elems; }
  size_t GetNumElems() const { return num_elems; }
  size_t GetElemSize() const { return elem_size; }
  CUdeviceptr GpuMem() { return gpuMem; }
  ~CudaBuffer();

private:
  CudaBuffer(size_t elemSize, size_t numElems, CUcontext context);
  CudaBuffer(const void* ptr, size_t elemSize, size_t numElems,
             CUcontext context, CUstream str);
  bool Allocate();
  void Deallocate();

  CUdeviceptr gpuMem = 0UL;
  CUcontext ctx = nullptr;
  size_t elem_size = 0U;
  size_t num_elems = 0U;

#ifdef TRACK_TOKEN_ALLOCATIONS
  uint64_t id = 0U;
#endif
};

/* RAII-style CUDA Context (un)lock;
 */
class DllExport CudaCtxPush final {
public:
  explicit CudaCtxPush(CUcontext ctx) { cuCtxPushCurrent(ctx); }
  ~CudaCtxPush() { cuCtxPopCurrent(nullptr); }
};

/* RAII-style CUDA Context sync;
 */
class DllExport CudaStrSync final {
  CUstream str;
public:
  explicit CudaStrSync(CUstream stream) {str = stream;}
  ~CudaStrSync() { cuStreamSynchronize(str); }
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
  void Export(CUdeviceptr dst, uint32_t dst_pitch, CUcontext ctx, CUstream str,
              uint32_t roi_x, uint32_t roi_y, uint32_t roi_w, uint32_t roi_h,
              uint32_t pos_x, uint32_t pos_y);

  /* Copy to SurfacePlane memory from given pointer.
   * User must check that memory allocation referenced by ptr is enough.
   */
  void Import(CUdeviceptr src, uint32_t src_pitch, CUcontext ctx, CUstream str);
  void Import(CUdeviceptr src, uint32_t src_pitch, CUcontext ctx, CUstream str,
              uint32_t roi_x, uint32_t roi_y, uint32_t roi_w, uint32_t roi_h,
              uint32_t pos_x, uint32_t pos_y);

  /* Copy from SurfacePlane;
   */
  void Export(SurfacePlane& dst, CUcontext ctx, CUstream str);
  void Export(SurfacePlane& dst, CUcontext ctx, CUstream str, uint32_t roi_x,
              uint32_t roi_y, uint32_t roi_w, uint32_t roi_h, uint32_t pos_x,
              uint32_t pos_y);

  /* Copy to SurfacePlane;
   */
  void Import(SurfacePlane& src, CUcontext ctx, CUstream str);
  void Import(SurfacePlane& src, CUcontext ctx, CUstream str, uint32_t roi_x,
              uint32_t roi_y, uint32_t roi_w, uint32_t roi_h, uint32_t pos_x,
              uint32_t pos_y);

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

  /* Get CUDA context associated with memory object;
   */
  CUcontext GetContext() const
  {
    CUcontext ctx;
    cuPointerGetAttribute((void*)&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, GpuMem());
    return ctx;
  }

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

  /* Import from another Surface.
   * Given ROI within src will be copied to (pos_x; pos_y) of self.
   */
  void Import(Surface& src, CUcontext ctx, CUstream str, uint32_t roi_x,
                      uint32_t roi_y, uint32_t roi_w, uint32_t roi_h,
                      uint32_t pos_x, uint32_t pos_y);

  /* Export to another Surface.
   * Given ROI within self will be copied to (pos_x; pos_y) of dst.
   */
  void Export(Surface& dst, CUcontext ctx, CUstream str, uint32_t roi_x,
                      uint32_t roi_y, uint32_t roi_w, uint32_t roi_h,
                      uint32_t pos_x, uint32_t pos_y);

  /* Get associated CUDA context;
   */
  CUcontext Context() { return GetSurfacePlane()->GetContext(); }

  bool OwnMemory();

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
class DllExport SurfaceNV12 : public Surface {
public:
  ~SurfaceNV12();

  SurfaceNV12();
  SurfaceNV12(const SurfaceNV12 &other);
  SurfaceNV12(uint32_t width, uint32_t height, CUcontext context);
  SurfaceNV12 &operator=(const SurfaceNV12 &other);

  virtual Surface *Clone() override;
  virtual Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  virtual Pixel_Format PixelFormat() const override { return NV12; }
  uint32_t NumPlanes() const override { return 2; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }
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
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }
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

class DllExport SurfaceP10 : public SurfaceNV12 {
public:
  virtual uint32_t ElemSize() const override { return sizeof(uint16_t); }
  virtual Pixel_Format PixelFormat() const override { return P10; }

  SurfaceP10();
  SurfaceP10(const SurfaceP10& other);
  SurfaceP10(uint32_t width, uint32_t height, CUcontext context);
  SurfaceP10& operator=(const SurfaceP10& other);

  Surface* Clone() override;
  Surface* Create() override;
};

class DllExport SurfaceP12 : public SurfaceNV12 {
public:
  virtual uint32_t ElemSize() const override { return sizeof(uint16_t); }
  virtual Pixel_Format PixelFormat() const override { return P12; }

  SurfaceP12();
  SurfaceP12(const SurfaceP12& other);
  SurfaceP12(uint32_t width, uint32_t height, CUcontext context);
  SurfaceP12& operator=(const SurfaceP12& other);

  Surface* Clone() override;
  Surface* Create() override;
};

class DllExport SurfaceYUV422 : public Surface {
public:
  ~SurfaceYUV422();

  SurfaceYUV422();
  SurfaceYUV422(const SurfaceYUV422 &other);
  SurfaceYUV422(uint32_t width, uint32_t height, CUcontext context);
  SurfaceYUV422 &operator=(const SurfaceYUV422 &other);

  virtual Surface *Clone() override;
  virtual Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  virtual Pixel_Format PixelFormat() const override { return YUV422; }
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

/* 32-bit float RGB image;
 */
class DllExport SurfaceRGB32F : public Surface {
public:
  ~SurfaceRGB32F();

  SurfaceRGB32F();
  SurfaceRGB32F(const SurfaceRGB32F &other);
  SurfaceRGB32F(uint32_t width, uint32_t height, CUcontext context);
  SurfaceRGB32F &operator=(const SurfaceRGB32F &other);

  Surface *Clone() override;
  Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return RGB_32F; }
  uint32_t NumPlanes() const override { return 1; }
  virtual uint32_t ElemSize() const override { return sizeof(float); }
  bool Empty() const override { return 0UL == plane.GpuMem(); }

  void Update(const SurfacePlane &newPlane);
  bool Update(SurfacePlane *pPlanes, size_t planesNum) override;
  SurfacePlane *GetSurfacePlane(uint32_t planeNumber = 0U) override;

protected:
  SurfacePlane plane;
};

/* 32-bit float planar RGB image;
 */
class DllExport SurfaceRGB32FPlanar : public Surface {
public:
  ~SurfaceRGB32FPlanar();

  SurfaceRGB32FPlanar();
  SurfaceRGB32FPlanar(const SurfaceRGB32FPlanar &other);
  SurfaceRGB32FPlanar(uint32_t width, uint32_t height, CUcontext context);
  SurfaceRGB32FPlanar &operator=(const SurfaceRGB32FPlanar &other);

  virtual Surface *Clone() override;
  virtual Surface *Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  uint32_t HostMemSize() const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return RGB_32F_PLANAR; }
  uint32_t NumPlanes() const override { return 3; }
  virtual uint32_t ElemSize() const override { return sizeof(float); }
  bool Empty() const override { return 0UL == plane.GpuMem(); }

  void Update(const SurfacePlane &newPlane);
  bool Update(SurfacePlane *pPlanes, size_t planesNum) override;
  SurfacePlane *GetSurfacePlane(uint32_t planeNumber = 0U) override;

protected:
  SurfacePlane plane;
};

} // namespace VPF
