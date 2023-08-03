/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Videonetics Technology Private Limited
 * Copyright 2023 VisionLabs LLC
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

#include <chrono>
#include <fstream>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>

#include "CodecsSupport.hpp"
#include "MemoryInterfaces.hpp"
#include "NppCommon.hpp"
#include "Tasks.hpp"

#include "NvCodecCLIOptions.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

using namespace VPF;
using namespace std;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

namespace VPF
{

static auto ThrowOnCudaError = [](CUresult res, int lineNum = -1) {
  if (CUDA_SUCCESS != res) {
    stringstream ss;

    if (lineNum > 0) {
      ss << __FILE__ << ":";
      ss << lineNum << endl;
    }

    const char* errName = nullptr;
    if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {
      ss << "CUDA error with code " << res << endl;
    } else {
      ss << "CUDA error: " << errName << endl;
    }

    const char* errDesc = nullptr;
    cuGetErrorString(res, &errDesc);

    if (!errDesc) {
      ss << "No error string available" << endl;
    } else {
      ss << errDesc << endl;
    }

    throw runtime_error(ss.str());
  }
};

auto const cuda_stream_sync = [](void* stream) {
  cuStreamSynchronize((CUstream)stream);
};
} // namespace VPF

namespace VPF
{

#define TC_PIXEL_FORMAT_STRINGIFY(s) TC_PIXEL_FORMAT_STRINGIFY_(s)
#define TC_PIXEL_FORMAT_STRINGIFY_(s) #s##sv
#define TC_PIXEL_FORMAT_ENUM_CASE(s)                                           \
  case s:                                                                      \
    return TC_PIXEL_FORMAT_STRINGIFY(s)

auto const format_name = [](Pixel_Format format) {
  using namespace std::literals::string_view_literals;

  switch (format) {
    TC_PIXEL_FORMAT_ENUM_CASE(UNDEFINED);
    TC_PIXEL_FORMAT_ENUM_CASE(Y);
    TC_PIXEL_FORMAT_ENUM_CASE(RGB);
    TC_PIXEL_FORMAT_ENUM_CASE(NV12);
    TC_PIXEL_FORMAT_ENUM_CASE(YUV420);
    TC_PIXEL_FORMAT_ENUM_CASE(YCBCR);
    TC_PIXEL_FORMAT_ENUM_CASE(YUV444);
    TC_PIXEL_FORMAT_ENUM_CASE(RGB_32F);
    TC_PIXEL_FORMAT_ENUM_CASE(RGB_32F_PLANAR);
    TC_PIXEL_FORMAT_ENUM_CASE(YUV422);
  default:
    throw std::runtime_error("Invalid variant for Pixel_Format constructed!");
  }
};

static size_t GetElemSize(Pixel_Format format)
{
  stringstream ss;

  switch (format) {
  case RGB_PLANAR:
  case YUV444:
  case YUV420:
  case YUV422:
  case YCBCR:
  case NV12:
  case RGB:
  case BGR:
  case Y:
    return sizeof(uint8_t);
  case P10:
  case P12:
  case YUV420_10bit:
  case YUV444_10bit:
    return sizeof(uint16_t);
  case RGB_32F:
  case RGB_32F_PLANAR:
    return sizeof(float);
  default:
    ss << __FUNCTION__;
    ss << ": unsupported pixel format: " << format_name(format);
    throw invalid_argument(ss.str());
  }
}

struct CudaUploadFrame_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  Surface* pSurface = nullptr;
  Pixel_Format pixelFormat;

  CudaUploadFrame_Impl() = delete;
  CudaUploadFrame_Impl(const CudaUploadFrame_Impl& other) = delete;
  CudaUploadFrame_Impl& operator=(const CudaUploadFrame_Impl& other) = delete;

  CudaUploadFrame_Impl(CUstream stream, CUcontext context, uint32_t _width,
                       uint32_t _height, Pixel_Format _pix_fmt)
      : cuStream(stream), cuContext(context), pixelFormat(_pix_fmt)
  {
    pSurface = Surface::Make(pixelFormat, _width, _height, context);
  }

  ~CudaUploadFrame_Impl() { delete pSurface; }
};
} // namespace VPF

CudaUploadFrame* CudaUploadFrame::Make(CUstream cuStream, CUcontext cuContext,
                                       uint32_t width, uint32_t height,
                                       Pixel_Format pixelFormat)
{
  return new CudaUploadFrame(cuStream, cuContext, width, height, pixelFormat);
}

CudaUploadFrame::CudaUploadFrame(CUstream cuStream, CUcontext cuContext,
                                 uint32_t width, uint32_t height,
                                 Pixel_Format pix_fmt)
    :

      Task("CudaUploadFrame", CudaUploadFrame::numInputs,
           CudaUploadFrame::numOutputs, cuda_stream_sync, (void*)cuStream)
{
  pImpl = new CudaUploadFrame_Impl(cuStream, cuContext, width, height, pix_fmt);
}

CudaUploadFrame::~CudaUploadFrame() { delete pImpl; }

TaskExecStatus CudaUploadFrame::Run()
{
  NvtxMark tick(GetName());
  if (!GetInput()) {
    return TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pSurface = pImpl->pSurface;
  auto pSrcHost = ((Buffer*)GetInput())->GetDataAs<uint8_t>();

  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = CU_MEMORYTYPE_HOST;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;

  for (auto plane = 0; plane < pSurface->NumPlanes(); plane++) {
    CudaCtxPush lock(context);

    m.srcHost = pSrcHost;
    m.srcPitch = pSurface->WidthInBytes(plane);
    m.dstDevice = pSurface->PlanePtr(plane);
    m.dstPitch = pSurface->Pitch(plane);
    m.WidthInBytes = pSurface->WidthInBytes(plane);
    m.Height = pSurface->Height(plane);

    if (CUDA_SUCCESS != cuMemcpy2DAsync(&m, stream)) {
      return TASK_EXEC_FAIL;
    }

    pSrcHost += m.WidthInBytes * m.Height;
  }

  SetOutput(pSurface, 0);
  return TASK_EXEC_SUCCESS;
}

namespace VPF
{
struct UploadBuffer_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  CudaBuffer* pBuffer = nullptr;

  UploadBuffer_Impl() = delete;
  UploadBuffer_Impl(const UploadBuffer_Impl& other) = delete;
  UploadBuffer_Impl& operator=(const UploadBuffer_Impl& other) = delete;

  UploadBuffer_Impl(CUstream stream, CUcontext context, uint32_t elem_size,
                    uint32_t num_elems)
      : cuStream(stream), cuContext(context)
  {
    pBuffer = CudaBuffer::Make(elem_size, num_elems, context);
  }

  ~UploadBuffer_Impl() { delete pBuffer; }
};
} // namespace VPF

UploadBuffer* UploadBuffer::Make(CUstream cuStream, CUcontext cuContext,
                                 uint32_t elem_size, uint32_t num_elems)
{
  return new UploadBuffer(cuStream, cuContext, elem_size, num_elems);
}

UploadBuffer::UploadBuffer(CUstream cuStream, CUcontext cuContext,
                           uint32_t elem_size, uint32_t num_elems)
    :

      Task("UploadBuffer", UploadBuffer::numInputs, UploadBuffer::numOutputs,
           cuda_stream_sync, (void*)cuStream)
{
  pImpl = new UploadBuffer_Impl(cuStream, cuContext, elem_size, num_elems);
}

UploadBuffer::~UploadBuffer() { delete pImpl; }

TaskExecStatus UploadBuffer::Run()
{
  NvtxMark tick(GetName());
  if (!GetInput()) {
    return TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pBuffer = pImpl->pBuffer;
  auto pSrcHost = ((Buffer*)GetInput())->GetDataAs<void>();

  CudaCtxPush lock(context);
  if (CUDA_SUCCESS != cuMemcpyHtoDAsync(pBuffer->GpuMem(),
                                        (const void*)pSrcHost,
                                        pBuffer->GetRawMemSize(), stream)) {
    return TASK_EXEC_FAIL;
  }

  SetOutput(pBuffer, 0);
  return TASK_EXEC_SUCCESS;
}

namespace VPF
{
struct CudaDownloadSurface_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  Pixel_Format format;
  Buffer* pHostFrame = nullptr;

  CudaDownloadSurface_Impl() = delete;
  CudaDownloadSurface_Impl(const CudaDownloadSurface_Impl& other) = delete;
  CudaDownloadSurface_Impl&
  operator=(const CudaDownloadSurface_Impl& other) = delete;

  CudaDownloadSurface_Impl(CUstream stream, CUcontext context, uint32_t _width,
                           uint32_t _height, Pixel_Format _pix_fmt)
      : cuStream(stream), cuContext(context), format(_pix_fmt)
  {
    auto bufferSize = _width * _height * GetElemSize(_pix_fmt);
    stringstream ss;

    if (YUV420 == _pix_fmt || NV12 == _pix_fmt || YCBCR == _pix_fmt ||
        P10 == _pix_fmt || P12 == _pix_fmt || YUV420_10bit == _pix_fmt) {
      bufferSize = bufferSize * 3U / 2U;
    } else if (RGB == _pix_fmt || RGB_PLANAR == _pix_fmt || BGR == _pix_fmt ||
               YUV444 == _pix_fmt || RGB_32F == _pix_fmt ||
               RGB_32F_PLANAR == _pix_fmt || YUV444_10bit == _pix_fmt ) {
      bufferSize = bufferSize * 3U;
    } else if (YUV422 == _pix_fmt) {
      bufferSize = bufferSize * 2U;
    } else if (Y == _pix_fmt) {
    } else {
      stringstream ss;
      ss << __FUNCTION__ << ": unsupported pixel format: " << _pix_fmt << endl;
      throw invalid_argument(ss.str());
    }

    pHostFrame = Buffer::MakeOwnMem(bufferSize, context);
  }

  ~CudaDownloadSurface_Impl() { delete pHostFrame; }
};

struct DownloadCudaBuffer_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  Buffer* pHostBuffer = nullptr;

  DownloadCudaBuffer_Impl() = delete;
  DownloadCudaBuffer_Impl(const DownloadCudaBuffer_Impl& other) = delete;
  DownloadCudaBuffer_Impl&
  operator=(const DownloadCudaBuffer_Impl& other) = delete;

  DownloadCudaBuffer_Impl(CUstream stream, CUcontext context,
                          uint32_t elem_size, uint32_t num_elems)
      : cuStream(stream), cuContext(context)
  {
    pHostBuffer = Buffer::MakeOwnMem(elem_size * num_elems, context);
  }

  ~DownloadCudaBuffer_Impl() { delete pHostBuffer; }
};
} // namespace VPF

CudaDownloadSurface* CudaDownloadSurface::Make(CUstream cuStream,
                                               CUcontext cuContext,
                                               uint32_t width, uint32_t height,
                                               Pixel_Format pixelFormat)
{
  return new CudaDownloadSurface(cuStream, cuContext, width, height,
                                 pixelFormat);
}

CudaDownloadSurface::CudaDownloadSurface(CUstream cuStream, CUcontext cuContext,
                                         uint32_t width, uint32_t height,
                                         Pixel_Format pix_fmt)
    :

      Task("CudaDownloadSurface", CudaDownloadSurface::numInputs,
           CudaDownloadSurface::numOutputs, cuda_stream_sync, (void*)cuStream)
{
  pImpl =
      new CudaDownloadSurface_Impl(cuStream, cuContext, width, height, pix_fmt);
}

CudaDownloadSurface::~CudaDownloadSurface() { delete pImpl; }

TaskExecStatus CudaDownloadSurface::Run()
{
  NvtxMark tick(GetName());

  if (!GetInput()) {
    return TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pSurface = (Surface*)GetInput();
  auto pDstHost = ((Buffer*)pImpl->pHostFrame)->GetDataAs<uint8_t>();

  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstMemoryType = CU_MEMORYTYPE_HOST;

  for (auto plane = 0; plane < pSurface->NumPlanes(); plane++) {
    CudaCtxPush lock(context);

    m.srcDevice = pSurface->PlanePtr(plane);
    m.srcPitch = pSurface->Pitch(plane);
    m.dstHost = pDstHost;
    m.dstPitch = pSurface->WidthInBytes(plane);
    m.WidthInBytes = pSurface->WidthInBytes(plane);
    m.Height = pSurface->Height(plane);

    auto const ret = cuMemcpy2DAsync(&m, stream);
    if (CUDA_SUCCESS != ret) {
      return TASK_EXEC_FAIL;
    }

    pDstHost += m.WidthInBytes * m.Height;
  }

  SetOutput(pImpl->pHostFrame, 0);
  return TASK_EXEC_SUCCESS;
}

DownloadCudaBuffer* DownloadCudaBuffer::Make(CUstream cuStream,
                                             CUcontext cuContext,
                                             uint32_t elem_size,
                                             uint32_t num_elems)
{
  return new DownloadCudaBuffer(cuStream, cuContext, elem_size, num_elems);
}

DownloadCudaBuffer::DownloadCudaBuffer(CUstream cuStream, CUcontext cuContext,
                                       uint32_t elem_size, uint32_t num_elems)
    : Task("DownloadCudaBuffer", DownloadCudaBuffer::numInputs,
           DownloadCudaBuffer::numOutputs, cuda_stream_sync, (void*)cuStream)
{
  pImpl =
      new DownloadCudaBuffer_Impl(cuStream, cuContext, elem_size, num_elems);
}

DownloadCudaBuffer::~DownloadCudaBuffer() { delete pImpl; }

TaskExecStatus DownloadCudaBuffer::Run()
{
  NvtxMark tick(GetName());

  if (!GetInput()) {
    return TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pCudaBuffer = (CudaBuffer*)GetInput();
  auto pDstHost = ((Buffer*)pImpl->pHostBuffer)->GetDataAs<void>();

  CudaCtxPush lock(context);
  if (CUDA_SUCCESS != cuMemcpyDtoHAsync(pDstHost, pCudaBuffer->GpuMem(),
                                        pCudaBuffer->GetRawMemSize(), stream)) {
    return TASK_EXEC_FAIL;
  }

  SetOutput(pImpl->pHostBuffer, 0);
  return TASK_EXEC_SUCCESS;
}

namespace VPF
{
struct DemuxFrame_Impl {
  size_t videoBytes = 0U;

  Buffer* pElementaryVideo;
  Buffer* pMuxingParams;
  Buffer* pSei;
  Buffer* pPktData;
  unique_ptr<FFmpegDemuxer> demuxer;
  unique_ptr<DataProvider> d_prov;

  DemuxFrame_Impl() = delete;
  DemuxFrame_Impl(const DemuxFrame_Impl& other) = delete;
  DemuxFrame_Impl& operator=(const DemuxFrame_Impl& other) = delete;

  explicit DemuxFrame_Impl(const string& url,
                           const map<string, string>& ffmpeg_options)
  {
    demuxer.reset(new FFmpegDemuxer(url.c_str(), ffmpeg_options));
    pElementaryVideo = Buffer::MakeOwnMem(0U);
    pMuxingParams = Buffer::MakeOwnMem(sizeof(MuxingParams));
    pSei = Buffer::MakeOwnMem(0U);
    pPktData = Buffer::MakeOwnMem(0U);
  }

  explicit DemuxFrame_Impl(istream& istr,
                           const map<string, string>& ffmpeg_options)
  {
    d_prov.reset(new DataProvider(istr));
    demuxer.reset(new FFmpegDemuxer(*d_prov.get(), ffmpeg_options));

    pElementaryVideo = Buffer::MakeOwnMem(0U);
    pMuxingParams = Buffer::MakeOwnMem(sizeof(MuxingParams));
    pSei = Buffer::MakeOwnMem(0U);
    pPktData = Buffer::MakeOwnMem(0U);
  }

  ~DemuxFrame_Impl()
  {
    delete pElementaryVideo;
    delete pMuxingParams;
    delete pSei;
    delete pPktData;
  }
};
} // namespace VPF

DemuxFrame* DemuxFrame::Make(istream& i_str, const char** ffmpeg_options,
                             uint32_t opts_size)
{
  return new DemuxFrame(i_str, ffmpeg_options, opts_size);
}

DemuxFrame* DemuxFrame::Make(const char* url, const char** ffmpeg_options,
                             uint32_t opts_size)
{
  return new DemuxFrame(url, ffmpeg_options, opts_size);
}

DemuxFrame::DemuxFrame(istream& i_str, const char** ffmpeg_options,
                       uint32_t opts_size)
    : Task("DemuxFrame", DemuxFrame::numInputs, DemuxFrame::numOutputs)
{
  map<string, string> options;
  if (0 == opts_size % 2) {
    for (auto i = 0; i < opts_size;) {
      auto key = string(ffmpeg_options[i]);
      i++;
      auto value = string(ffmpeg_options[i]);
      i++;

      options.insert(pair<string, string>(key, value));
    }
  }
  pImpl = new DemuxFrame_Impl(i_str, options);
}

DemuxFrame::DemuxFrame(const char* url, const char** ffmpeg_options,
                       uint32_t opts_size)
    : Task("DemuxFrame", DemuxFrame::numInputs, DemuxFrame::numOutputs)
{
  map<string, string> options;
  if (0 == opts_size % 2) {
    for (auto i = 0; i < opts_size;) {
      auto key = string(ffmpeg_options[i]);
      i++;
      auto value = string(ffmpeg_options[i]);
      i++;

      options.insert(pair<string, string>(key, value));
    }
  }
  pImpl = new DemuxFrame_Impl(url, options);
}

DemuxFrame::~DemuxFrame() { delete pImpl; }

void DemuxFrame::Flush() { pImpl->demuxer->Flush(); }

int64_t DemuxFrame::TsFromTime(double ts_sec)
{
  return pImpl->demuxer->TsFromTime(ts_sec);
}

int64_t DemuxFrame::TsFromFrameNumber(int64_t frame_num)
{
  return pImpl->demuxer->TsFromFrameNumber(frame_num);
}

TaskExecStatus DemuxFrame::Run()
{
  NvtxMark tick(GetName());
  ClearOutputs();

  uint8_t* pVideo = nullptr;
  MuxingParams params = {0};
  PacketData pkt_data = {0};

  auto& videoBytes = pImpl->videoBytes;
  auto& demuxer = pImpl->demuxer;

  uint8_t* pSEI = nullptr;
  size_t seiBytes = 0U;
  bool needSEI = (nullptr != GetInput(0U));

  auto pSeekCtxBuf = (Buffer*)GetInput(1U);
  if (pSeekCtxBuf) {
    SeekContext seek_ctx = *pSeekCtxBuf->GetDataAs<SeekContext>();
    auto ret = demuxer->Seek(seek_ctx, pVideo, videoBytes, pkt_data,
                             needSEI ? &pSEI : nullptr, &seiBytes);
    if (!ret) {
      return TASK_EXEC_FAIL;
    }
  } else if (!demuxer->Demux(pVideo, videoBytes, pkt_data,
                             needSEI ? &pSEI : nullptr, &seiBytes)) {
    return TASK_EXEC_FAIL;
  }

  if (videoBytes) {
    pImpl->pElementaryVideo->Update(videoBytes, pVideo);
    SetOutput(pImpl->pElementaryVideo, 0U);

    GetParams(params);
    pImpl->pMuxingParams->Update(sizeof(MuxingParams), &params);
    SetOutput(pImpl->pMuxingParams, 1U);
  }

  if (pSEI) {
    pImpl->pSei->Update(seiBytes, pSEI);
    SetOutput(pImpl->pSei, 2U);
  }

  pImpl->pPktData->Update(sizeof(pkt_data), &pkt_data);
  SetOutput((Token*)pImpl->pPktData, 3U);

  return TASK_EXEC_SUCCESS;
}

void DemuxFrame::GetParams(MuxingParams& params) const
{
  params.videoContext.width = pImpl->demuxer->GetWidth();
  params.videoContext.height = pImpl->demuxer->GetHeight();
  params.videoContext.num_frames = pImpl->demuxer->GetNumFrames();
  params.videoContext.frameRate = pImpl->demuxer->GetFramerate();
  params.videoContext.avgFrameRate = pImpl->demuxer->GetAvgFramerate();
  params.videoContext.is_vfr = pImpl->demuxer->IsVFR();
  params.videoContext.timeBase = pImpl->demuxer->GetTimebase();
  params.videoContext.streamIndex = pImpl->demuxer->GetVideoStreamIndex();
  params.videoContext.codec = FFmpeg2NvCodecId(pImpl->demuxer->GetVideoCodec());
  params.videoContext.gop_size = pImpl->demuxer->GetGopSize();

  switch (pImpl->demuxer->GetPixelFormat()) {
  case AV_PIX_FMT_YUVJ420P:
  case AV_PIX_FMT_YUV420P:
  case AV_PIX_FMT_NV12:
    params.videoContext.format = NV12;
    break;
  case AV_PIX_FMT_YUV444P16LE:
  case AV_PIX_FMT_YUV444P10LE:
    params.videoContext.format = YUV444_10bit;
    break;
  case AV_PIX_FMT_P016LE:
    params.videoContext.format = P12;
      break;
  case AV_PIX_FMT_YUV444P:
    params.videoContext.format = YUV444;
    break;
  case AV_PIX_FMT_YUV422P:
    params.videoContext.format = YUV422;
    break;
  case AV_PIX_FMT_YUV420P10:
    params.videoContext.format = P10;
    break;
  case AV_PIX_FMT_YUV420P12:
    params.videoContext.format = P12;
    break;
  default:
    stringstream ss;
    ss << "Unsupported FFmpeg pixel format: "
       << av_get_pix_fmt_name(pImpl->demuxer->GetPixelFormat()) << endl;
    throw invalid_argument(ss.str());
    params.videoContext.format = UNDEFINED;
    break;
  }

  switch (pImpl->demuxer->GetColorSpace()) {
  case AVCOL_SPC_BT709:
    params.videoContext.color_space = BT_709;
    break;
  case AVCOL_SPC_BT470BG:
  case AVCOL_SPC_SMPTE170M:
    params.videoContext.color_space = BT_601;
    break;
  default:
    params.videoContext.color_space = UNSPEC;
    break;
  }

  switch (pImpl->demuxer->GetColorRange()) {
  case AVCOL_RANGE_MPEG:
    params.videoContext.color_range = MPEG;
    break;
  case AVCOL_RANGE_JPEG:
    params.videoContext.color_range = JPEG;
    break;
  default:
    params.videoContext.color_range = UDEF;
    break;
  }
}

namespace VPF
{
struct ResizeSurface_Impl {
  Surface* pSurface = nullptr;
  CUcontext cu_ctx;
  CUstream cu_str;
  NppStreamContext nppCtx;

  ResizeSurface_Impl(uint32_t width, uint32_t height, Pixel_Format format,
                     CUcontext ctx, CUstream str)
      : cu_ctx(ctx), cu_str(str)
  {
    SetupNppContext(cu_ctx, cu_str, nppCtx);
  }

  virtual ~ResizeSurface_Impl() = default;

  virtual TaskExecStatus Run(Surface& source) = 0;
};

struct NppResizeSurfacePacked3C_Impl final : ResizeSurface_Impl {
  NppResizeSurfacePacked3C_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                                CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str)
  {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurfacePacked3C_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface& source)
  {
    NvtxMark tick("NppResizeSurfacePacked3C");

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    auto srcPlane = source.GetSurfacePlane();
    auto dstPlane = pSurface->GetSurfacePlane();

    const Npp8u* pSrc = (const Npp8u*)srcPlane->GpuMem();
    int nSrcStep = (int)source.Pitch();
    NppiSize oSrcSize = {0};
    oSrcSize.width = source.Width();
    oSrcSize.height = source.Height();
    NppiRect oSrcRectROI = {0};
    oSrcRectROI.width = oSrcSize.width;
    oSrcRectROI.height = oSrcSize.height;

    Npp8u* pDst = (Npp8u*)dstPlane->GpuMem();
    int nDstStep = (int)pSurface->Pitch();
    NppiSize oDstSize = {0};
    oDstSize.width = pSurface->Width();
    oDstSize.height = pSurface->Height();
    NppiRect oDstRectROI = {0};
    oDstRectROI.width = oDstSize.width;
    oDstRectROI.height = oDstSize.height;
    int eInterpolation = NPPI_INTER_LANCZOS;

    CudaCtxPush ctxPush(cu_ctx);
    auto ret = nppiResize_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                     pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppCtx);
    if (NPP_NO_ERROR != ret) {
      cerr << "Can't resize 3-channel packed image. Error code: " << ret
           << endl;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

// Resize planar 8 bit surface (YUV420, YCbCr420);
struct NppResizeSurfacePlanar_Impl final : ResizeSurface_Impl {
  NppResizeSurfacePlanar_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                              CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str)
  {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurfacePlanar_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface& source)
  {
    NvtxMark tick("NppResizeSurfacePlanar");

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      cerr << "Actual pixel format is " << source.PixelFormat() << endl;
      cerr << "Expected input format is " << pSurface->PixelFormat() << endl;
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    for (auto plane = 0; plane < pSurface->NumPlanes(); plane++) {
      auto srcPlane = source.GetSurfacePlane(plane);
      auto dstPlane = pSurface->GetSurfacePlane(plane);

      const Npp8u* pSrc = (const Npp8u*)srcPlane->GpuMem();
      int nSrcStep = (int)srcPlane->Pitch();
      NppiSize oSrcSize = {0};
      oSrcSize.width = srcPlane->Width();
      oSrcSize.height = srcPlane->Height();
      NppiRect oSrcRectROI = {0};
      oSrcRectROI.width = oSrcSize.width;
      oSrcRectROI.height = oSrcSize.height;

      Npp8u* pDst = (Npp8u*)dstPlane->GpuMem();
      int nDstStep = (int)dstPlane->Pitch();
      NppiSize oDstSize = {0};
      oDstSize.width = dstPlane->Width();
      oDstSize.height = dstPlane->Height();
      NppiRect oDstRectROI = {0};
      oDstRectROI.width = oDstSize.width;
      oDstRectROI.height = oDstSize.height;
      int eInterpolation = NPPI_INTER_LANCZOS;

      CudaCtxPush ctxPush(cu_ctx);
      auto ret = nppiResize_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                       pDst, nDstStep, oDstSize, oDstRectROI,
                                       eInterpolation, nppCtx);
      if (NPP_NO_ERROR != ret) {
        cerr << "NPP error with code " << ret << endl;
        return TASK_EXEC_FAIL;
      }
    }

    return TASK_EXEC_SUCCESS;
  }
};

// Resize semiplanar 8 bit NV12 surface;
struct ResizeSurfaceSemiPlanar_Impl final : ResizeSurface_Impl {
  ResizeSurfaceSemiPlanar_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                               CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str), _ctx(ctx),
        _str(str), last_h(0U), last_w(0U)
  {
    pSurface = nullptr;
    nv12_yuv420 = nullptr;
    pResizeYuv = ResizeSurface::Make(width, height, YUV420, ctx, str);
    yuv420_nv12 = ConvertSurface::Make(width, height, YUV420, NV12, ctx, str);
  }

  ~ResizeSurfaceSemiPlanar_Impl()
  {
    pSurface = nullptr;
    delete pResizeYuv;
    delete nv12_yuv420;
    delete yuv420_nv12;
  }

  TaskExecStatus Run(Surface& source)
  {
    NvtxMark tick("NppResizeSurfaceSemiPlanar");

    auto const resolution_change =
        source.Width() != last_w || source.Height() != last_h;

    if (nv12_yuv420 && resolution_change) {
      delete nv12_yuv420;
      nv12_yuv420 = nullptr;
    }

    if (!nv12_yuv420) {
      nv12_yuv420 = ConvertSurface::Make(source.Width(), source.Height(), NV12,
                                         YUV420, _ctx, _str);
    }

    // Convert from NV12 to YUV420;
    nv12_yuv420->SetInput((Token*)&source, 0U);
    if (TASK_EXEC_SUCCESS != nv12_yuv420->Execute())
      return TASK_EXEC_FAIL;
    auto surf_yuv420 = nv12_yuv420->GetOutput(0U);

    // Resize YUV420;
    pResizeYuv->SetInput(surf_yuv420, 0U);
    if (TASK_EXEC_SUCCESS != pResizeYuv->Execute())
      return TASK_EXEC_FAIL;
    auto surf_res = pResizeYuv->GetOutput(0U);

    // Convert back to NV12;
    yuv420_nv12->SetInput(surf_res, 0U);
    if (TASK_EXEC_SUCCESS != yuv420_nv12->Execute())
      return TASK_EXEC_FAIL;
    pSurface = (Surface*)yuv420_nv12->GetOutput(0U);

    last_w = source.Width();
    last_h = source.Height();

    return TASK_EXEC_SUCCESS;
  }

  ResizeSurface* pResizeYuv;
  ConvertSurface* nv12_yuv420;
  ConvertSurface* yuv420_nv12;
  CUcontext _ctx;
  CUstream _str;
  uint32_t last_w, last_h;
};

struct NppResizeSurfacePacked32F3C_Impl final : ResizeSurface_Impl {
  NppResizeSurfacePacked32F3C_Impl(uint32_t width, uint32_t height,
                                   CUcontext ctx, CUstream str,
                                   Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str)
  {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurfacePacked32F3C_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface& source)
  {
    NvtxMark tick("NppResizeSurfacePacked32F3C");

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    auto srcPlane = source.GetSurfacePlane();
    auto dstPlane = pSurface->GetSurfacePlane();

    const Npp32f* pSrc = (const Npp32f*)srcPlane->GpuMem();
    int nSrcStep = (int)source.Pitch();
    NppiSize oSrcSize = {0};
    oSrcSize.width = source.Width();
    oSrcSize.height = source.Height();
    NppiRect oSrcRectROI = {0};
    oSrcRectROI.width = oSrcSize.width;
    oSrcRectROI.height = oSrcSize.height;

    Npp32f* pDst = (Npp32f*)dstPlane->GpuMem();
    int nDstStep = (int)pSurface->Pitch();
    NppiSize oDstSize = {0};
    oDstSize.width = pSurface->Width();
    oDstSize.height = pSurface->Height();
    NppiRect oDstRectROI = {0};
    oDstRectROI.width = oDstSize.width;
    oDstRectROI.height = oDstSize.height;
    int eInterpolation = NPPI_INTER_LANCZOS;

    CudaCtxPush ctxPush(cu_ctx);
    auto ret = nppiResize_32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                      pDst, nDstStep, oDstSize, oDstRectROI,
                                      eInterpolation, nppCtx);
    if (NPP_NO_ERROR != ret) {
      cerr << "Can't resize 3-channel packed image. Error code: " << ret
           << endl;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

// Resize planar 8 bit surface (YUV420, YCbCr420);
struct NppResizeSurface32FPlanar_Impl final : ResizeSurface_Impl {
  NppResizeSurface32FPlanar_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                                 CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str)
  {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurface32FPlanar_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface& source)
  {
    NvtxMark tick("NppResizeSurface32FPlanar");

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      cerr << "Actual pixel format is " << source.PixelFormat() << endl;
      cerr << "Expected input format is " << pSurface->PixelFormat() << endl;
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    for (auto plane = 0; plane < pSurface->NumPlanes(); plane++) {
      auto srcPlane = source.GetSurfacePlane(plane);
      auto dstPlane = pSurface->GetSurfacePlane(plane);

      const Npp32f* pSrc = (const Npp32f*)srcPlane->GpuMem();
      int nSrcStep = (int)srcPlane->Pitch();
      NppiSize oSrcSize = {0};
      oSrcSize.width = srcPlane->Width();
      oSrcSize.height = srcPlane->Height();
      NppiRect oSrcRectROI = {0};
      oSrcRectROI.width = oSrcSize.width;
      oSrcRectROI.height = oSrcSize.height;

      Npp32f* pDst = (Npp32f*)dstPlane->GpuMem();
      int nDstStep = (int)dstPlane->Pitch();
      NppiSize oDstSize = {0};
      oDstSize.width = dstPlane->Width();
      oDstSize.height = dstPlane->Height();
      NppiRect oDstRectROI = {0};
      oDstRectROI.width = oDstSize.width;
      oDstRectROI.height = oDstSize.height;
      int eInterpolation = NPPI_INTER_LANCZOS;

      CudaCtxPush ctxPush(cu_ctx);
      auto ret = nppiResize_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                        pDst, nDstStep, oDstSize, oDstRectROI,
                                        eInterpolation, nppCtx);
      if (NPP_NO_ERROR != ret) {
        cerr << "NPP error with code " << ret << endl;
        return TASK_EXEC_FAIL;
      }
    }

    return TASK_EXEC_SUCCESS;
  }
};

}; // namespace VPF

auto const cuda_stream_sync = [](void* stream) {
  cuStreamSynchronize((CUstream)stream);
};

ResizeSurface::ResizeSurface(uint32_t width, uint32_t height,
                             Pixel_Format format, CUcontext ctx, CUstream str)
    : Task("NppResizeSurface", ResizeSurface::numInputs,
           ResizeSurface::numOutputs, cuda_stream_sync, (void*)str)
{
  if (RGB == format || BGR == format) {
    pImpl = new NppResizeSurfacePacked3C_Impl(width, height, ctx, str, format);
  } else if (YUV420 == format || YCBCR == format || YUV444 == format ||
             RGB_PLANAR == format) {
    pImpl = new NppResizeSurfacePlanar_Impl(width, height, ctx, str, format);
  } else if (RGB_32F == format) {
    pImpl =
        new NppResizeSurfacePacked32F3C_Impl(width, height, ctx, str, format);
  } else if (RGB_32F_PLANAR == format) {
    pImpl = new NppResizeSurface32FPlanar_Impl(width, height, ctx, str, format);
  } else if (NV12 == format) {
    pImpl = new ResizeSurfaceSemiPlanar_Impl(width, height, ctx, str, format);
  } else {
    stringstream ss;
    ss << __FUNCTION__;
    ss << ": pixel format not supported";
    throw runtime_error(ss.str());
  }
}

ResizeSurface::~ResizeSurface() { delete pImpl; }

TaskExecStatus ResizeSurface::Run()
{
  NvtxMark tick(GetName());
  ClearOutputs();

  auto pInputSurface = (Surface*)GetInput();
  if (!pInputSurface) {
    return TASK_EXEC_FAIL;
  }

  if (TASK_EXEC_SUCCESS != pImpl->Run(*pInputSurface)) {
    return TASK_EXEC_FAIL;
  }

  SetOutput(pImpl->pSurface, 0U);
  return TASK_EXEC_SUCCESS;
}

ResizeSurface* ResizeSurface::Make(uint32_t width, uint32_t height,
                                   Pixel_Format format, CUcontext ctx,
                                   CUstream str)
{
  return new ResizeSurface(width, height, format, ctx, str);
}

namespace VPF
{
struct RemapSurface_Impl {
  Surface* pSurface = nullptr;
  CudaBuffer* xMapPlane = nullptr;
  CudaBuffer* yMapPlane = nullptr;
  CUcontext cu_ctx;
  CUstream cu_str;
  NppStreamContext nppCtx;

  uint32_t map_w;
  uint32_t map_h;

  RemapSurface_Impl(const float* x_map, const float* y_map, uint32_t remap_w,
                    uint32_t remap_h, Pixel_Format format, CUcontext ctx,
                    CUstream str)
      : cu_ctx(ctx), cu_str(str), map_w(remap_w), map_h(remap_h)
  {
    xMapPlane = CudaBuffer::Make((const void*)x_map, sizeof(float),
                                 map_w * map_h, ctx, str);
    yMapPlane = CudaBuffer::Make((const void*)y_map, sizeof(float),
                                 map_w * map_h, ctx, str);

    SetupNppContext(cu_ctx, cu_str, nppCtx);
  }

  virtual ~RemapSurface_Impl()
  {
    if (xMapPlane) {
      delete xMapPlane;
    }
    if (yMapPlane) {
      delete yMapPlane;
    }
  }

  virtual TaskExecStatus Run(Surface& source) = 0;
};

struct NppRemapSurfacePacked3C_Impl final : RemapSurface_Impl {
  NppRemapSurfacePacked3C_Impl(const float* x_map, const float* y_map,
                               uint32_t remap_w, uint32_t remap_h,
                               CUcontext ctx, CUstream str, Pixel_Format format)
      : RemapSurface_Impl(x_map, y_map, remap_w, remap_h, format, ctx, str)
  {
    pSurface = Surface::Make(format, map_w, map_h, ctx);
  }

  ~NppRemapSurfacePacked3C_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface& source)
  {
    NvtxMark tick("NppRemapSurfacePacked3C");

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      cerr << "Input PixelFormat(" << source.PixelFormat()
           << ") != Remaper's PixelFormat (" << pSurface->PixelFormat() << ")."
           << endl;
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    auto srcPlane = source.GetSurfacePlane();
    auto dstPlane = pSurface->GetSurfacePlane();

    const Npp8u* pSrc = (const Npp8u*)srcPlane->GpuMem();
    int nSrcStep = (int)source.Pitch();
    NppiSize oSrcSize = {0};
    oSrcSize.width = source.Width();
    oSrcSize.height = source.Height();
    NppiRect oSrcRectROI = {0};
    oSrcRectROI.width = oSrcSize.width;
    oSrcRectROI.height = oSrcSize.height;

    Npp8u* pDst = (Npp8u*)dstPlane->GpuMem();
    int nDstStep = (int)pSurface->Pitch();
    NppiSize oDstSize = {0};
    oDstSize.width = pSurface->Width();
    oDstSize.height = pSurface->Height();

    auto pXMap = (const Npp32f*)xMapPlane->GpuMem();
    int nXMapStep = map_w * xMapPlane->GetElemSize();

    auto pYMap = (const Npp32f*)yMapPlane->GpuMem();
    int nYMapStep = map_w * yMapPlane->GetElemSize();

    int eInterpolation = NPPI_INTER_LINEAR;

    CudaCtxPush ctxPush(cu_ctx);
    auto ret = nppiRemap_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcRectROI,
                                    pXMap, nXMapStep, pYMap, nYMapStep, pDst,
                                    nDstStep, oDstSize, eInterpolation, nppCtx);
    if (NPP_NO_ERROR != ret) {
      cerr << "Can't remap 3-channel packed image. Error code: " << ret << endl;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};
} // namespace VPF

RemapSurface::RemapSurface(const float* x_map, const float* y_map,
                           uint32_t remap_w, uint32_t remap_h,
                           Pixel_Format format, CUcontext ctx, CUstream str)
    : Task("NppRemapSurface", RemapSurface::numInputs, RemapSurface::numOutputs,
           cuda_stream_sync, (void*)str)
{
  if (RGB == format || BGR == format) {
    pImpl = new NppRemapSurfacePacked3C_Impl(x_map, y_map, remap_w, remap_h,
                                             ctx, str, format);
  } else {
    stringstream ss;
    ss << __FUNCTION__;
    ss << ": pixel format not supported";
    throw runtime_error(ss.str());
  }
}

RemapSurface::~RemapSurface() { delete pImpl; }

TaskExecStatus RemapSurface::Run()
{
  NvtxMark tick(GetName());
  ClearOutputs();

  auto pInputSurface = (Surface*)GetInput();
  if (!pInputSurface) {
    return TASK_EXEC_FAIL;
  }

  if (TASK_EXEC_SUCCESS != pImpl->Run(*pInputSurface)) {
    return TASK_EXEC_FAIL;
  }

  SetOutput(pImpl->pSurface, 0U);
  return TASK_EXEC_SUCCESS;
}

RemapSurface* RemapSurface::Make(const float* x_map, const float* y_map,
                                 uint32_t remap_w, uint32_t remap_h,
                                 Pixel_Format format, CUcontext ctx,
                                 CUstream str)
{
  return new RemapSurface(x_map, y_map, remap_w, remap_h, format, ctx, str);
}
