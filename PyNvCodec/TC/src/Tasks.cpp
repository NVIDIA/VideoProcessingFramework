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

#include <chrono>
#include <fstream>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CodecsSupport.hpp"
#include "MemoryInterfaces.hpp"
#include "NppCommon.hpp"
#include "Tasks.hpp"

#include "NvCodecCLIOptions.h"
#include "NvCodecUtils.h"
#include "NvEncoderCuda.h"

#include "FFmpegDemuxer.h"
#include "NvDecoder.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

using namespace VPF;
using namespace std;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

namespace VPF {

struct NvencEncodeFrame_Impl {
  using packet = vector<uint8_t>;

  NV_ENC_BUFFER_FORMAT enc_buffer_format;
  queue<packet> packetQueue;
  vector<uint8_t> lastPacket;
  Buffer *pElementaryVideo;
  NvEncoderCuda *pEncoderCuda = nullptr;
  CUcontext context = nullptr;
  CUstream stream = 0;
  bool didEncode = false;
  bool didFlush = false;
  NV_ENC_RECONFIGURE_PARAMS recfg_params;
  NV_ENC_INITIALIZE_PARAMS &init_params;
  NV_ENC_CONFIG encodeConfig;

  NvencEncodeFrame_Impl() = delete;
  NvencEncodeFrame_Impl(const NvencEncodeFrame_Impl &other) = delete;
  NvencEncodeFrame_Impl &operator=(const NvencEncodeFrame_Impl &other) = delete;

  NvencEncodeFrame_Impl(NV_ENC_BUFFER_FORMAT format,
                        NvEncoderClInterface &cli_iface, CUcontext ctx,
                        CUstream str, int32_t width, int32_t height,
                        bool verbose)
      : init_params(recfg_params.reInitEncodeParams) {
    pElementaryVideo = Buffer::Make(0U);

    context = ctx;
    stream = str;
    pEncoderCuda = new NvEncoderCuda(context, width, height, format);
    enc_buffer_format = format;

    init_params = {NV_ENC_INITIALIZE_PARAMS_VER};
    encodeConfig = {NV_ENC_CONFIG_VER};
    init_params.encodeConfig = &encodeConfig;

    cli_iface.SetupInitParams(init_params, false, pEncoderCuda->GetApi(),
                              pEncoderCuda->GetEncoder(), verbose);

    pEncoderCuda->CreateEncoder(&init_params);
  }

  bool Reconfigure(NvEncoderClInterface &cli_iface, bool force_idr,
                   bool reset_enc, bool verbose) {
    recfg_params.version = NV_ENC_RECONFIGURE_PARAMS_VER;
    recfg_params.resetEncoder = reset_enc;
    recfg_params.forceIDR = force_idr;

    cli_iface.SetupInitParams(init_params, true, pEncoderCuda->GetApi(),
                              pEncoderCuda->GetEncoder(), verbose);

    return pEncoderCuda->Reconfigure(&recfg_params);
  }

  ~NvencEncodeFrame_Impl() {
    pEncoderCuda->DestroyEncoder();
    delete pEncoderCuda;
    delete pElementaryVideo;
  }
};
} // namespace VPF

NvencEncodeFrame *NvencEncodeFrame::Make(CUstream cuStream, CUcontext cuContext,
                                         NvEncoderClInterface &cli_iface,
                                         NV_ENC_BUFFER_FORMAT format,
                                         uint32_t width, uint32_t height,
                                         bool verbose) {
  return new NvencEncodeFrame(cuStream, cuContext, cli_iface, format, width,
                              height, verbose);
}

bool VPF::NvencEncodeFrame::Reconfigure(NvEncoderClInterface &cli_iface,
                                        bool force_idr, bool reset_enc,
                                        bool verbose) {
  return pImpl->Reconfigure(cli_iface, force_idr, reset_enc, verbose);
}

NvencEncodeFrame::NvencEncodeFrame(CUstream cuStream, CUcontext cuContext,
                                   NvEncoderClInterface &cli_iface,
                                   NV_ENC_BUFFER_FORMAT format, uint32_t width,
                                   uint32_t height, bool verbose)
    :

      Task("NvencEncodeFrame", NvencEncodeFrame::numInputs,
           NvencEncodeFrame::numOutputs) {
  pImpl = new NvencEncodeFrame_Impl(format, cli_iface, cuContext, cuStream,
                                    width, height, verbose);
}

NvencEncodeFrame::~NvencEncodeFrame() { delete pImpl; };

TaskExecStatus NvencEncodeFrame::Run() {
  NvtxMark tick(__FUNCTION__);
  SetOutput(nullptr, 0U);

  try {
    auto &pEncoderCuda = pImpl->pEncoderCuda;
    auto &didFlush = pImpl->didFlush;
    auto &didEncode = pImpl->didEncode;
    auto &context = pImpl->context;
    auto input = (Surface *)GetInput(0U);
    vector<vector<uint8_t>> encPackets;

    if (input) {
      auto &stream = pImpl->stream;
      const NvEncInputFrame *encoderInputFrame =
          pEncoderCuda->GetNextInputFrame();
      auto width = input->Width(), height = input->Height(),
           pitch = input->Pitch();

      bool is_resize_needed = (pEncoderCuda->GetEncodeWidth() != width) ||
                              (pEncoderCuda->GetEncodeHeight() != height);

      if (is_resize_needed) {
        return TASK_EXEC_FAIL;
      } else {
        NvEncoderCuda::CopyToDeviceFrame(
            context, stream, (void *)input->PlanePtr(), pitch,
            (CUdeviceptr)encoderInputFrame->inputPtr,
            (int32_t)encoderInputFrame->pitch, pEncoderCuda->GetEncodeWidth(),
            pEncoderCuda->GetEncodeHeight(), CU_MEMORYTYPE_DEVICE,
            encoderInputFrame->bufferFormat, encoderInputFrame->chromaOffsets,
            encoderInputFrame->numChromaPlanes);
      }
      cudaStreamSynchronize(stream);

      auto pSEI = (Buffer *)GetInput(2U);
      NV_ENC_SEI_PAYLOAD payload = {0};
      if (pSEI) {
        payload.payloadSize = pSEI->GetRawMemSize();
        // Unregistered user data for H.265 and H.264 both;
        payload.payloadType = 5;
        payload.payload = pSEI->GetDataAs<uint8_t>();
      }

      auto const seiNumber = pSEI ? 1U : 0U;
      auto pPayload = pSEI ? &payload : nullptr;

      auto sync = GetInput(1U);
      if (sync) {
        pEncoderCuda->EncodeFrame(encPackets, nullptr, false, seiNumber,
                                  pPayload);
      } else {
        pEncoderCuda->EncodeFrame(encPackets, nullptr, true, seiNumber,
                                  pPayload);
      }
      didEncode = true;
    } else if (didEncode && !didFlush) {
      // No input after a while means we're flushing;
      pEncoderCuda->EndEncode(encPackets);
      didFlush = true;
    }

    /* Push encoded packets into queue;
     */
    for (auto &packet : encPackets) {
      pImpl->packetQueue.push(packet);
    }

    /* Then return least recent packet;
     */
    pImpl->lastPacket.clear();
    if (!pImpl->packetQueue.empty()) {
      pImpl->lastPacket = pImpl->packetQueue.front();
      pImpl->pElementaryVideo->Update(pImpl->lastPacket.size(),
                                      (void *)pImpl->lastPacket.data());
      pImpl->packetQueue.pop();
      SetOutput(pImpl->pElementaryVideo, 0U);
    }

    return TASK_EXEC_SUCCESS;
  } catch (exception &e) {
    cerr << e.what() << endl;
    return TASK_EXEC_FAIL;
  }
}

namespace VPF {
struct NvdecDecodeFrame_Impl {
  NvDecoder nvDecoder;
  Surface *pLastSurface = nullptr;
  Buffer *pPacketData = nullptr;
  CUstream stream = 0;
  CUcontext context = nullptr;
  bool didDecode = false;

  NvdecDecodeFrame_Impl() = delete;
  NvdecDecodeFrame_Impl(const NvdecDecodeFrame_Impl &other) = delete;
  NvdecDecodeFrame_Impl &operator=(const NvdecDecodeFrame_Impl &other) = delete;

  NvdecDecodeFrame_Impl(CUstream cuStream, CUcontext cuContext,
                        cudaVideoCodec videoCodec, Pixel_Format format)
      : stream(cuStream), context(cuContext),
        nvDecoder(cuStream, cuContext, videoCodec) {
    pLastSurface = Surface::Make(format);
    pPacketData = Buffer::MakeOwnMem(sizeof(PacketData));
  }

  ~NvdecDecodeFrame_Impl() {
    delete pLastSurface;
    delete pPacketData;
  }
};
} // namespace VPF

NvdecDecodeFrame *NvdecDecodeFrame::Make(CUstream cuStream, CUcontext cuContext,
                                         cudaVideoCodec videoCodec,
                                         uint32_t decodedFramesPoolSize,
                                         uint32_t coded_width,
                                         uint32_t coded_height,
                                         Pixel_Format format) {
  return new NvdecDecodeFrame(cuStream, cuContext, videoCodec,
                              decodedFramesPoolSize, coded_width, coded_height,
                              format);
}

NvdecDecodeFrame::NvdecDecodeFrame(CUstream cuStream, CUcontext cuContext,
                                   cudaVideoCodec videoCodec,
                                   uint32_t decodedFramesPoolSize,
                                   uint32_t coded_width, uint32_t coded_height,
                                   Pixel_Format format)
    :

      Task("NvdecDecodeFrame", NvdecDecodeFrame::numInputs,
           NvdecDecodeFrame::numOutputs) {
  pImpl = new NvdecDecodeFrame_Impl(cuStream, cuContext, videoCodec, format);
}

NvdecDecodeFrame::~NvdecDecodeFrame() {
  auto lastSurface = pImpl->pLastSurface->PlanePtr();
  pImpl->nvDecoder.UnlockSurface(lastSurface);
  delete pImpl;
}

TaskExecStatus NvdecDecodeFrame::Run() {
  NvtxMark tick(__FUNCTION__);
  ClearOutputs();

  auto &decoder = pImpl->nvDecoder;
  auto pEncFrame = (Buffer *)GetInput();

  if (!pEncFrame && !pImpl->didDecode) {
    /* Empty input given + we've never did decoding means something went wrong;
     * Otherwise (no input + we did decode) means we're flushing;
     */
    return TASK_EXEC_FAIL;
  }

  bool isSurfaceReturned = false;
  uint64_t timestamp = 0U;
  auto pPktData = (Buffer *)GetInput(1U);
  if (pPktData) {
    auto p_pkt_data = pPktData->GetDataAs<PacketData>();
    timestamp = p_pkt_data->pts;
    pImpl->pPacketData->Update(sizeof(*p_pkt_data), p_pkt_data);
  }

  /* This will feed decoder with input timestamp.
   * It will also return surface + it's timestamp.
   * So timestamp is input + output parameter. */
  DecodedFrameContext dec_ctx;
  try {
    isSurfaceReturned =
        decoder.DecodeLockSurface(pEncFrame, timestamp, dec_ctx);
    pImpl->didDecode = true;
  } catch (exception &e) {
    cerr << e.what() << endl;
    return TASK_EXEC_FAIL;
  }

  if (isSurfaceReturned) {
    // Unlock last surface because we will use it later;
    auto lastSurface = pImpl->pLastSurface->PlanePtr();
    decoder.UnlockSurface(lastSurface);

    // Update the reconstructed frame data;
    auto rawW = decoder.GetWidth();
    auto rawH = decoder.GetHeight() + decoder.GetChromaHeight();
    auto rawP = decoder.GetDeviceFramePitch();

    SurfacePlane tmpPlane(rawW, rawH, rawP, sizeof(uint8_t), dec_ctx.mem);
    pImpl->pLastSurface->Update(&tmpPlane, 1);
    SetOutput(pImpl->pLastSurface, 0U);

    // Update the reconstructed frame timestamp;
    auto p_packet_data = pImpl->pPacketData->GetDataAs<PacketData>();
    memset(p_packet_data, 0, sizeof(*p_packet_data));
    p_packet_data->pts = dec_ctx.pts;
    p_packet_data->poc = dec_ctx.poc;
    SetOutput(pImpl->pPacketData, 1U);

    return TASK_EXEC_SUCCESS;
  }

  /* If we have input and don't get output so far that's fine.
   * Otherwise input is NULL and we're flusing so we shall get frame.
   */
  return pEncFrame ? TASK_EXEC_SUCCESS : TASK_EXEC_FAIL;
}

void NvdecDecodeFrame::GetDecodedFrameParams(uint32_t &width, uint32_t &height,
                                             uint32_t &elem_size) {
  width = pImpl->nvDecoder.GetWidth();
  height = pImpl->nvDecoder.GetHeight();
  elem_size = (pImpl->nvDecoder.GetBitDepth() + 7) / 8;
}

uint32_t NvdecDecodeFrame::GetDeviceFramePitch() {
  return uint32_t(pImpl->nvDecoder.GetDeviceFramePitch());
}

namespace VPF {
static size_t GetElemSize(Pixel_Format format) {
  stringstream ss;

  switch (format) {
  case RGB_PLANAR:
  case YUV444:
  case YUV420:
  case YCBCR:
  case NV12:
  case RGB:
  case BGR:
  case Y:
    return sizeof(uint8_t);
  default:
    ss << __FUNCTION__;
    ss << ": unsupported pixel format";
    throw invalid_argument(ss.str());
  }
}

auto const cuda_stream_sync = [](void *stream) {
  cuStreamSynchronize((CUstream)stream);
};

struct CudaUploadFrame_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  Surface *pSurface = nullptr;
  Pixel_Format pixelFormat;

  CudaUploadFrame_Impl() = delete;
  CudaUploadFrame_Impl(const CudaUploadFrame_Impl &other) = delete;
  CudaUploadFrame_Impl &operator=(const CudaUploadFrame_Impl &other) = delete;

  CudaUploadFrame_Impl(CUstream stream, CUcontext context, uint32_t _width,
                       uint32_t _height, Pixel_Format _pix_fmt)
      : cuStream(stream), cuContext(context), pixelFormat(_pix_fmt) {
    pSurface = Surface::Make(pixelFormat, _width, _height, context);
  }

  ~CudaUploadFrame_Impl() { delete pSurface; }
};
} // namespace VPF

CudaUploadFrame *CudaUploadFrame::Make(CUstream cuStream, CUcontext cuContext,
                                       uint32_t width, uint32_t height,
                                       Pixel_Format pixelFormat) {
  return new CudaUploadFrame(cuStream, cuContext, width, height, pixelFormat);
}

CudaUploadFrame::CudaUploadFrame(CUstream cuStream, CUcontext cuContext,
                                 uint32_t width, uint32_t height,
                                 Pixel_Format pix_fmt)
    :

      Task("CudaUploadFrame", CudaUploadFrame::numInputs,
           CudaUploadFrame::numOutputs, cuda_stream_sync, (void *)cuStream) {
  pImpl = new CudaUploadFrame_Impl(cuStream, cuContext, width, height, pix_fmt);
}

CudaUploadFrame::~CudaUploadFrame() { delete pImpl; }

TaskExecStatus CudaUploadFrame::Run() {
  NvtxMark tick(__FUNCTION__);
  if (!GetInput()) {
    return TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pSurface = pImpl->pSurface;
  auto pSrcHost = ((Buffer *)GetInput())->GetDataAs<uint8_t>();

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

namespace VPF {
struct CudaDownloadSurface_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  Pixel_Format format;
  Buffer *pHostFrame = nullptr;

  CudaDownloadSurface_Impl() = delete;
  CudaDownloadSurface_Impl(const CudaDownloadSurface_Impl &other) = delete;
  CudaDownloadSurface_Impl &
  operator=(const CudaDownloadSurface_Impl &other) = delete;

  CudaDownloadSurface_Impl(CUstream stream, CUcontext context, uint32_t _width,
                           uint32_t _height, Pixel_Format _pix_fmt)
      : cuStream(stream), cuContext(context), format(_pix_fmt) {

    auto bufferSize = _width * _height * GetElemSize(_pix_fmt);

    if (YUV420 == _pix_fmt || NV12 == _pix_fmt || YCBCR == _pix_fmt) {
      bufferSize = bufferSize * 3U / 2U;
    } else if (RGB == _pix_fmt || RGB_PLANAR == _pix_fmt || BGR == _pix_fmt ||
               YUV444 == _pix_fmt) {
      bufferSize = bufferSize * 3U;
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
} // namespace VPF

CudaDownloadSurface *CudaDownloadSurface::Make(CUstream cuStream,
                                               CUcontext cuContext,
                                               uint32_t width, uint32_t height,
                                               Pixel_Format pixelFormat) {
  return new CudaDownloadSurface(cuStream, cuContext, width, height,
                                 pixelFormat);
}

CudaDownloadSurface::CudaDownloadSurface(CUstream cuStream, CUcontext cuContext,
                                         uint32_t width, uint32_t height,
                                         Pixel_Format pix_fmt)
    :

      Task("CudaDownloadSurface", CudaDownloadSurface::numInputs,
           CudaDownloadSurface::numOutputs, cuda_stream_sync,
           (void *)cuStream) {
  pImpl =
      new CudaDownloadSurface_Impl(cuStream, cuContext, width, height, pix_fmt);
}

CudaDownloadSurface::~CudaDownloadSurface() { delete pImpl; }

TaskExecStatus CudaDownloadSurface::Run() {
  NvtxMark tick(__FUNCTION__);

  if (!GetInput()) {
    return TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pSurface = (Surface *)GetInput();
  auto pDstHost = ((Buffer *)pImpl->pHostFrame)->GetDataAs<uint8_t>();

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

    if (CUDA_SUCCESS != cuMemcpy2DAsync(&m, stream)) {
      return TASK_EXEC_FAIL;
    }

    pDstHost += m.WidthInBytes * m.Height;
  }

  SetOutput(pImpl->pHostFrame, 0);
  return TASK_EXEC_SUCCESS;
}

namespace VPF {
struct DemuxFrame_Impl {
  size_t videoBytes = 0U;
  FFmpegDemuxer demuxer;
  Buffer *pElementaryVideo;
  Buffer *pMuxingParams;
  Buffer *pSei;
  Buffer *pPktData;

  DemuxFrame_Impl() = delete;
  DemuxFrame_Impl(const DemuxFrame_Impl &other) = delete;
  DemuxFrame_Impl &operator=(const DemuxFrame_Impl &other) = delete;

  explicit DemuxFrame_Impl(const string &url,
                           const map<string, string> &ffmpeg_options)
      : demuxer(url.c_str(), ffmpeg_options) {
    pElementaryVideo = Buffer::MakeOwnMem(0U);
    pMuxingParams = Buffer::MakeOwnMem(sizeof(MuxingParams));
    pSei = Buffer::MakeOwnMem(0U);
    pPktData = Buffer::MakeOwnMem(0U);
  }

  ~DemuxFrame_Impl() {
    delete pElementaryVideo;
    delete pMuxingParams;
    delete pSei;
    delete pPktData;
  }
};
} // namespace VPF

DemuxFrame *DemuxFrame::Make(const char *url, const char **ffmpeg_options,
                             uint32_t opts_size) {
  return new DemuxFrame(url, ffmpeg_options, opts_size);
}

DemuxFrame::DemuxFrame(const char *url, const char **ffmpeg_options,
                       uint32_t opts_size)
    : Task("DemuxFrame", DemuxFrame::numInputs, DemuxFrame::numOutputs) {
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

void DemuxFrame::Flush() { pImpl->demuxer.Flush(); }

TaskExecStatus DemuxFrame::Run() {
  NvtxMark tick(__FUNCTION__);
  ClearOutputs();

  uint8_t *pVideo = nullptr;
  MuxingParams params = {0};
  PacketData pkt_data = {0};

  auto &videoBytes = pImpl->videoBytes;
  auto &demuxer = pImpl->demuxer;

  uint8_t *pSEI = nullptr;
  size_t seiBytes = 0U;
  bool needSEI = (nullptr != GetInput(0U));

  auto pSeekCtxBuf = (Buffer *)GetInput(1U);
  if (pSeekCtxBuf) {
    SeekContext seek_ctx = *pSeekCtxBuf->GetDataAs<SeekContext>();
    auto ret = demuxer.Seek(seek_ctx, pVideo, videoBytes, pkt_data,
                            needSEI ? &pSEI : nullptr, &seiBytes);
    if (!ret) {
      return TASK_EXEC_FAIL;
    }
  } else if (!demuxer.Demux(pVideo, videoBytes, pkt_data,
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

void DemuxFrame::GetParams(MuxingParams &params) const {
  params.videoContext.width = pImpl->demuxer.GetWidth();
  params.videoContext.height = pImpl->demuxer.GetHeight();
  params.videoContext.num_frames = pImpl->demuxer.GetNumFrames();
  params.videoContext.frameRate = pImpl->demuxer.GetFramerate();
  params.videoContext.timeBase = pImpl->demuxer.GetTimebase();
  params.videoContext.streamIndex = pImpl->demuxer.GetVideoStreamIndex();
  params.videoContext.codec = FFmpeg2NvCodecId(pImpl->demuxer.GetVideoCodec());
  params.videoContext.gop_size = pImpl->demuxer.GetGopSize();

  switch (pImpl->demuxer.GetPixelFormat()) {
  case AV_PIX_FMT_YUVJ420P:
  case AV_PIX_FMT_YUV420P:
  case AV_PIX_FMT_NV12:
    params.videoContext.format = NV12;
    break;
  case AV_PIX_FMT_YUV444P:
    params.videoContext.format = YUV444;
    break;
  default:
    stringstream ss;
    ss << "Unsupported FFmpeg pixel format: "
       << av_get_pix_fmt_name(pImpl->demuxer.GetPixelFormat()) << endl;
    throw invalid_argument(ss.str());
    params.videoContext.format = UNDEFINED;
    break;
  }
}

namespace VPF {
struct ResizeSurface_Impl {
  Surface *pSurface = nullptr;
  CUcontext cu_ctx;
  CUstream cu_str;
  NppStreamContext nppCtx;

  ResizeSurface_Impl(uint32_t width, uint32_t height, Pixel_Format format,
                     CUcontext ctx, CUstream str)
      : cu_ctx(ctx), cu_str(str) {
    SetupNppContext(cu_ctx, cu_str, nppCtx);
  }

  virtual ~ResizeSurface_Impl() = default;

  virtual TaskExecStatus Run(Surface &source) = 0;
};

struct NppResizeSurfacePacked3C_Impl final : ResizeSurface_Impl {
  NppResizeSurfacePacked3C_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                                CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str) {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurfacePacked3C_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface &source) {
    NvtxMark tick(__FUNCTION__);

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    auto srcPlane = source.GetSurfacePlane();
    auto dstPlane = pSurface->GetSurfacePlane();

    const Npp8u *pSrc = (const Npp8u *)srcPlane->GpuMem();
    int nSrcStep = (int)source.Pitch();
    NppiSize oSrcSize = {0};
    oSrcSize.width = source.Width();
    oSrcSize.height = source.Height();
    NppiRect oSrcRectROI = {0};
    oSrcRectROI.width = oSrcSize.width;
    oSrcRectROI.height = oSrcSize.height;

    Npp8u *pDst = (Npp8u *)dstPlane->GpuMem();
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

  NppStreamContext nppCtx;
  CUcontext ctx;
};

// Resize planar 8 bit surface (YUV420, YCbCr420);
struct NppResizeSurfacePlanar420_Impl final : ResizeSurface_Impl {
  NppResizeSurfacePlanar420_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                                 CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str) {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurfacePlanar420_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface &source) {
    NvtxMark tick(__FUNCTION__);

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      cerr << "Actual pixel format is " << source.PixelFormat() << endl;
      cerr << "Expected input format is " << pSurface->PixelFormat() << endl;
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    for (auto plane = 0; plane < pSurface->NumPlanes(); plane++) {
      auto srcPlane = source.GetSurfacePlane(plane);
      auto dstPlane = pSurface->GetSurfacePlane(plane);

      const Npp8u *pSrc = (const Npp8u *)srcPlane->GpuMem();
      int nSrcStep = (int)srcPlane->Pitch();
      NppiSize oSrcSize = {0};
      oSrcSize.width = srcPlane->Width();
      oSrcSize.height = srcPlane->Height();
      NppiRect oSrcRectROI = {0};
      oSrcRectROI.width = oSrcSize.width;
      oSrcRectROI.height = oSrcSize.height;

      Npp8u *pDst = (Npp8u *)dstPlane->GpuMem();
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

}; // namespace VPF

ResizeSurface::ResizeSurface(uint32_t width, uint32_t height,
                             Pixel_Format format, CUcontext ctx, CUstream str)
    : Task("NppResizeSurface", ResizeSurface::numInputs,
           ResizeSurface::numOutputs) {
  if (RGB == format || BGR == format) {
    pImpl = new NppResizeSurfacePacked3C_Impl(width, height, ctx, str, format);
  } else if (YUV420 == format || YCBCR == format) {
    pImpl = new NppResizeSurfacePlanar420_Impl(width, height, ctx, str, format);
  } else {
    stringstream ss;
    ss << __FUNCTION__;
    ss << ": pixel format not supported";
    throw runtime_error(ss.str());
  }
}

ResizeSurface::~ResizeSurface() { delete pImpl; }

TaskExecStatus ResizeSurface::Run() {
  NvtxMark tick(__FUNCTION__);
  ClearOutputs();

  auto pInputSurface = (Surface *)GetInput();
  if (!pInputSurface) {
    return TASK_EXEC_FAIL;
  }

  if (TASK_EXEC_SUCCESS != pImpl->Run(*pInputSurface)) {
    return TASK_EXEC_FAIL;
  }

  SetOutput(pImpl->pSurface, 0U);
  return TASK_EXEC_SUCCESS;
}

ResizeSurface *ResizeSurface::Make(uint32_t width, uint32_t height,
                                   Pixel_Format format, CUcontext ctx,
                                   CUstream str) {
  return new ResizeSurface(width, height, format, ctx, str);
}
