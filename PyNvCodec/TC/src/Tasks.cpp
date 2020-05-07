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

#include "NvCodecUtils.h"
#include "NvEncoderCLIOptions.h"
#include "NvEncoderCuda.h"

#include "FFmpegDemuxer.h"
#include "NvDecoder.h"

using namespace VPF;
using namespace std;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

namespace VPF {

struct NvencEncodeFrame_Impl {
  using packet = vector<uint8_t>;

  queue<packet> packetQueue;
  vector<uint8_t> lastPacket;
  Buffer *pElementaryVideo;
  NvEncoderCuda *pEncoderCuda = nullptr;
  CUcontext context = nullptr;
  CUstream stream = 0;
  bool didEncode = false;
  bool didFlush = false;

  NvencEncodeFrame_Impl() = delete;
  NvencEncodeFrame_Impl(const NvencEncodeFrame_Impl &other) = delete;
  NvencEncodeFrame_Impl &operator=(const NvencEncodeFrame_Impl &other) = delete;

  NvencEncodeFrame_Impl(NV_ENC_BUFFER_FORMAT format,
                        NvEncoderInitParam &initParam, CUcontext ctx,
                        CUstream str, int32_t width, int32_t height) {
    pElementaryVideo = Buffer::Make(0U);

    context = ctx;
    stream = str;
    pEncoderCuda = new NvEncoderCuda(context, width, height, format);

    NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
    initializeParams.encodeConfig = &encodeConfig;

    pEncoderCuda->CreateDefaultEncoderParams(&initializeParams,
                                             initParam.GetEncodeGUID(),
                                             initParam.GetPresetGUID());

    initParam.SetInitParams(&initializeParams, format);
    pEncoderCuda->CreateEncoder(&initializeParams);
  }

  ~NvencEncodeFrame_Impl() {
    pEncoderCuda->DestroyEncoder();
    delete pEncoderCuda;
    delete pElementaryVideo;
  }
};
} // namespace VPF

NvencEncodeFrame *NvencEncodeFrame::Make(CUstream cuStream, CUcontext cuContext,
                                         NvEncoderInitParam &initParams,
                                         NV_ENC_BUFFER_FORMAT format,
                                         uint32_t width, uint32_t height) {
  return new NvencEncodeFrame(cuStream, cuContext, initParams, format, width,
                              height);
}

NvencEncodeFrame::NvencEncodeFrame(CUstream cuStream, CUcontext cuContext,
                                   NvEncoderInitParam &initParams,
                                   NV_ENC_BUFFER_FORMAT format, uint32_t width,
                                   uint32_t height)
    :

      Task("NvencEncodeFrame", NvencEncodeFrame::numInputs,
           NvencEncodeFrame::numOutputs) {
  pImpl = new NvencEncodeFrame_Impl(format, initParams, cuContext, cuStream,
                                    width, height);
}

NvencEncodeFrame::~NvencEncodeFrame() { delete pImpl; };

TaskExecStatus NvencEncodeFrame::Execute() {
  SetOutput(nullptr, 0U);

  try {
    auto &pEncoderCuda = pImpl->pEncoderCuda;
    auto &didFlush = pImpl->didFlush;
    auto &didEncode = pImpl->didEncode;
    auto &context = pImpl->context;
    auto input = (Surface *)GetInput();
    vector<vector<uint8_t>> encPackets;

    if (input && NV12 == input->PixelFormat()) {
      auto &stream = pImpl->stream;
      const NvEncInputFrame *encoderInputFrame =
          pEncoderCuda->GetNextInputFrame();
      auto width = input->Width(), height = input->Height(),
           pitch = input->Pitch();

      bool is_resize_needed = (pEncoderCuda->GetEncodeWidth() != width) ||
                              (pEncoderCuda->GetEncodeHeight() != height);

      if (is_resize_needed) {
        return TASK_EXEC_FAIL;
        /*cuCtxPushCurrent((CUcontext)pEncoderCuda->GetDevice());
        ResizeNv12(
            (unsigned char *)encoderInputFrame->inputPtr,
            (int32_t)encoderInputFrame->pitch, pEncoderCuda->GetEncodeWidth(),
            pEncoderCuda->GetEncodeHeight(), (unsigned char *)input->PlanePtr(),
            pitch, width, height, nullptr, stream);
        cuCtxPopCurrent(nullptr);*/
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

      pEncoderCuda->EncodeFrame(encPackets);
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
  } catch (...) {
    return TASK_EXEC_FAIL;
  }
}

namespace VPF {
struct NvdecDecodeFrame_Impl {
  NvDecoder nvDecoder;
  SurfaceNV12 *pLastSurface = nullptr;
  CUstream stream = 0;
  CUcontext context = nullptr;
  bool didDecode = false;

  NvdecDecodeFrame_Impl() = delete;
  NvdecDecodeFrame_Impl(const NvdecDecodeFrame_Impl &other) = delete;
  NvdecDecodeFrame_Impl &operator=(const NvdecDecodeFrame_Impl &other) = delete;

  NvdecDecodeFrame_Impl(CUstream cuStream, CUcontext cuContext,
                        cudaVideoCodec videoCodec)
      : stream(cuStream), context(cuContext),
        nvDecoder(cuStream, cuContext, videoCodec) {
    pLastSurface = new SurfaceNV12();
  }

  ~NvdecDecodeFrame_Impl() { delete pLastSurface; }
};
} // namespace VPF

NvdecDecodeFrame *NvdecDecodeFrame::Make(CUstream cuStream, CUcontext cuContext,
                                         cudaVideoCodec videoCodec,
                                         uint32_t decodedFramesPoolSize,
                                         uint32_t coded_width,
                                         uint32_t coded_height) {
  return new NvdecDecodeFrame(cuStream, cuContext, videoCodec,
                              decodedFramesPoolSize, coded_width, coded_height);
}

NvdecDecodeFrame::NvdecDecodeFrame(CUstream cuStream, CUcontext cuContext,
                                   cudaVideoCodec videoCodec,
                                   uint32_t decodedFramesPoolSize,
                                   uint32_t coded_width, uint32_t coded_height)
    :

      Task("NvdecDecodeFrame", NvdecDecodeFrame::numInputs,
           NvdecDecodeFrame::numOutputs) {
  pImpl = new NvdecDecodeFrame_Impl(cuStream, cuContext, videoCodec);
}

NvdecDecodeFrame::~NvdecDecodeFrame() {
  auto lastSurface = pImpl->pLastSurface->PlanePtr();
  pImpl->nvDecoder.UnlockSurface(lastSurface);
  delete pImpl;
}

TaskExecStatus NvdecDecodeFrame::Execute() {
  ClearOutputs();

  auto &decoder = pImpl->nvDecoder;
  auto pElementaryVideoStream = (Buffer *)GetInput();

  uint8_t *pVideo = nullptr;
  size_t nVideoBytes = 0U;

  if (pElementaryVideoStream) {
    pVideo = (uint8_t *)pElementaryVideoStream->GetRawMemPtr();
    nVideoBytes = pElementaryVideoStream->GetRawMemSize();
  } else if (!pImpl->didDecode) {
    /* Empty input given + we've never did decoding means something went wrong;
     * Otherwise (no input + we did decode) means we're flushing;
     */
    return TASK_EXEC_FAIL;
  }

  CUdeviceptr surface = 0U;
  bool isSurfaceReturned = false;
  auto res = decoder.DecodeLockSurface(pVideo, nVideoBytes, surface,
                                       isSurfaceReturned);
  pImpl->didDecode = true;
  if (!res) {
    return TASK_EXEC_FAIL;
  }

  if (isSurfaceReturned) {
    auto lastSurface = pImpl->pLastSurface->PlanePtr();
    decoder.UnlockSurface(lastSurface);

    auto rawW = decoder.GetWidth();
    auto rawH = decoder.GetHeight() * 3 / 2;
    auto rawP = decoder.GetDeviceFramePitch();

    SurfacePlane tmpPlane(rawW, rawH, rawP, sizeof(uint8_t), surface);
    pImpl->pLastSurface->Update(tmpPlane);
    SetOutput(pImpl->pLastSurface, 0U);
    return TASK_EXEC_SUCCESS;
  }

  return (nVideoBytes == 0) ? TASK_EXEC_FAIL : TASK_EXEC_SUCCESS;
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
  case YUV420:
  case NV12:
  case RGB:
  case Y:
    return sizeof(uint8_t);
  default:
    ss << __FUNCTION__;
    ss << ": unsupported pixel format";
    throw invalid_argument(ss.str());
  }
}

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
           CudaUploadFrame::numOutputs) {
  pImpl = new CudaUploadFrame_Impl(cuStream, cuContext, width, height, pix_fmt);
}

CudaUploadFrame::~CudaUploadFrame() { delete pImpl; }

TaskExecStatus CudaUploadFrame::Execute() {
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

  if (CUDA_SUCCESS != cuStreamSynchronize(stream)) {
    return TASK_EXEC_FAIL;
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

    if (YUV420 == _pix_fmt || NV12 == _pix_fmt) {
      bufferSize = bufferSize * 3U / 2U;
    } else if (RGB == _pix_fmt || RGB_PLANAR == _pix_fmt) {
      bufferSize = bufferSize * 3U;
    } else if (Y == _pix_fmt) {
    } else {
      stringstream ss;
      ss << __FUNCTION__ << ": unsupported pixel format: " << _pix_fmt << endl;
      throw invalid_argument(ss.str());
    }

    pHostFrame = Buffer::MakeOwnMem(bufferSize);
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
           CudaDownloadSurface::numOutputs) {
  pImpl =
      new CudaDownloadSurface_Impl(cuStream, cuContext, width, height, pix_fmt);
}

CudaDownloadSurface::~CudaDownloadSurface() { delete pImpl; }

TaskExecStatus CudaDownloadSurface::Execute() {

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

  if (CUDA_SUCCESS != cuStreamSynchronize(stream)) {
    return TASK_EXEC_FAIL;
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

  DemuxFrame_Impl() = delete;
  DemuxFrame_Impl(const DemuxFrame_Impl &other) = delete;
  DemuxFrame_Impl &operator=(const DemuxFrame_Impl &other) = delete;

  explicit DemuxFrame_Impl(const string &url,
                           const map<string, string> &ffmpeg_options)
      : demuxer(url.c_str(), ffmpeg_options) {
    pElementaryVideo = Buffer::MakeOwnMem(0U);
    pMuxingParams = Buffer::MakeOwnMem(sizeof(MuxingParams));
  }

  ~DemuxFrame_Impl() {
    delete pElementaryVideo;
    delete pMuxingParams;
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

TaskExecStatus DemuxFrame::Execute() {
  ClearOutputs();

  uint8_t *pVideo = nullptr;
  MuxingParams params = {0};

  auto &videoBytes = pImpl->videoBytes;
  auto &demuxer = pImpl->demuxer;

  if (!demuxer.Demux(pVideo, videoBytes)) {
    return TASK_EXEC_FAIL;
  }

  if (videoBytes) {
    pImpl->pElementaryVideo->Update(videoBytes, pVideo);
    pImpl->demuxer.GetLastPacketData(params.videoContext.packetData);
    SetOutput(pImpl->pElementaryVideo, 0U);

    GetParams(params);
    pImpl->pMuxingParams->Update(sizeof(MuxingParams), &params);
    SetOutput(pImpl->pMuxingParams, 1U);
  }

  return TASK_EXEC_SUCCESS;
}

void DemuxFrame::GetParams(MuxingParams &params) const {
  params.videoContext.width = pImpl->demuxer.GetWidth();
  params.videoContext.height = pImpl->demuxer.GetHeight();
  params.videoContext.frameRate = pImpl->demuxer.GetFramerate();
  params.videoContext.streamIndex = pImpl->demuxer.GetVideoStreamIndex();
  params.videoContext.codec = FFmpeg2NvCodecId(pImpl->demuxer.GetVideoCodec());
  params.videoContext.format = NV12;
}

namespace VPF {
struct MuxFrame_Impl {
  AVFormatContext *outFmtCtx = nullptr;
  AVStream *videoStream = nullptr;
  map<uint32_t, uint32_t> streamMapping;

  MuxFrame_Impl() = delete;
  MuxFrame_Impl(const MuxFrame_Impl &other) = delete;
  MuxFrame_Impl &operator=(const MuxFrame_Impl &other) = delete;

  bool hasVideo = false;

private:
  void SetupVideoStream(MuxingParams &params) {
    auto &videoCtx = params.videoContext;

    videoStream = avformat_new_stream(outFmtCtx, nullptr);
    if (!videoStream) {
      stringstream ss;
      ss << __FUNCTION__;
      ss << ": can't open video stream. ";
      throw runtime_error(ss.str());
    }

    videoStream->index = videoCtx.streamIndex;
    videoStream->time_base = av_make_q(1, videoCtx.frameRate);

    AVCodecParameters *videoCodecParams = videoStream->codecpar;
    videoCodecParams->codec_type = AVMEDIA_TYPE_VIDEO;
    stringstream ss;

    switch (videoCtx.codec) {
    case cudaVideoCodec_H264:
      videoCodecParams->codec_id = AV_CODEC_ID_H264;
      break;
    case cudaVideoCodec_HEVC:
      videoCodecParams->codec_id = AV_CODEC_ID_H265;
      break;
    default:
      ss << __FUNCTION__;
      ss << ": unsupported video codec";
      throw runtime_error(ss.str());
      break;
    }
    videoCodecParams->codec_tag = 0;
    videoCodecParams->width = videoCtx.width;
    videoCodecParams->height = videoCtx.height;
  }

public:
  MuxFrame_Impl(MuxingParams &params, const char *url) {
    auto ret =
        avformat_alloc_output_context2(&outFmtCtx, nullptr, nullptr, url);
    if (ret < 0) {
      stringstream ss;
      ss << __FUNCTION__ << ": can't alloc output context. Error code " << ret
         << endl;
      throw runtime_error(ss.str());
    }

    SetupVideoStream(params);
    streamMapping[videoStream->index] = outFmtCtx->nb_streams - 1;
    cout << "Video steam mapping: " << videoStream->index << "->"
         << streamMapping[videoStream->index] << endl;

    ret = avio_open(&outFmtCtx->pb, url, AVIO_FLAG_WRITE);
    if (ret < 0) {
      stringstream ss;
      ss << __FUNCTION__ << ": can't open output URL. Error code " << ret
         << endl;
      throw runtime_error(ss.str());
    }

    ret = avformat_write_header(outFmtCtx, NULL);
    if (ret < 0) {
      stringstream ss;
      ss << __FUNCTION__ << ": can't write header to output URL. Error code "
         << ret << endl;
      throw runtime_error(ss.str());
    }
  }

  ~MuxFrame_Impl() {
    av_write_trailer(outFmtCtx);
    if (outFmtCtx && !(outFmtCtx->oformat->flags & AVFMT_NOFILE))
      avio_closep(&outFmtCtx->pb);
    avformat_free_context(outFmtCtx);
  }
};
} // namespace VPF

MuxFrame *MuxFrame::Make(const char *url) { return new MuxFrame(url); }

MuxFrame::MuxFrame(const char *url)
    : Task("MuxFrame", MuxFrame::numInputs, MuxFrame::numOutputs) {
  output = (char *)calloc(strlen(url) + 1, sizeof(char));
  strcpy(output, url);
}

MuxFrame::~MuxFrame() {
  if (pImpl) {
    delete pImpl;
  }

  if (output) {
    free(output);
  }
}

TaskExecStatus MuxFrame::Execute() {
  auto elementaryVideo = (Buffer *)GetInput(0U);
  auto muxingParamsBuffer = (Buffer *)GetInput(1U);

  if (!muxingParamsBuffer) {
    return TASK_EXEC_FAIL;
  }

  auto muxingParams = muxingParamsBuffer->GetDataAs<MuxingParams>();
  if (!pImpl) {
    pImpl = new MuxFrame_Impl(*muxingParams, output);
  }

  auto FindMappedStreamIndex = [&](map<uint32_t, uint32_t> &map,
                                   int32_t nativeStreamIndex) {
    auto MappedIdxIt = map.find(nativeStreamIndex);
    if (MappedIdxIt == map.end()) {
      stringstream ss;
      ss << __FUNCTION__ << ": didn't found mapping for native stream #"
         << nativeStreamIndex << endl;
      throw runtime_error(ss.str());
    } else {
      return MappedIdxIt->second;
    }
  };

  auto writePacket = [&](Buffer &elementaryData, MuxingParams &muxParams,
                         AVStream *stream, AVFormatContext *outFmtCtx,
                         map<uint32_t, uint32_t> &streamMapping,
                         uint32_t nativeStreamIndex) {
    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.size = 0U;
    pkt.data = nullptr;

    pkt.size = elementaryData.GetRawMemSize();
    pkt.data = (uint8_t *)elementaryData.GetRawMemPtr();
    pkt.stream_index = FindMappedStreamIndex(streamMapping, nativeStreamIndex);

    auto timeBase = stream->time_base;
    auto &packetData = muxParams.videoContext.packetData;
    pkt.pos = -1;

    auto ret = av_interleaved_write_frame(outFmtCtx, &pkt);
    if (ret < 0) {
      stringstream ss;
      ss << __FUNCTION__ << ": can't write video packet to URL. Error code "
         << ret << endl;
      throw runtime_error(ss.str());
    }
  };

  try {
    if (elementaryVideo) {
      writePacket(*elementaryVideo, *muxingParams, pImpl->videoStream,
                  pImpl->outFmtCtx, pImpl->streamMapping,
                  muxingParams->videoContext.streamIndex);
    } else {
      return TASK_EXEC_FAIL;
    }
  } catch (exception &e) {
    cerr << e.what() << endl;
    return TASK_EXEC_FAIL;
  }

  return TASK_EXEC_SUCCESS;
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

  virtual TaskExecStatus Execute(Surface &source) = 0;
};

struct NppResizeSurfaceRGB_Impl final : ResizeSurface_Impl {
  NppResizeSurfaceRGB_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                           CUstream str)
      : ResizeSurface_Impl(width, height, RGB, ctx, str) {
    pSurface = Surface::Make(RGB, width, height, ctx);
  }

  ~NppResizeSurfaceRGB_Impl() { delete pSurface; }

  TaskExecStatus Execute(Surface &source) {

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    auto srcPlane = source.GetSurfacePlane();
    auto dstPlane = pSurface->GetSurfacePlane();

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
    int eInterpolation = NPPI_INTER_LINEAR;

    NppLock lock(nppCtx);
    CudaCtxPush ctxPush(cu_ctx);
    auto ret = nppiResize_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                     pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppCtx);
    if (NPP_NO_ERROR != ret) {
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }

  NppStreamContext nppCtx;
  CUcontext ctx;
};

struct NppResizeSurfaceYUV420_Impl final : ResizeSurface_Impl {
  NppResizeSurfaceYUV420_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                              CUstream str)
      : ResizeSurface_Impl(width, height, YUV420, ctx, str) {
    pSurface = Surface::Make(YUV420, width, height, ctx);
  }

  ~NppResizeSurfaceYUV420_Impl() { delete pSurface; }

  TaskExecStatus Execute(Surface &source) {

    if (pSurface->PixelFormat() != source.PixelFormat()) {
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
      int eInterpolation = NPPI_INTER_SUPER;

      NppLock lock(nppCtx);
      CudaCtxPush ctxPush(cu_ctx);
      auto ret = nppiResize_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                       pDst, nDstStep, oDstSize, oDstRectROI,
                                       eInterpolation, nppCtx);
      if (NPP_NO_ERROR != ret) {
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
  if (RGB == format) {
    pImpl = new NppResizeSurfaceRGB_Impl(width, height, ctx, str);
  } else if (YUV420 == format) {
    pImpl = new NppResizeSurfaceYUV420_Impl(width, height, ctx, str);
  } /*else if (NV12 == format) {
    pImpl = new CudaResizeSurfaceNV12_Impl(width, height, ctx, str);
  }*/ else {
    stringstream ss;
    ss << __FUNCTION__;
    ss << ": pixel format not supported";
    throw runtime_error(ss.str());
  }
}

ResizeSurface::~ResizeSurface() { delete pImpl; }

TaskExecStatus ResizeSurface::Execute() {
  ClearOutputs();

  auto pInputSurface = (Surface *)GetInput();
  if (!pInputSurface) {
    return TASK_EXEC_FAIL;
  }

  if (TASK_EXEC_SUCCESS != pImpl->Execute(*pInputSurface)) {
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
