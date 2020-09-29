/*
 * Copyright 2020 NVIDIA Corporation
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

#include "VPF_Classes.hpp"

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <sstream>

using namespace std;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

auto MakeErrMsg = [](const char *msg, const char *func, uint32_t line_number) {
  std::stringstream ss;
  ss << __FILE__ << ":" << line_number << ". Function " << func
     << " returned error: " << msg << endl;
  return ss.str();
};

VpfFrameUploader::VpfFrameUploader(const VpfFrameUploaderContext &ctx) {
  upCtx.reset(new VpfFrameUploaderContext(ctx));

  upUploader.reset(
      CudaUploadFrame::Make(CudaResMgr::Instance().GetStream(upCtx->gpuID),
                            CudaResMgr::Instance().GetCtx(upCtx->gpuID),
                            upCtx->width, upCtx->height, upCtx->format));
}

bool VpfFrameUploader::UploadSingleFrame(VpfFrameUploaderArgs &args) {
  args.errorMessage = string("");

  auto pRawFrame = Buffer::Make(args.frameSize, (void *)args.frame);
  upUploader->SetInput(pRawFrame, 0U);
  auto res = upUploader->Execute();
  delete pRawFrame;

  if (TASK_EXEC_FAIL == res) {
    args.errorMessage =
        MakeErrMsg("Can't upload frame to GPU", __FUNCTION__, __LINE__);
    return false;
  }

  auto pSurface = (Surface *)upUploader->GetOutput(0U);
  if (!pSurface) {
    args.errorMessage =
        MakeErrMsg("Can't get uploader output surface", __FUNCTION__, __LINE__);
    return false;
  } else {
    args.surface = shared_ptr<Surface>(pSurface->Clone());
    return true;
  }
}

VpfSurfaceDownloader::VpfSurfaceDownloader(
    const VpfSurfaceDownloaderContext &ctx) {
  upCtx.reset(new VpfSurfaceDownloaderContext(ctx));

  upDownloader.reset(
      CudaDownloadSurface::Make(CudaResMgr::Instance().GetStream(upCtx->gpuID),
                                CudaResMgr::Instance().GetCtx(upCtx->gpuID),
                                upCtx->width, upCtx->height, upCtx->format));
}

bool VpfSurfaceDownloader::DownloadSingleSurface(
    VpfSurfaceDownloaderArgs &args) {
  args.errorMessage = string("");

  upDownloader->SetInput(args.surface.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    args.errorMessage =
        MakeErrMsg("Can't download surface from GPU", __FUNCTION__, __LINE__);
    return false;
  }

  auto *pRawFrame = (Buffer *)upDownloader->GetOutput(0U);
  if (pRawFrame) {
    args.frameSize = pRawFrame->GetRawMemSize();
    args.frame.reset(new uint8_t[args.frameSize]);
    memcpy((void *)args.frame.get(), pRawFrame->GetRawMemPtr(), args.frameSize);
    return true;
  } else {
    args.errorMessage =
        MakeErrMsg("Can't get downloader output", __FUNCTION__, __LINE__);
    return false;
  }

  return false;
}

VpfSurfaceConverter::VpfSurfaceConverter(
    const VpfSurfaceConverterContext &ctx) {
  upCtx.reset(new VpfSurfaceConverterContext(ctx));

  upConverter.reset(ConvertSurface::Make(
      upCtx->width, upCtx->height, upCtx->srcFormat, upCtx->dstFormat,
      CudaResMgr::Instance().GetCtx(upCtx->gpuID),
      CudaResMgr::Instance().GetStream(upCtx->gpuID)));
}

bool VpfSurfaceConverter::ConvertSingleSurface(VpfSurfaceConverterArgs &args) {
  args.errorMessage = string("");

  if (!args.srcSurface) {
    args.errorMessage =
        MakeErrMsg("No input surface given", __FUNCTION__, __LINE__);
    return false;
  }

  upConverter->SetInput(args.srcSurface.get(), 0U);
  if (TASK_EXEC_SUCCESS != upConverter->Execute()) {
    args.errorMessage =
        MakeErrMsg("Can't perform color conversion", __FUNCTION__, __LINE__);
    return false;
  }

  auto pSurface = (Surface *)upConverter->GetOutput(0U);
  if (pSurface) {
    args.dstSurface = shared_ptr<Surface>(pSurface->Clone());
    return true;
  } else {
    args.errorMessage =
        MakeErrMsg("Can't get converter output", __FUNCTION__, __LINE__);
    return false;
  }
}

VpfSurfaceResizer::VpfSurfaceResizer(const VpfSurfaceResizerContext &ctx) {
  upCtx.reset(new VpfSurfaceResizerContext(ctx));

  upResizer.reset(
      ResizeSurface::Make(upCtx->width, upCtx->height, upCtx->format,
                          CudaResMgr::Instance().GetCtx(upCtx->gpuID),
                          CudaResMgr::Instance().GetStream(upCtx->gpuID)));
}

bool VpfSurfaceResizer::ResizeSingleSurface(VpfSurfaceResizerArgs &args) {
  args.errorMessage = string("");

  if (!args.srcSurface) {
    args.errorMessage =
        MakeErrMsg("No input surface given", __FUNCTION__, __LINE__);
    return false;
  }

  upResizer->SetInput(args.srcSurface.get(), 0U);

  if (TASK_EXEC_SUCCESS != upResizer->Execute()) {
    args.errorMessage =
        MakeErrMsg("Can't resize surface", __FUNCTION__, __LINE__);
    return false;
  }

  auto pSurface = (Surface *)upResizer->GetOutput(0U);
  if (pSurface) {
    args.dstSurface = shared_ptr<Surface>(pSurface->Clone());
    return true;
  } else {
    args.errorMessage =
        MakeErrMsg("Can't get resizer output", __FUNCTION__, __LINE__);
    return false;
  }
}

VpfFfmpegDecoder::VpfFfmpegDecoder(const VpfFfmpegDecoderContext &ctx) {
  upCtx.reset(new VpfFfmpegDecoderContext(ctx));

  NvDecoderClInterface cli_iface(upCtx->ffmpegOptions);
  upDecoder.reset(
      FfmpegDecodeFrame::Make(upCtx->pathToFile.c_str(), cli_iface));
}

void *VpfFfmpegDecoder::GetSideData(AVFrameSideDataType data_type,
                                    size_t &raw_size) {
  if (TASK_EXEC_SUCCESS == upDecoder->GetSideData(data_type)) {
    auto pSideData = (Buffer *)upDecoder->GetOutput(1U);
    if (pSideData) {
      raw_size = pSideData->GetRawMemSize();
      return pSideData->GetDataAs<void>();
    }
  }
  return nullptr;
}

bool VpfFfmpegDecoder::GetMotionVectors(VpfFfmpegDecoderArgs &args) {
  size_t size = 0U;
  auto ptr = (AVMotionVector *)GetSideData(AV_FRAME_DATA_MOTION_VECTORS, size);
  size /= sizeof(*ptr);

  if (ptr && size) {
    args.motionVectors.reset(new MotionVector[size]);
    auto mvc = args.motionVectors.get();

    for (auto i = 0; i < size; i++) {
      mvc[i].source = ptr[i].source;
      mvc[i].w = ptr[i].w;
      mvc[i].h = ptr[i].h;
      mvc[i].src_x = ptr[i].src_x;
      mvc[i].src_y = ptr[i].src_y;
      mvc[i].dst_x = ptr[i].dst_x;
      mvc[i].dst_y = ptr[i].dst_y;
      mvc[i].motion_x = ptr[i].motion_x;
      mvc[i].motion_y = ptr[i].motion_y;
      mvc[i].motion_scale = ptr[i].motion_scale;
    }

    return true;
  } else {
    args.errorMessage =
        MakeErrMsg("Can't get motion vectors", __FUNCTION__, __LINE__);
    return false;
  }
}

bool VpfFfmpegDecoder::DecodeSingleFrame(VpfFfmpegDecoderArgs &args) {
  args.errorMessage = string("");

  if (TASK_EXEC_SUCCESS != upDecoder->Execute()) {
    args.errorMessage =
        MakeErrMsg("Can't decode frame", __FUNCTION__, __LINE__);
    return false;
  }

  auto pRawFrame = (Buffer *)upDecoder->GetOutput(0U);
  if (!pRawFrame) {
    args.errorMessage =
        MakeErrMsg("Can't get decoder output", __FUNCTION__, __LINE__);
    return false;
  }

  args.frameSize = pRawFrame->GetRawMemSize();
  args.frame.reset(new uint8_t[args.frameSize]);
  memcpy((void *)args.frame.get(), pRawFrame->GetRawMemPtr(), args.frameSize);

  if (args.needMotionVectors) {
    return GetMotionVectors(args);
  }

  return true;
}

VpfNvDecoder::VpfNvDecoder(const VpfNvDecoderContext &ctx) {
  upCtx.reset(new VpfNvDecoderContext(ctx));

  vector<const char *> options;
  for (auto &pair : ctx.ffmpegOptions) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }

  upDemuxer.reset(
      DemuxFrame::Make(ctx.pathToFile.c_str(), options.data(), options.size()));

  MuxingParams params;
  upDemuxer->GetParams(params);

  upDecoder.reset(NvdecDecodeFrame::Make(
      CudaResMgr::Instance().GetStream(upCtx->gpuID),
      CudaResMgr::Instance().GetCtx(upCtx->gpuID), params.videoContext.codec,
      poolFrameSize, params.videoContext.width, params.videoContext.height));
}

Buffer *VpfNvDecoder::GetVideoPacket() {
  Buffer *elementaryVideo = nullptr;
  /* Demuxer may also extracts elementary audio etc. from stream, so we run
   * it until we get elementary video;
   */
  do {
    if (TASK_EXEC_FAIL == upDemuxer->Execute()) {
      return nullptr;
    }
    elementaryVideo = (Buffer *)upDemuxer->GetOutput(0U);
  } while (!elementaryVideo);

  return elementaryVideo;
}

bool VpfNvDecoder::DecodeSurfaceInternal(VpfNvDecoderArgs &args) {
  args.decoderHwReset = false;
  args.decodedSurface = nullptr;

  do {
    /* Get encoded frame from demuxer;
     * May be null, but that's ok - it will flush decoder;
     */
    auto elementaryVideo = GetVideoPacket();

    /* Kick off HW decoding;
     * We may not have decoded surface here as decoder is async;
     * Decoder may throw exception. In such situation we reset it;
     */
    upDecoder->SetInput(elementaryVideo, 0U);
    try {
      if (TASK_EXEC_FAIL == upDecoder->Execute()) {
        args.errorMessage.append(
            MakeErrMsg("Can't decode a surface", __FUNCTION__, __LINE__));
        args.decodedSurface = nullptr;
        return false;
      }
    } catch (exception &e) {
      args.errorMessage.append(
          MakeErrMsg("HW decoder was reset", __FUNCTION__, __LINE__));
      args.decoderHwReset = true;
      args.decodedSurface = nullptr;
      return false;
    }

    auto pDecodedSurface = (Surface *)upDecoder->GetOutput(0U);
    if (pDecodedSurface) {
      args.decodedSurface = shared_ptr<Surface>(pDecodedSurface->Clone());
    }
    /* Repeat untill we got decoded surface;
     */
  } while (!args.decodedSurface);

  return true;
}

bool VpfNvDecoder::FlushSingleSurface(VpfNvDecoderArgs &args) {
  args.decodedSurface = nullptr;

  auto *elementaryVideo = Buffer::Make(0U);
  upDecoder->SetInput(elementaryVideo, 0U);
  auto res = upDecoder->Execute();
  delete elementaryVideo;

  if (TASK_EXEC_FAIL == res) {
    args.errorMessage.append(
        MakeErrMsg("Can't decode a surface", __FUNCTION__, __LINE__));
    args.decodedSurface = nullptr;
    return false;
  }

  auto pDecodedSurface = (Surface *)upDecoder->GetOutput(0U);
  args.decodedSurface = shared_ptr<Surface>(pDecodedSurface->Clone());
  return true;
}

void VpfNvDecoder::LastPacketData(PacketData &packetData) const {
  auto mp_buffer = (Buffer *)upDemuxer->GetOutput(1U);
  if (mp_buffer) {
    auto mp = mp_buffer->GetDataAs<MuxingParams>();
    packetData = mp->videoContext.packetData;
  }
}

uint32_t VpfNvDecoder::Width() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.width;
}

uint32_t VpfNvDecoder::Height() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.height;
}

double VpfNvDecoder::Framerate() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.frameRate;
}

double VpfNvDecoder::Timebase() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.timeBase;
}

Pixel_Format VpfNvDecoder::GetPixelFormat() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.format;
}

bool VpfNvDecoder::DecodeSingleSurface(VpfNvDecoderArgs &args) {
  if (!DecodeSurfaceInternal(args)) {
    return false;
  }

  if (args.decoderHwReset) {
    MuxingParams params;
    upDemuxer->GetParams(params);

    upDecoder.reset(NvdecDecodeFrame::Make(
        CudaResMgr::Instance().GetStream(upCtx->gpuID),
        CudaResMgr::Instance().GetCtx(upCtx->gpuID), params.videoContext.codec,
        poolFrameSize, params.videoContext.width, params.videoContext.height));

    return false;
  }

  if (args.needSei) {
    auto seiBuffer = (Buffer *)upDemuxer->GetOutput(2U);
    if (seiBuffer) {
      args.sei.reset(new uint8_t[seiBuffer->GetRawMemSize()]);
      memcpy(args.sei.get(), seiBuffer->GetRawMemPtr(),
             seiBuffer->GetRawMemSize());
    } else {
      args.errorMessage.append(
          MakeErrMsg("Can't get SEI message", __FUNCTION__, __LINE__));
      return false;
    }
  }
  return true;
}

bool VpfNvDecoder::DecodeSingleFrame(VpfNvDecoderArgs &args) {
  args.errorMessage.erase();

  if (!DecodeSingleSurface(args)) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);

    VpfSurfaceDownloaderContext ctx(upCtx->gpuID, width, height, NV12);
    upDownloader.reset(new VpfSurfaceDownloader(ctx));
  }

  VpfSurfaceDownloaderArgs downloaderArgs = {0};
  downloaderArgs.surface = args.decodedSurface;
  auto res = upDownloader->DownloadSingleSurface(downloaderArgs);
  if (!res) {
    args.errorMessage = downloaderArgs.errorMessage;
    return false;
  }

  args.decodedFrame = downloaderArgs.frame;
  args.decodedFrameSize = downloaderArgs.frameSize;
  return true;
}

bool VpfNvEncoder::Reconfigure(const VpfNvEncoderReconfigureContext &ctx) {
  if (upEncoder) {
    NvEncoderClInterface cli_interface(upCtx->options);
    return upEncoder->Reconfigure(cli_interface, ctx.forceIDR, ctx.reset,
                                  ctx.verbose);
  }

  return true;
}

VpfNvEncoder::VpfNvEncoder(const VpfNvEncoderContext &ctx) {
  upCtx.reset(new VpfNvEncoderContext(ctx));

  // Parse resolution;
  auto ParseResolution = [&](const string &res_string, uint32_t &width,
                             uint32_t &height) {
    string::size_type xPos = res_string.find('x');

    if (xPos != string::npos) {
      // Parse width;
      stringstream ssWidth;
      ssWidth << res_string.substr(0, xPos);
      ssWidth >> width;

      // Parse height;
      stringstream ssHeight;
      ssHeight << res_string.substr(xPos + 1);
      ssHeight >> height;
    } else {
      throw invalid_argument("Invalid resolution.");
    }
  };

  auto it = upCtx->options.find("s");
  if (it != upCtx->options.end()) {
    ParseResolution(it->second, upCtx->width, upCtx->height);
  } else {
    throw invalid_argument("No resolution given");
  }

  // Parse pixel format;
  string fmt_string;
  switch (upCtx->format) {
  case NV12:
    fmt_string = "NV12";
    break;
  case YUV444:
    fmt_string = "YUV444";
    break;
  default:
    fmt_string = "UNDEFINED";
    break;
  }

  it = upCtx->options.find("fmt");
  if (it != upCtx->options.end()) {
    it->second = fmt_string;
  } else {
    upCtx->options["fmt"] = fmt_string;
  }

  if (upCtx->gpuID < 0 || upCtx->gpuID >= CudaResMgr::Instance().GetNumGpus()) {
    upCtx->gpuID = 0U;
  }
  cout << "Encoding on GPU " << upCtx->gpuID << endl;

  /* Don't initialize uploader & encoder here, ust prepare config params;
   */
  VpfNvEncoderReconfigureContext reconfCtx;
  reconfCtx.options = upCtx->options;
  reconfCtx.forceIDR = false;
  reconfCtx.reset = false;
  reconfCtx.verbose = upCtx->verbose;

  Reconfigure(reconfCtx);
}

bool VpfNvEncoder::EncodeSingleSurface(VpfNvEncoderArgs &args) {
  args.errorMessage.erase();

  shared_ptr<Buffer> spSEI = nullptr;
  if (args.seiMessage && args.seiMessageSize) {
    spSEI = shared_ptr<Buffer>(
        Buffer::MakeOwnMem(args.seiMessageSize, (const void *)args.seiMessage));
  }

  if (!upEncoder) {
    NvEncoderClInterface cli_interface(upCtx->options);

    upEncoder.reset(NvencEncodeFrame::Make(
        CudaResMgr::Instance().GetStream(upCtx->gpuID),
        CudaResMgr::Instance().GetCtx(upCtx->gpuID), cli_interface,
        NV12 == upCtx->format
            ? NV_ENC_BUFFER_FORMAT_NV12
            : YUV444 == upCtx->format ? NV_ENC_BUFFER_FORMAT_YUV444
                                      : NV_ENC_BUFFER_FORMAT_UNDEFINED,
        upCtx->width, upCtx->height, upCtx->verbose));
  }

  upEncoder->ClearInputs();

  if (args.surface) {
    upEncoder->SetInput(args.surface.get(), 0U);
  } else {
    /* Flush encoder this way;
     */
    upEncoder->SetInput(nullptr, 0U);
  }

  if (args.sync) {
    /* Set 2nd input to any non-zero value
     * to signal sync encode;
     */
    upEncoder->SetInput((Token *)0xdeadbeef, 1U);
  }

  if (spSEI) {
    /* Set 3rd input in case we have SEI message;
     */
    upEncoder->SetInput(spSEI.get(), 2U);
  }

  if (TASK_EXEC_FAIL == upEncoder->Execute()) {
    args.errorMessage.append(
        MakeErrMsg("Can't encode surface", __FUNCTION__, __LINE__));
    return false;
  }

  auto encodedFrame = (Buffer *)upEncoder->GetOutput(0U);
  if (encodedFrame) {
    if (args.append) {
      auto new_size = args.packetSize + encodedFrame->GetRawMemSize();
      auto new_packet = make_shared<uint8_t>(new_size);
      memcpy(new_packet.get(), args.packet.get(), args.packetSize);
      memcpy(new_packet.get() + args.packetSize, encodedFrame->GetRawMemPtr(),
             encodedFrame->GetRawMemSize());

      swap(args.packet, new_packet);
      args.packetSize = new_size;
    } else {
      args.packet.reset(new uint8_t[encodedFrame->GetRawMemSize()]);
      memcpy(args.packet.get(), encodedFrame->GetRawMemPtr(),
             encodedFrame->GetRawMemSize());
    }
    return true;
  }

  return false;
}

bool VpfNvEncoder::EncodeSingleFrame(VpfNvEncoderArgs &args) {
  if (!upUploader) {
    VpfSurfaceDownloaderContext ctx(upCtx->gpuID, upCtx->width, upCtx->height,
                                    upCtx->format);
    upUploader.reset(new VpfFrameUploader(ctx));
  }

  VpfFrameUploaderArgs uploaderArgs(args.frame, args.frameSize);

  if (!upUploader->UploadSingleFrame(uploaderArgs)) {
    args.errorMessage.append(
        MakeErrMsg("Can't upload surface", __FUNCTION__, __LINE__));
    return false;
  }

  args.surface = uploaderArgs.surface;

  return EncodeSingleSurface(args);
}

bool VpfNvEncoder::Flush(VpfNvEncoderArgs &args) {
  uint32_t num_packets = 0U;
  do {
    /* Keep feeding encoder with null input until it returns zero-size
     * surface; */
    args.append = true;
    args.surface = nullptr;
    auto success = EncodeSingleSurface(args);
    if (!success) {
      break;
    }
    num_packets++;
  } while (true);

  return (num_packets > 0U);
}

static auto ThrowOnCudaError = [](CUresult res, int lineNum = -1) {
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

CudaResMgr::CudaResMgr() {
  ThrowOnCudaError(cuInit(0), __LINE__);

  int nGpu;
  ThrowOnCudaError(cuDeviceGetCount(&nGpu), __LINE__);

  for (int i = 0; i < nGpu; i++) {
    CUcontext cuContext = nullptr;
    CUstream cuStream = nullptr;

    g_Contexts.push_back(cuContext);
    g_Streams.push_back(cuStream);
  }
  return;
}

CudaResMgr &CudaResMgr::Instance() {
  static CudaResMgr instance;
  return instance;
}

CUcontext CudaResMgr::GetCtx(size_t idx) {
  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto &ctx = g_Contexts[idx];
  if (!ctx) {
    CUdevice cuDevice = 0;
    ThrowOnCudaError(cuDeviceGet(&cuDevice, idx), __LINE__);
    ThrowOnCudaError(cuCtxCreate(&ctx, 0, cuDevice), __LINE__);
  }

  return g_Contexts[idx];
}

CUstream CudaResMgr::GetStream(size_t idx) {
  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto &str = g_Streams[idx];
  if (!str) {
    auto ctx = GetCtx(idx);
    CudaCtxPush push(ctx);
    ThrowOnCudaError(cuStreamCreate(&str, 0), __LINE__);
  }

  return g_Streams[idx];
}

CudaResMgr::~CudaResMgr() {
  stringstream ss;
  try {
    for (auto &cuStream : g_Streams) {
      if (cuStream) {
        ThrowOnCudaError(cuStreamDestroy(cuStream), __LINE__);
      }
    }
    g_Streams.clear();

    for (auto &cuContext : g_Contexts) {
      if (cuContext) {
        ThrowOnCudaError(cuCtxDestroy(cuContext), __LINE__);
      }
    }
    g_Contexts.clear();
  } catch (runtime_error &e) {
    cerr << e.what() << endl;
  }

#ifdef TRACK_TOKEN_ALLOCATIONS
  cout << "Checking token allocation counters: ";
  auto res = CheckAllocationCounters();
  cout << (res ? "No leaks dectected" : "Leaks detected") << endl;
#endif
}

size_t CudaResMgr::GetNumGpus() { return Instance().g_Contexts.size(); }