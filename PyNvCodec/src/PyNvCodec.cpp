/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Kognia Sports Intelligence
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

#include "PyNvCodec.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

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

class CudaResMgr {
private:
  CudaResMgr() {
    lock_guard<mutex> lock_ctx(CudaResMgr::gInsMutex);

    ThrowOnCudaError(cuInit(0), __LINE__);

    int nGpu;
    ThrowOnCudaError(cuDeviceGetCount(&nGpu), __LINE__);


    for (int i = 0; i < nGpu; i++) {
      CUdevice cuDevice = 0;
      CUcontext cuContext = nullptr;
      g_Contexts.push_back(make_pair(cuDevice,cuContext));

      CUstream cuStream = nullptr;
      g_Streams.push_back(cuStream);
    }
    return;
  }

public:
  CUcontext GetCtx(size_t idx) {
    lock_guard<mutex> lock_ctx(CudaResMgr::gCtxMutex);
    
    if (idx >= GetNumGpus()) {
      return nullptr;
    }
    
    auto &ctx = g_Contexts[idx];
    if (!ctx.second) {
      CUdevice cuDevice = 0;
      ThrowOnCudaError(cuDeviceGet(&cuDevice, idx), __LINE__);
      ThrowOnCudaError(cuDevicePrimaryCtxRetain(&ctx.second, cuDevice), __LINE__);
    }

    return g_Contexts[idx].second;
  }

  CUstream GetStream(size_t idx) {
    lock_guard<mutex> lock_ctx(CudaResMgr::gStrMutex);

    if (idx >= GetNumGpus()) {
      return nullptr;
    }

    auto &str = g_Streams[idx];
    if (!str) {
      auto ctx = GetCtx(idx);
      CudaCtxPush push(ctx);
      ThrowOnCudaError(cuStreamCreate(&str, CU_STREAM_NON_BLOCKING), __LINE__);
    }

    return g_Streams[idx];
  }

  ~CudaResMgr() {
    lock_guard<mutex> ins_lock(CudaResMgr::gInsMutex);
    lock_guard<mutex> ctx_lock(CudaResMgr::gCtxMutex);
    lock_guard<mutex> str_lock(CudaResMgr::gStrMutex);

    stringstream ss;
    try {
      {
        for (auto &cuStream : g_Streams) {
          if (cuStream) {
            ThrowOnCudaError(cuStreamDestroy(cuStream), __LINE__);
          }
        }
        g_Streams.clear();
      }

      {
        for (int i=0;i<g_Contexts.size();i++) {
          if (g_Contexts[i].second) {
            ThrowOnCudaError(cuDevicePrimaryCtxRelease(g_Contexts[i].first), __LINE__);
          }
        }
        g_Contexts.clear();
      }
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }

#ifdef TRACK_TOKEN_ALLOCATIONS
    cout << "Checking token allocation counters: ";
    auto res = CheckAllocationCounters();
    cout << (res ? "No leaks dectected" : "Leaks detected") << endl;
#endif
  }

  static CudaResMgr &Instance() {
    static CudaResMgr instance;
    return instance;
  }

  static size_t GetNumGpus() { 
    return Instance().g_Contexts.size(); 
  }

  vector<pair<CUdevice,CUcontext>> g_Contexts;
  vector<CUstream> g_Streams;

  static mutex gInsMutex;
  static mutex gCtxMutex;
  static mutex gStrMutex;
};

mutex CudaResMgr::gInsMutex;
mutex CudaResMgr::gCtxMutex;
mutex CudaResMgr::gStrMutex;

PyFrameUploader::PyFrameUploader(uint32_t width, uint32_t height,
                                 Pixel_Format format, uint32_t gpu_ID) {
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  uploader.reset(CudaUploadFrame::Make(CudaResMgr::Instance().GetStream(gpu_ID),
                                       CudaResMgr::Instance().GetCtx(gpu_ID),
                                       surfaceWidth, surfaceHeight,
                                       surfaceFormat));
}

PyFrameUploader::PyFrameUploader(uint32_t width, uint32_t height,
                                 Pixel_Format format, CUcontext ctx, CUstream str) {
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  uploader.reset(CudaUploadFrame::Make(str, ctx, surfaceWidth, surfaceHeight,
                                       surfaceFormat));
}

Pixel_Format PyFrameUploader::GetFormat() { return surfaceFormat; }

/* Will upload numpy array to GPU;
 * Surface returned is valid untill next call;
 */
shared_ptr<Surface>
PyFrameUploader::UploadSingleFrame(py::array_t<uint8_t> &frame) {
  /* Upload to GPU;
   */
  auto pRawFrame = Buffer::Make(frame.size(), frame.mutable_data());
  uploader->SetInput(pRawFrame, 0U);
  auto res = uploader->Execute();
  delete pRawFrame;

  if (TASK_EXEC_FAIL == res) {
    throw runtime_error("Error uploading frame to GPU");
  }

  /* Get surface;
   */
  auto pSurface = (Surface *)uploader->GetOutput(0U);
  if (!pSurface) {
    throw runtime_error("Error uploading frame to GPU");
  }

  return shared_ptr<Surface>(pSurface->Clone());
}

PySurfaceDownloader::PySurfaceDownloader(uint32_t width, uint32_t height,
                                         Pixel_Format format, uint32_t gpu_ID) {
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  upDownloader.reset(
      CudaDownloadSurface::Make(CudaResMgr::Instance().GetStream(gpu_ID),
                                CudaResMgr::Instance().GetCtx(gpu_ID),
                                surfaceWidth, surfaceHeight, surfaceFormat));
}

PySurfaceDownloader::PySurfaceDownloader(uint32_t width, uint32_t height,
                                         Pixel_Format format, CUcontext ctx, 
                                         CUstream str) {
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  upDownloader.reset(
      CudaDownloadSurface::Make(str, ctx, surfaceWidth, surfaceHeight, surfaceFormat));
}

Pixel_Format PySurfaceDownloader::GetFormat() { return surfaceFormat; }

bool PySurfaceDownloader::DownloadSingleSurface(shared_ptr<Surface> surface,
                                                py::array_t<uint8_t> &frame) {
  upDownloader->SetInput(surface.get(), 0U);
  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  auto *pRawFrame = (Buffer *)upDownloader->GetOutput(0U);
  if (pRawFrame) {
    auto const downloadSize = pRawFrame->GetRawMemSize();
    if (downloadSize != frame.size()) {
      frame.resize({downloadSize}, false);
    }

    memcpy(frame.mutable_data(), pRawFrame->GetRawMemPtr(), downloadSize);
    return true;
  }

  return false;
}

PySurfaceConverter::PySurfaceConverter(uint32_t width, uint32_t height,
                                       Pixel_Format inFormat,
                                       Pixel_Format outFormat, uint32_t gpuID)
    : outputFormat(outFormat) {
  upConverter.reset(ConvertSurface::Make(
      width, height, inFormat, outFormat, CudaResMgr::Instance().GetCtx(gpuID),
      CudaResMgr::Instance().GetStream(gpuID)));
  upCtxBuffer.reset(Buffer::MakeOwnMem(sizeof(ColorspaceConversionContext)));
}

PySurfaceConverter::PySurfaceConverter(uint32_t width, uint32_t height,
                                       Pixel_Format inFormat,
                                       Pixel_Format outFormat, CUcontext ctx, 
                                       CUstream str)
    : outputFormat(outFormat) {
  upConverter.reset(ConvertSurface::Make(
      width, height, inFormat, outFormat, ctx, str));
  upCtxBuffer.reset(Buffer::MakeOwnMem(sizeof(ColorspaceConversionContext)));
}

shared_ptr<Surface>
PySurfaceConverter::Execute(shared_ptr<Surface> surface,
                            shared_ptr<ColorspaceConversionContext> context) {
  if (!surface) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  upConverter->ClearInputs();

  upConverter->SetInput(surface.get(), 0U);
  
  if (context) {
    upCtxBuffer->CopyFrom(sizeof(ColorspaceConversionContext), context.get());
    upConverter->SetInput((Token *)upCtxBuffer.get(), 1U);
  }
  
  if (TASK_EXEC_SUCCESS != upConverter->Execute()) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  auto pSurface = (Surface *)upConverter->GetOutput(0U);
  return shared_ptr<Surface>(pSurface ? pSurface->Clone()
                                      : Surface::Make(outputFormat));
}

Pixel_Format PySurfaceConverter::GetFormat() { return outputFormat; }

PySurfaceResizer::PySurfaceResizer(uint32_t width, uint32_t height,
                                   Pixel_Format format, uint32_t gpuID)
    : outputFormat(format) {
  upResizer.reset(ResizeSurface::Make(width, height, format,
                                      CudaResMgr::Instance().GetCtx(gpuID),
                                      CudaResMgr::Instance().GetStream(gpuID)));
}

PySurfaceResizer::PySurfaceResizer(uint32_t width, uint32_t height,
                                   Pixel_Format format, CUcontext ctx, 
                                   CUstream str)
    : outputFormat(format) {
  upResizer.reset(ResizeSurface::Make(width, height, format, ctx, str));
}

Pixel_Format PySurfaceResizer::GetFormat() { return outputFormat; }

shared_ptr<Surface> PySurfaceResizer::Execute(shared_ptr<Surface> surface) {
  if (!surface) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  upResizer->SetInput(surface.get(), 0U);

  if (TASK_EXEC_SUCCESS != upResizer->Execute()) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  auto pSurface = (Surface *)upResizer->GetOutput(0U);
  return shared_ptr<Surface>(pSurface ? pSurface->Clone()
                                      : Surface::Make(outputFormat));
}

PyFfmpegDecoder::PyFfmpegDecoder(const string &pathToFile,
                                 const map<string, string> &ffmpeg_options) {
  NvDecoderClInterface cli_iface(ffmpeg_options);
  upDecoder.reset(FfmpegDecodeFrame::Make(pathToFile.c_str(), cli_iface));
}

bool PyFfmpegDecoder::DecodeSingleFrame(py::array_t<uint8_t> &frame) {
  if (TASK_EXEC_SUCCESS == upDecoder->Execute()) {
    auto pRawFrame = (Buffer *)upDecoder->GetOutput(0U);
    if (pRawFrame) {
      auto const frame_size = pRawFrame->GetRawMemSize();
      if (frame_size != frame.size()) {
        frame.resize({frame_size}, false);
      }

      memcpy(frame.mutable_data(), pRawFrame->GetRawMemPtr(), frame_size);
      return true;
    }
  }
  return false;
}

void *PyFfmpegDecoder::GetSideData(AVFrameSideDataType data_type,
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

py::array_t<MotionVector> PyFfmpegDecoder::GetMotionVectors() {
  size_t size = 0U;
  auto ptr = (AVMotionVector *)GetSideData(AV_FRAME_DATA_MOTION_VECTORS, size);
  size /= sizeof(*ptr);

  if (ptr && size) {
    py::array_t<MotionVector> mv({size});
    auto req = mv.request(true);
    auto mvc = static_cast<MotionVector *>(req.ptr);

    for (auto i = 0; i < req.shape[0]; i++) {
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

    return move(mv);
  }

  return move(py::array_t<MotionVector>({0}));
}

PyFFmpegDemuxer::PyFFmpegDemuxer(const string &pathToFile)
    : PyFFmpegDemuxer(pathToFile, map<string, string>()) {}

PyFFmpegDemuxer::PyFFmpegDemuxer(const string &pathToFile,
                                 const map<string, string> &ffmpeg_options) {
  vector<const char *> options;
  for (auto &pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));
}

bool PyFFmpegDemuxer::DemuxSinglePacket(py::array_t<uint8_t> &packet) {
  Buffer *elementaryVideo = nullptr;
  do {
    if (TASK_EXEC_FAIL == upDemuxer->Execute()) {
      upDemuxer->ClearInputs();
      return false;
    }
    elementaryVideo = (Buffer *)upDemuxer->GetOutput(0U);
  } while (!elementaryVideo);

  packet.resize({elementaryVideo->GetRawMemSize()}, false);
  memcpy(packet.mutable_data(), elementaryVideo->GetDataAs<void>(),
         elementaryVideo->GetRawMemSize());

  upDemuxer->ClearInputs();
  return true;
}

void PyFFmpegDemuxer::GetLastPacketData(PacketData &pkt_data) {
  auto pkt_data_buf = (Buffer*)upDemuxer->GetOutput(3U);
  if (pkt_data_buf) {
    auto pkt_data_ptr = pkt_data_buf->GetDataAs<PacketData>();
    pkt_data = *pkt_data_ptr;
  }
}

uint32_t PyFFmpegDemuxer::Width() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.width;
}

ColorSpace PyFFmpegDemuxer::GetColorSpace() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.color_space;
};

ColorRange PyFFmpegDemuxer::GetColorRange() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.color_range;
};

uint32_t PyFFmpegDemuxer::Height() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.height;
}

Pixel_Format PyFFmpegDemuxer::Format() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.format;
}

cudaVideoCodec PyFFmpegDemuxer::Codec() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.codec;
}

double PyFFmpegDemuxer::Framerate() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.frameRate;
}

double PyFFmpegDemuxer::AvgFramerate() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.avgFrameRate;
}

bool PyFFmpegDemuxer::IsVFR() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.is_vfr;  
}

double PyFFmpegDemuxer::Timebase() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.timeBase;
}

uint32_t PyFFmpegDemuxer::Numframes() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.num_frames;
}

bool PyFFmpegDemuxer::Seek(SeekContext &ctx, py::array_t<uint8_t> &packet) {
  Buffer *elementaryVideo = nullptr;
  auto pSeekCtxBuf = shared_ptr<Buffer>(Buffer::MakeOwnMem(sizeof(ctx), &ctx));
  do {
    upDemuxer->SetInput((Token *)pSeekCtxBuf.get(), 1U);
    if (TASK_EXEC_FAIL == upDemuxer->Execute()) {
      upDemuxer->ClearInputs();
      return false;
    }
    elementaryVideo = (Buffer *)upDemuxer->GetOutput(0U);
  } while (!elementaryVideo);

  packet.resize({elementaryVideo->GetRawMemSize()}, false);
  memcpy(packet.mutable_data(), elementaryVideo->GetDataAs<void>(),
         elementaryVideo->GetRawMemSize());

  auto pktDataBuf = (Buffer*)upDemuxer->GetOutput(3U);
  if (pktDataBuf) {
    auto pPktData = pktDataBuf->GetDataAs<PacketData>();
    ctx.out_frame_pts = pPktData->pts;
    ctx.out_frame_duration = pPktData->duration;
  }

  upDemuxer->ClearInputs();
  return true;
}

PyNvDecoder::PyNvDecoder(const string &pathToFile, int gpuOrdinal)
    : PyNvDecoder(pathToFile, gpuOrdinal, map<string, string>()) {}

PyNvDecoder::PyNvDecoder(const string &pathToFile, CUcontext ctx, CUstream str)
    : PyNvDecoder(pathToFile, ctx, str, map<string, string>()) {}

PyNvDecoder::PyNvDecoder(const string &pathToFile, int gpuOrdinal,
                         const map<string, string> &ffmpeg_options) {
  if (gpuOrdinal < 0 || gpuOrdinal >= CudaResMgr::Instance().GetNumGpus()) {
    gpuOrdinal = 0U;
  }
  gpuID = gpuOrdinal;
  cout << "Decoding on GPU " << gpuID << endl;

  vector<const char *> options;
  for (auto &pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));

  MuxingParams params;
  upDemuxer->GetParams(params);
  format = params.videoContext.format;

  upDecoder.reset(NvdecDecodeFrame::Make(
      CudaResMgr::Instance().GetStream(gpuID),
      CudaResMgr::Instance().GetCtx(gpuID), params.videoContext.codec,
      poolFrameSize, params.videoContext.width, params.videoContext.height,
      format));
}

PyNvDecoder::PyNvDecoder(const string &pathToFile, CUcontext ctx, CUstream str,
                         const map<string, string> &ffmpeg_options) {
  vector<const char *> options;
  for (auto &pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));

  MuxingParams params;
  upDemuxer->GetParams(params);
  format = params.videoContext.format;

  upDecoder.reset(NvdecDecodeFrame::Make(
      str, ctx, params.videoContext.codec,
      poolFrameSize, params.videoContext.width, params.videoContext.height,
      format));
}

PyNvDecoder::PyNvDecoder(uint32_t width, uint32_t height,
                         Pixel_Format new_format, cudaVideoCodec codec,
                         uint32_t gpuOrdinal)
    : format(new_format) {
  if (gpuOrdinal >= CudaResMgr::Instance().GetNumGpus()) {
    gpuOrdinal = 0U;
  }
  gpuID = gpuOrdinal;
  cout << "Decoding on GPU " << gpuID << endl;

  upDecoder.reset(
      NvdecDecodeFrame::Make(CudaResMgr::Instance().GetStream(gpuID),
                             CudaResMgr::Instance().GetCtx(gpuID), codec,
                             poolFrameSize, width, height, format));
}

PyNvDecoder::PyNvDecoder(uint32_t width, uint32_t height,
                         Pixel_Format new_format, cudaVideoCodec codec,
                         CUcontext ctx, CUstream str)
    : format(new_format)
{
  upDecoder.reset(
      NvdecDecodeFrame::Make(str, ctx, codec,
                             poolFrameSize, width, height, format));
}

Buffer *PyNvDecoder::getElementaryVideo(DemuxFrame *demuxer,
                                        SeekContext &seek_ctx, bool needSEI) {
  Buffer *elementaryVideo = nullptr;
  Buffer *pktData = nullptr;
  shared_ptr<Buffer> pSeekCtxBuf = nullptr;

  do {
    // Set 1st demuxer input to any non-zero value if we need SEI;
    if (needSEI) {
      demuxer->SetInput((Token *)0xdeadbeef, 0U);
    }

    // Set 2nd demuxer input to seek context if we need to seek;
    if (seek_ctx.use_seek) {
      pSeekCtxBuf =
          shared_ptr<Buffer>(Buffer::MakeOwnMem(sizeof(seek_ctx), &seek_ctx));
      demuxer->SetInput((Token *)pSeekCtxBuf.get(), 1U);
    }
    if (TASK_EXEC_FAIL == demuxer->Execute()) {
      return nullptr;
    }
    elementaryVideo = (Buffer *)demuxer->GetOutput(0U);

    /* Clear inputs and set down seek flag or we will seek
     * for one and the same frame multiple times. */
    seek_ctx.use_seek = false;
    demuxer->ClearInputs();
  } while (!elementaryVideo);

  auto pktDataBuf = (Buffer *)demuxer->GetOutput(3U);
  if (pktDataBuf) {
    auto pPktData = pktDataBuf->GetDataAs<PacketData>();
    seek_ctx.out_frame_pts = pPktData->pts;
    seek_ctx.out_frame_duration = pPktData->duration;
  }

  return elementaryVideo;
};

Surface *PyNvDecoder::getDecodedSurface(NvdecDecodeFrame *decoder,
                                        DemuxFrame *demuxer,
                                        SeekContext &seek_ctx,
                                        bool needSEI) {
  decoder->ClearInputs();
  decoder->ClearOutputs();

  Surface *surface = nullptr;
  do {
    auto elementaryVideo = getElementaryVideo(demuxer, seek_ctx, needSEI);
    auto pktData = (Buffer *)demuxer->GetOutput(3U);

    decoder->SetInput(elementaryVideo, 0U);
    decoder->SetInput(pktData, 1U);
    if (TASK_EXEC_FAIL == decoder->Execute()) {
      break;
    }

    surface = (Surface *)decoder->GetOutput(0U);
  } while (!surface);

  return surface;
};

Surface *PyNvDecoder::getDecodedSurfaceFromPacket(py::array_t<uint8_t> *pPacket,
                                                  bool no_eos) {
  upDecoder->ClearInputs();
  upDecoder->ClearOutputs();

  Surface *surface = nullptr;
  unique_ptr<Buffer> elementaryVideo = nullptr;

  if (pPacket && pPacket->size()) {
    elementaryVideo = unique_ptr<Buffer>(
        Buffer::MakeOwnMem(pPacket->size(), pPacket->data()));
  }

  if (no_eos) {
    upDecoder->SetInput((Token*)0xbaddf00d, 2U);
  }

  upDecoder->SetInput(elementaryVideo ? elementaryVideo.get() : nullptr, 0U);
  if (TASK_EXEC_FAIL == upDecoder->Execute()) {
    return nullptr;
  }

  return (Surface *)upDecoder->GetOutput(0U);
};

uint32_t PyNvDecoder::Width() const {
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.width;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get width from demuxer instead");
  }
}

void PyNvDecoder::LastPacketData(PacketData &packetData) const {
  if (upDemuxer) {
    auto mp_buffer = (Buffer *)upDemuxer->GetOutput(3U);
    if (mp_buffer) {
      auto mp = mp_buffer->GetDataAs<PacketData>();
      packetData = *mp;
    }
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get packet data from demuxer instead");
  }
}

ColorSpace PyNvDecoder::GetColorSpace() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.color_space;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get color space from demuxer instead");
  }
}

ColorRange PyNvDecoder::GetColorRange() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.color_range;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get color range from demuxer instead");
  }
}

uint32_t PyNvDecoder::Height() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.height;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get height from demuxer instead");
  }
}

double PyNvDecoder::Framerate() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.frameRate;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get framerate from demuxer instead");
  }
}

double PyNvDecoder::AvgFramerate() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.avgFrameRate;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get avg framerate from demuxer instead");
  }
}

bool PyNvDecoder::IsVFR() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.is_vfr;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please check variable framerate flag from demuxer instead");
  }
}

double PyNvDecoder::Timebase() const {
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.timeBase;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get time base from demuxer instead");
  }
}

uint32_t PyNvDecoder::Framesize() const {
  if (upDemuxer) {
    auto pSurface = Surface::Make(GetPixelFormat(), Width(), Height(),
                                  CudaResMgr::Instance().GetCtx(gpuID));
    if (!pSurface) {
      throw runtime_error("Failed to determine video frame size.");
    }
    uint32_t size = pSurface->HostMemSize();
    delete pSurface;
    return size;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get frame size from demuxer instead");
  }
}

uint32_t PyNvDecoder::Numframes() const {
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.num_frames;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get num_frames from demuxer instead");
  }
}

Pixel_Format PyNvDecoder::GetPixelFormat() const { return format; }

struct DecodeContext {
  std::shared_ptr<Surface> pSurface;
  py::array_t<uint8_t> *pSei;
  py::array_t<uint8_t> *pPacket;
  SeekContext seek_ctx;
  PacketData pkt_data;
  bool usePacket;

  DecodeContext(py::array_t<uint8_t> *sei, py::array_t<uint8_t> *packet,
                SeekContext const &ctx)
      : pSurface(nullptr), pSei(sei), pPacket(packet), seek_ctx(ctx),
        usePacket(true) {}

  DecodeContext(py::array_t<uint8_t> *sei, py::array_t<uint8_t> *packet)
      : pSurface(nullptr), pSei(sei), pPacket(packet), seek_ctx(),
        usePacket(true) {}

  DecodeContext(py::array_t<uint8_t> *sei, SeekContext const &ctx)
      : pSurface(nullptr), pSei(sei), pPacket(nullptr), seek_ctx(ctx),
        usePacket(false) {}

  DecodeContext(py::array_t<uint8_t> *sei)
      : pSurface(nullptr), pSei(sei), pPacket(nullptr), seek_ctx(),
        usePacket(false) {}

  DecodeContext(SeekContext const &ctx)
      : pSurface(nullptr), pSei(nullptr), pPacket(nullptr), seek_ctx(ctx),
        usePacket(false) {}

  DecodeContext()
      : pSurface(nullptr), pSei(nullptr), pPacket(nullptr), seek_ctx(),
        usePacket(false) {}
};

bool PyNvDecoder::DecodeSurface(struct DecodeContext &ctx) {
  bool loop_end = false;
  // If we feed decoder with Annex.B from outside we can't seek;
  bool const use_seek = ctx.seek_ctx.use_seek && !ctx.usePacket;
  bool dec_error = false, dmx_error = false;

  Surface *pRawSurf = nullptr;

  // Check seek params & flush decoder if we need to seek;
  if (use_seek) {
    MuxingParams params;
    upDemuxer->GetParams(params);

    if (ctx.seek_ctx.mode != PREV_KEY_FRAME) {
      throw runtime_error("Decoder can only seek to closest previous key frame");
    }

    // Flush decoder without setting eos flag;
    Surface *p_surf = nullptr;
    do {
      try {
        auto const no_eos = true;
        p_surf = getDecodedSurfaceFromPacket(nullptr, no_eos);
      } catch (decoder_error &dec_exc) {
        dec_error = true;
        cerr << dec_exc.what() << endl;
      } catch (cuvid_parser_error &cvd_exc) {
        dmx_error = true;
        cerr << cvd_exc.what() << endl;
      }
    } while (p_surf && !p_surf->Empty());
    upDecoder->ClearOutputs();

    // Set number of decoded frames to zero before the loop;
    ctx.seek_ctx.num_frames_decoded = 0U;
  }

  /* Decode frames in loop if seek was done.
   * Otherwise will return after 1st iteration. */
  do {
    try {
      pRawSurf = ctx.usePacket
                     // In this case we get packet data from demuxer;
                     ? getDecodedSurfaceFromPacket(ctx.pPacket)
                     // In that case we will get packet data later from decoder;
                     : getDecodedSurface(upDecoder.get(), upDemuxer.get(),
                                         ctx.seek_ctx, ctx.pSei != nullptr);
    } catch (decoder_error &dec_exc) {
      dec_error = true;
      cerr << dec_exc.what() << endl;
    } catch (cuvid_parser_error &cvd_exc) {
      dmx_error = true;
      cerr << cvd_exc.what() << endl;
    }

    // Increase the counter;
    ctx.seek_ctx.num_frames_decoded++;

    /* Get timestamp from decoder.
     * However, this doesn't contain anything beside pts. */
    auto pktDataBuf = (Buffer *)upDecoder->GetOutput(1U);
    if (pktDataBuf) {
      ctx.pkt_data = *pktDataBuf->GetDataAs<PacketData>();
    }

    /* Check if seek loop is done.
     * Assuming video file with constant FPS. */
    if(use_seek) {
      int64_t const seek_pts =
          ctx.seek_ctx.seek_frame * ctx.seek_ctx.out_frame_duration;
      loop_end = (ctx.pkt_data.pts >= seek_pts);
    } else {
      loop_end = true;
    }

    if (dmx_error) {
      cerr << "Cuvid parser exception happened." << endl;
      throw CuvidParserException();
    }

    if (dec_error && upDemuxer) {
      time_point<system_clock> then = system_clock::now();

      MuxingParams params;
      upDemuxer->GetParams(params);

      upDecoder.reset(NvdecDecodeFrame::Make(
          CudaResMgr::Instance().GetStream(gpuID),
          CudaResMgr::Instance().GetCtx(gpuID), params.videoContext.codec,
          poolFrameSize, params.videoContext.width, params.videoContext.height,
          format));

      time_point<system_clock> now = system_clock::now();
      auto duration = duration_cast<milliseconds>(now - then).count();
      cerr << "HW decoder reset time: " << duration << " milliseconds" << endl;

      throw HwResetException();
    } else if (dec_error) {
      cerr << "HW exception happened. Please reset class instance" << endl;
      throw HwResetException();
    }

    if (ctx.pSei) {
      auto seiBuffer = (Buffer *)upDemuxer->GetOutput(2U);
      if (seiBuffer) {
        ctx.pSei->resize({seiBuffer->GetRawMemSize()}, false);
        memcpy(ctx.pSei->mutable_data(), seiBuffer->GetRawMemPtr(),
               seiBuffer->GetRawMemSize());
      } else {
        ctx.pSei->resize({0}, false);
      }
    }

  } while (use_seek && !loop_end);

  if (pRawSurf) {
    ctx.pSurface = shared_ptr<Surface>(pRawSurf->Clone());
    return true;
  } else {
    return false;
  }
}

shared_ptr<Surface>
PyNvDecoder::DecodeSingleSurface(py::array_t<uint8_t> &sei) {
    SeekContext seek_ctx;
    return DecodeSingleSurface(sei, seek_ctx);
}

std::shared_ptr<Surface>
PyNvDecoder::DecodeSingleSurface(py::array_t<uint8_t> &sei,
                                 PacketData &pkt_data) {
  DecodeContext ctx(&sei);
  if (DecodeSurface(ctx)) {
    pkt_data = ctx.pkt_data;
    return ctx.pSurface;
  }
  else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

shared_ptr<Surface>
PyNvDecoder::DecodeSingleSurface(py::array_t<uint8_t> &sei,
                                 SeekContext &seek_ctx,
                                 PacketData &pkt_data) {
  DecodeContext ctx(&sei, seek_ctx);
  if (DecodeSurface(ctx)) {
    seek_ctx = ctx.seek_ctx;
    pkt_data = ctx.pkt_data;
    return ctx.pSurface;
  }
  else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

shared_ptr<Surface>
PyNvDecoder::DecodeSingleSurface(py::array_t<uint8_t> &sei,
                                 SeekContext &seek_ctx) {
  DecodeContext ctx(&sei, seek_ctx);
  if (DecodeSurface(ctx)) {
    seek_ctx = ctx.seek_ctx;
    return ctx.pSurface;
  }
  else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

shared_ptr<Surface> 
PyNvDecoder::DecodeSingleSurface() {
  SeekContext seek_ctx;
  return DecodeSingleSurface(seek_ctx);
}

shared_ptr<Surface> 
PyNvDecoder::DecodeSingleSurface(PacketData &pkt_data) {
  DecodeContext ctx;
  if (DecodeSurface(ctx)) {
    pkt_data = ctx.pkt_data;
    return ctx.pSurface;
  } else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

shared_ptr<Surface> 
PyNvDecoder::DecodeSingleSurface(SeekContext &seek_ctx) {
  DecodeContext ctx(seek_ctx);
  if (DecodeSurface(ctx)) {
    seek_ctx = ctx.seek_ctx;
    return ctx.pSurface;
  } else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

shared_ptr<Surface>
PyNvDecoder::DecodeSingleSurface(SeekContext &seek_ctx,
                                 PacketData &pkt_data) {
  DecodeContext ctx(seek_ctx);
  if (DecodeSurface(ctx)) {
    seek_ctx = ctx.seek_ctx;
    pkt_data = ctx.pkt_data;
    return ctx.pSurface;
  } else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

shared_ptr<Surface>
PyNvDecoder::DecodeSurfaceFromPacket(py::array_t<uint8_t> &sei,
                                     py::array_t<uint8_t> &packet) {
  DecodeContext ctx(&sei, &packet);
  if (DecodeSurface(ctx)) {
    return ctx.pSurface;
  } else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

shared_ptr<Surface>
PyNvDecoder::DecodeSurfaceFromPacket(py::array_t<uint8_t> &packet) {
  DecodeContext ctx(nullptr, &packet);
  if (DecodeSurface(ctx)) {
    return ctx.pSurface;
  } else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

shared_ptr<Surface> PyNvDecoder::FlushSingleSurface() {
  DecodeContext ctx(nullptr, nullptr);
  if (DecodeSurface(ctx)) {
    return ctx.pSurface;
  } else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

bool PyNvDecoder::DecodeSingleFrame(py::array_t<uint8_t> &frame,
                                    py::array_t<uint8_t> &sei) {
  SeekContext seek_ctx;
  return DecodeSingleFrame(frame, sei, seek_ctx);
}

bool PyNvDecoder::DecodeSingleFrame(py::array_t<uint8_t> &frame,
                                    py::array_t<uint8_t> &sei,
                                    PacketData &pkt_data) {
  auto spRawSufrace = DecodeSingleSurface(sei, pkt_data);
  if (spRawSufrace->Empty()) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }

  return upDownloader->DownloadSingleSurface(spRawSufrace, frame);
}

bool PyNvDecoder::DecodeSingleFrame(py::array_t<uint8_t> &frame,
                                    py::array_t<uint8_t> &sei,
                                    SeekContext &seek_ctx) {
  auto spRawSufrace = DecodeSingleSurface(sei, seek_ctx);
  if (spRawSufrace->Empty()) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }

  return upDownloader->DownloadSingleSurface(spRawSufrace, frame);
}

bool PyNvDecoder::DecodeSingleFrame(py::array_t<uint8_t> &frame,
                                    py::array_t<uint8_t> &sei,
                                    SeekContext &seek_ctx,
                                    PacketData &pkt_data) {
  auto spRawSufrace = DecodeSingleSurface(sei, seek_ctx, pkt_data);
  if (spRawSufrace->Empty()) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }

  return upDownloader->DownloadSingleSurface(spRawSufrace, frame);
}

bool PyNvDecoder::FlushSingleFrame(py::array_t<uint8_t> &frame) {
  auto spRawSufrace = FlushSingleSurface();
  if (spRawSufrace->Empty()) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }

  return upDownloader->DownloadSingleSurface(spRawSufrace, frame);
}

bool PyNvDecoder::DecodeSingleFrame(py::array_t<uint8_t> &frame) {
  SeekContext seek_ctx;
  return DecodeSingleFrame(frame, seek_ctx);
}

bool PyNvDecoder::DecodeSingleFrame(py::array_t<uint8_t> &frame,
                                    PacketData &pkt_data) {
  auto spRawSufrace = DecodeSingleSurface(pkt_data);
  if (spRawSufrace->Empty()) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }

  return upDownloader->DownloadSingleSurface(spRawSufrace, frame);
}

bool PyNvDecoder::DecodeSingleFrame(py::array_t<uint8_t> &frame,
                                    SeekContext &seek_ctx) {
  auto spRawSufrace = DecodeSingleSurface(seek_ctx);
  if (spRawSufrace->Empty()) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }

  return upDownloader->DownloadSingleSurface(spRawSufrace, frame);
}

bool PyNvDecoder::DecodeSingleFrame(py::array_t<uint8_t> &frame,
                                    SeekContext &seek_ctx,
                                    PacketData &pkt_data) {
  auto spRawSufrace = DecodeSingleSurface(seek_ctx, pkt_data);
  if (spRawSufrace->Empty()) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }

  return upDownloader->DownloadSingleSurface(spRawSufrace, frame);
}

bool PyNvDecoder::DecodeFrameFromPacket(py::array_t<uint8_t> &frame,
                                        py::array_t<uint8_t> &packet,
                                        py::array_t<uint8_t> &sei) {
  auto spRawSufrace = DecodeSurfaceFromPacket(sei, packet);
  if (spRawSufrace->Empty()) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }

  return upDownloader->DownloadSingleSurface(spRawSufrace, frame);
}

bool PyNvDecoder::DecodeFrameFromPacket(py::array_t<uint8_t> &frame,
                                        py::array_t<uint8_t> &packet) {
  auto spRawSufrace = DecodeSurfaceFromPacket(packet);
  if (spRawSufrace->Empty()) {
    return false;
  }

  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }

  return upDownloader->DownloadSingleSurface(spRawSufrace, frame);
}

uint32_t PyNvEncoder::Width() const { return encWidth; }

uint32_t PyNvEncoder::Height() const { return encHeight; }

Pixel_Format PyNvEncoder::GetPixelFormat() const { return eFormat; }

bool PyNvEncoder::Reconfigure(const map<string, string> &encodeOptions,
                              bool force_idr, bool reset_enc, bool verbose) {

  if (upEncoder) {
    NvEncoderClInterface cli_interface(encodeOptions);
    return upEncoder->Reconfigure(cli_interface, force_idr, reset_enc, verbose);
  }

  return true;
}

PyNvEncoder::PyNvEncoder(const map<string, string> &encodeOptions,
                         int gpuID, Pixel_Format format, bool verbose)
    : PyNvEncoder(encodeOptions, CudaResMgr::Instance().GetCtx(gpuID),
                  CudaResMgr::Instance().GetStream(gpuID), format, verbose)
{}

PyNvEncoder::PyNvEncoder(const map<string, string> &encodeOptions,
                         CUcontext ctx, CUstream str, Pixel_Format format, 
                         bool verbose)
    : upEncoder(nullptr), uploader(nullptr), options(encodeOptions),
      verbose_ctor(verbose), eFormat(format) {

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

  auto it = options.find("s");
  if (it != options.end()) {
    ParseResolution(it->second, encWidth, encHeight);
  } else {
    throw invalid_argument("No resolution given");
  }

  // Parse pixel format;
  string fmt_string;
  switch (eFormat) {
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

  it = options.find("fmt");
  if (it != options.end()) {
    it->second = fmt_string;
  } else {
    options["fmt"] = fmt_string;
  }

  cuda_ctx = ctx;
  cuda_str = str;

  /* Don't initialize uploader & encoder here, just prepare config params;
   */
  Reconfigure(options, false, false, verbose);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t> &packet,
                                const py::array_t<uint8_t> &messageSEI,
                                bool sync, bool append) {
  EncodeContext ctx(rawSurface, &packet, &messageSEI, sync, append);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t> &packet,
                                const py::array_t<uint8_t> &messageSEI,
                                bool sync) {
  EncodeContext ctx(rawSurface, &packet, &messageSEI, sync, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t> &packet, bool sync) {
  EncodeContext ctx(rawSurface, &packet, nullptr, sync, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t> &packet,
                                const py::array_t<uint8_t> &messageSEI) {
  EncodeContext ctx(rawSurface, &packet, &messageSEI, false, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t> &packet) {
  EncodeContext ctx(rawSurface, &packet, nullptr, false, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSingleSurface(EncodeContext &ctx) {
  shared_ptr<Buffer> spSEI = nullptr;
  if (ctx.pMessageSEI && ctx.pMessageSEI->size()) {
    spSEI = shared_ptr<Buffer>(
        Buffer::MakeOwnMem(ctx.pMessageSEI->size(), ctx.pMessageSEI->data()));
  }

  if (!upEncoder) {
    NvEncoderClInterface cli_interface(options);

    upEncoder.reset(NvencEncodeFrame::Make(
        cuda_str, cuda_ctx, cli_interface,
        NV12 == eFormat ? NV_ENC_BUFFER_FORMAT_NV12
                        : YUV444 == eFormat ? NV_ENC_BUFFER_FORMAT_YUV444
                                            : NV_ENC_BUFFER_FORMAT_UNDEFINED,
        encWidth, encHeight, verbose_ctor));
  }

  upEncoder->ClearInputs();

  if (ctx.rawSurface) {
    upEncoder->SetInput(ctx.rawSurface.get(), 0U);
  } else {
    /* Flush encoder this way;
     */
    upEncoder->SetInput(nullptr, 0U);
  }

  if (ctx.sync) {
    /* Set 2nd input to any non-zero value
     * to signal sync encode;
     */
    upEncoder->SetInput((Token *)0xdeadbeef, 1U);
  }

  if (ctx.pMessageSEI && ctx.pMessageSEI->size()) {
    /* Set 3rd input in case we have SEI message;
     */
    upEncoder->SetInput(spSEI.get(), 2U);
  }

  if (TASK_EXEC_FAIL == upEncoder->Execute()) {
    throw runtime_error("Error while encoding frame");
  }

  auto encodedFrame = (Buffer *)upEncoder->GetOutput(0U);
  if (encodedFrame) {
    if (ctx.append) {
      auto old_size = ctx.pPacket->size();
      ctx.pPacket->resize({old_size + encodedFrame->GetRawMemSize()}, false);
      memcpy(ctx.pPacket->mutable_data() + old_size,
             encodedFrame->GetRawMemPtr(), encodedFrame->GetRawMemSize());
    } else {
      ctx.pPacket->resize({encodedFrame->GetRawMemSize()}, false);
      memcpy(ctx.pPacket->mutable_data(), encodedFrame->GetRawMemPtr(),
             encodedFrame->GetRawMemSize());
    }
    return true;
  }

  return false;
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                              py::array_t<uint8_t> &packet) {
  if (!uploader) {
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                              py::array_t<uint8_t> &packet,
                              const py::array_t<uint8_t> &messageSEI) {
  if (!uploader) {
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet,
                       messageSEI);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                              py::array_t<uint8_t> &packet,
                              const py::array_t<uint8_t> &messageSEI,
                              bool sync) {
  if (!uploader) {
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet,
                       messageSEI, sync);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                              py::array_t<uint8_t> &packet, bool sync) {
  if (!uploader) {
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet, sync);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                              py::array_t<uint8_t> &packet,
                              const py::array_t<uint8_t> &messageSEI, bool sync,
                              bool append) {
  if (!uploader) {
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet,
                       messageSEI, sync, append);
}

bool PyNvEncoder::FlushSinglePacket(py::array_t<uint8_t> &packet) {
  /* Keep feeding encoder with null input until it returns zero-size
   * surface; */
  shared_ptr<Surface> spRawSurface = nullptr;
  const py::array_t<uint8_t> *messageSEI = nullptr;
  auto const is_sync = true;
  auto const is_append = false;
  EncodeContext ctx(spRawSurface, &packet, messageSEI, is_sync, is_append);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::Flush(py::array_t<uint8_t> &packets) {
  uint32_t num_packets = 0U;
  do {
    if (!FlushSinglePacket(packets)) {
      break;
    }
    num_packets++;
  } while (true);
  return (num_packets > 0U);
}

auto CopySurfaceStrCtx = [](shared_ptr<Surface> self, shared_ptr<Surface> other,
                            CUcontext cudaCtx, CUstream cudaStream)
{
  CudaCtxPush ctxPush(cudaCtx);

  for (auto plane = 0U; plane < self->NumPlanes(); plane++) {
    auto srcPlanePtr = self->PlanePtr(plane);
    auto dstPlanePtr = other->PlanePtr(plane);

    if (!srcPlanePtr || !dstPlanePtr) {
      break;
    }

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = srcPlanePtr;
    m.dstDevice = dstPlanePtr;
    m.srcPitch = self->Pitch(plane);
    m.dstPitch = other->Pitch(plane);
    m.Height = self->Height(plane);
    m.WidthInBytes = self->WidthInBytes(plane);

    ThrowOnCudaError(cuMemcpy2DAsync(&m, cudaStream), __LINE__);
  }

  ThrowOnCudaError(cuStreamSynchronize(cudaStream), __LINE__);
};

auto CopySurface = [](shared_ptr<Surface> self, shared_ptr<Surface> other,
                      int gpuID)
{
  auto ctx = CudaResMgr::Instance().GetCtx(gpuID);
  auto str = CudaResMgr::Instance().GetStream(gpuID);

  return CopySurfaceStrCtx(self, other, ctx, str);
};

PYBIND11_MODULE(PyNvCodec, m)
{
  m.doc() = "Python bindings for Nvidia-accelerated video processing";

  PYBIND11_NUMPY_DTYPE_EX(MotionVector, source, "source", w, "w", h, "h", src_x,
                          "src_x", src_y, "src_y", dst_x, "dst_x", dst_y,
                          "dst_y", motion_x, "motion_x", motion_y, "motion_y",
                          motion_scale, "motion_scale");

  py::class_<MotionVector>(m, "MotionVector");
  
  py::register_exception<HwResetException>(m, "HwResetException");

  py::register_exception<CuvidParserException>(m, "CuvidParserException");

  py::enum_<Pixel_Format>(m, "PixelFormat")
      .value("Y", Pixel_Format::Y)
      .value("RGB", Pixel_Format::RGB)
      .value("NV12", Pixel_Format::NV12)
      .value("YUV420", Pixel_Format::YUV420)
      .value("RGB_PLANAR", Pixel_Format::RGB_PLANAR)
      .value("BGR", Pixel_Format::BGR)
      .value("YCBCR", Pixel_Format::YCBCR)
      .value("YUV444", Pixel_Format::YUV444)
      .value("UNDEFINED", Pixel_Format::UNDEFINED)
      .export_values();

    py::enum_<ColorSpace>(m, "ColorSpace")
      .value("BT_601", ColorSpace::BT_601)
      .value("BT_709", ColorSpace::BT_709)
      .value("UNSPEC", ColorSpace::UNSPEC)
      .export_values();

    py::enum_<ColorRange>(m, "ColorRange")
        .value("MPEG", ColorRange::MPEG)
        .value("JPEG", ColorRange::JPEG)
        .value("UDEF", ColorRange::UDEF)
        .export_values();

  py::enum_<cudaVideoCodec>(m, "CudaVideoCodec")
      .value("H264", cudaVideoCodec::cudaVideoCodec_H264)
      .value("HEVC", cudaVideoCodec::cudaVideoCodec_HEVC)
      .value("VP9", cudaVideoCodec::cudaVideoCodec_VP9)
      .export_values();

  py::enum_<SeekMode>(m, "SeekMode")
      .value("EXACT_FRAME", SeekMode::EXACT_FRAME)
      .value("PREV_KEY_FRAME", SeekMode::PREV_KEY_FRAME)
      .export_values();

  py::class_<SeekContext, shared_ptr<SeekContext>>(m, "SeekContext")
      .def(py::init<int64_t>(), py::arg("seek_frame"))
      .def(py::init<int64_t, SeekMode>(), py::arg("seek_frame"), py::arg("mode"))
      .def_readwrite("seek_frame", &SeekContext::seek_frame)
      .def_readwrite("mode", &SeekContext::mode)
      .def_readwrite("out_frame_pts", &SeekContext::out_frame_pts)
      .def_readonly("num_frames_decoded", &SeekContext::num_frames_decoded);

  py::class_<PacketData, shared_ptr<PacketData>>(m, "PacketData")
      .def(py::init<>())
      .def_readwrite("pts", &PacketData::pts)
      .def_readwrite("dts", &PacketData::dts)
      .def_readwrite("pos", &PacketData::pos)
      .def_readwrite("poc", &PacketData::poc)
      .def_readwrite("duration", &PacketData::duration);

    py::class_<ColorspaceConversionContext,
             shared_ptr<ColorspaceConversionContext>>(
      m, "ColorspaceConversionContext")
      .def(py::init<>())
      .def(py::init<ColorSpace, ColorRange>(), py::arg("color_space"),
           py::arg("color_range"))
      .def_readwrite("color_space", &ColorspaceConversionContext::color_space)
      .def_readwrite("color_range", &ColorspaceConversionContext::color_range);

    py::class_<SurfacePlane, shared_ptr<SurfacePlane>>(m, "SurfacePlane")
        .def("Width", &SurfacePlane::Width)
        .def("Height", &SurfacePlane::Height)
        .def("Pitch", &SurfacePlane::Pitch)
        .def("GpuMem", &SurfacePlane::GpuMem)
        .def("ElemSize", &SurfacePlane::ElemSize)
        .def("HostFrameSize", &SurfacePlane::GetHostMemSize)
        .def("Import",
             [](shared_ptr<SurfacePlane> self, CUdeviceptr src, uint32_t src_pitch,
                int gpuID)
             {
               self->Import(src, src_pitch, CudaResMgr::Instance().GetCtx(gpuID),
                            CudaResMgr::Instance().GetStream(gpuID));
             })
        .def("Import",
             [](shared_ptr<SurfacePlane> self, CUdeviceptr src, uint32_t src_pitch,
                size_t  ctx, size_t  str)
             {
               self->Import(src, src_pitch, (CUcontext)ctx, (CUstream)str);
             })
        .def("Export",
             [](shared_ptr<SurfacePlane> self, CUdeviceptr dst, uint32_t dst_pitch,
                int gpuID) {
               self->Export(dst, dst_pitch, CudaResMgr::Instance().GetCtx(gpuID),
                            CudaResMgr::Instance().GetStream(gpuID));
             })
        .def("Export",
             [](shared_ptr<SurfacePlane> self, CUdeviceptr dst, uint32_t dst_pitch,
                size_t  ctx, size_t  str) {
               self->Export(dst, dst_pitch, (CUcontext)ctx, (CUstream)str);
             });

    py::class_<Surface, shared_ptr<Surface>>(m, "Surface")
        .def("Width", &Surface::Width, py::arg("planeNumber") = 0U)
        .def("Height", &Surface::Height, py::arg("planeNumber") = 0U)
        .def("Pitch", &Surface::Pitch, py::arg("planeNumber") = 0U)
        .def("Format", &Surface::PixelFormat)
        .def("Empty", &Surface::Empty)
        .def("NumPlanes", &Surface::NumPlanes)
        .def("HostSize", &Surface::HostMemSize)
        .def_static(
            "Make",
            [](Pixel_Format format, uint32_t newWidth, uint32_t newHeight,
               int gpuID)
            {
              auto pNewSurf = shared_ptr<Surface>(
                  Surface::Make(format, newWidth, newHeight,
                                CudaResMgr::Instance().GetCtx(gpuID)));
              return pNewSurf;
            },
            py::return_value_policy::take_ownership)
        .def_static(
            "Make",
            [](Pixel_Format format, uint32_t newWidth, uint32_t newHeight,
               size_t ctx)
            {
              auto pNewSurf = shared_ptr<Surface>(
                  Surface::Make(format, newWidth, newHeight, (CUcontext)ctx));
              return pNewSurf;
            },
            py::return_value_policy::take_ownership)
        .def(
            "PlanePtr",
            [](shared_ptr<Surface> self, int planeNumber)
            {
              auto pPlane = self->GetSurfacePlane(planeNumber);
              return make_shared<SurfacePlane>(*pPlane);
            },
            // Integral part of Surface, only reference it;
            py::arg("planeNumber") = 0U, py::return_value_policy::reference)
        .def("CopyFrom",
             [](shared_ptr<Surface> self, shared_ptr<Surface> other, int gpuID)
             {
               if (self->PixelFormat() != other->PixelFormat())
               {
                 throw runtime_error("Surfaces have different pixel formats");
               }

               if (self->Width() != other->Width() ||
                   self->Height() != other->Height())
               {
                 throw runtime_error("Surfaces have different size");
               }

               CopySurface(self, other, gpuID);
             })
        .def("CopyFrom",
             [](shared_ptr<Surface> self, shared_ptr<Surface> other, size_t ctx,
                size_t str)
             {
               if (self->PixelFormat() != other->PixelFormat())
               {
                 throw runtime_error("Surfaces have different pixel formats");
               }

               if (self->Width() != other->Width() ||
                   self->Height() != other->Height())
               {
                 throw runtime_error("Surfaces have different size");
               }

               CopySurfaceStrCtx(self, other, (CUcontext)ctx, (CUstream)str);
             })
        .def(
            "Clone",
            [](shared_ptr<Surface> self, int gpuID)
            {
              auto pNewSurf = shared_ptr<Surface>(Surface::Make(
                  self->PixelFormat(), self->Width(), self->Height(),
                  CudaResMgr::Instance().GetCtx(gpuID)));

              CopySurface(self, pNewSurf, gpuID);
              return pNewSurf;
            },
            py::return_value_policy::take_ownership,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "Clone",
            [](shared_ptr<Surface> self, size_t ctx,
                size_t str)
            {
              auto pNewSurf = shared_ptr<Surface>(Surface::Make(
                  self->PixelFormat(), self->Width(), self->Height(),
                  (CUcontext)ctx));

              CopySurfaceStrCtx(self, pNewSurf, (CUcontext)ctx, (CUstream)str);
              return pNewSurf;
            },
            py::return_value_policy::take_ownership,
            py::call_guard<py::gil_scoped_release>());

    py::class_<PyNvEncoder>(m, "PyNvEncoder")
        .def(py::init<const map<string, string> &, int, Pixel_Format, bool>(),
             py::arg("settings"), py::arg("gpu_id"), py::arg("format") = NV12,
             py::arg("verbose") = false)
        .def(py::init<const map<string, string> &, size_t , size_t ,
                      Pixel_Format, bool>(),
             py::arg("settings"), py::arg("cuda_context"),
             py::arg("cuda_stream"), py::arg("format") = NV12,
             py::arg("verbose") = false)
        .def("Reconfigure", &PyNvEncoder::Reconfigure, py::arg("settings"),
             py::arg("force_idr") = false, py::arg("reset_encoder") = false,
             py::arg("verbose") = false)
        .def("Width", &PyNvEncoder::Width)
        .def("Height", &PyNvEncoder::Height)
        .def("Format", &PyNvEncoder::GetPixelFormat)
        .def("EncodeSingleSurface",
             py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &,
                               const py::array_t<uint8_t> &, bool, bool>(
                 &PyNvEncoder::EncodeSurface),
             py::arg("surface"), py::arg("packet"), py::arg("sei"),
             py::arg("sync"), py::arg("append"),
             py::call_guard<py::gil_scoped_release>())
        .def("EncodeSingleSurface",
             py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &,
                               const py::array_t<uint8_t> &, bool>(
                 &PyNvEncoder::EncodeSurface),
             py::arg("surface"), py::arg("packet"), py::arg("sei"),
             py::arg("sync"),
             py::call_guard<py::gil_scoped_release>())
        .def("EncodeSingleSurface",
             py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &, bool>(
                 &PyNvEncoder::EncodeSurface),
             py::arg("surface"), py::arg("packet"), py::arg("sync"),
             py::call_guard<py::gil_scoped_release>())
        .def("EncodeSingleSurface",
             py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &,
                               const py::array_t<uint8_t> &>(
                 &PyNvEncoder::EncodeSurface),
             py::arg("surface"), py::arg("packet"), py::arg("sei"),
             py::call_guard<py::gil_scoped_release>())
        .def("EncodeSingleSurface",
             py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &>(
                 &PyNvEncoder::EncodeSurface),
             py::arg("surface"), py::arg("packet"),
             py::call_guard<py::gil_scoped_release>())
        .def("EncodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                               const py::array_t<uint8_t> &, bool, bool>(
                 &PyNvEncoder::EncodeFrame),
             py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"),
             py::arg("append"),
             py::call_guard<py::gil_scoped_release>())
        .def("EncodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                               const py::array_t<uint8_t> &, bool>(
                 &PyNvEncoder::EncodeFrame),
             py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"),
             py::call_guard<py::gil_scoped_release>())
        .def("EncodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                               bool>(&PyNvEncoder::EncodeFrame),
             py::arg("frame"), py::arg("packet"), py::arg("sync"),
             py::call_guard<py::gil_scoped_release>())
        .def("EncodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                               const py::array_t<uint8_t> &>(
                 &PyNvEncoder::EncodeFrame),
             py::arg("frame"), py::arg("packet"), py::arg("sei"),
             py::call_guard<py::gil_scoped_release>())
        .def("EncodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
                 &PyNvEncoder::EncodeFrame),
             py::arg("frame"), py::arg("packet"),
             py::call_guard<py::gil_scoped_release>())
        .def("Flush", &PyNvEncoder::Flush, py::arg("packets"),
             py::call_guard<py::gil_scoped_release>())
        .def("FlushSinglePacket", &PyNvEncoder::FlushSinglePacket,
             py::arg("packets"),
             py::call_guard<py::gil_scoped_release>());

    py::class_<PyFfmpegDecoder>(m, "PyFfmpegDecoder")
        .def(py::init<const string &, const map<string, string> &>())
        .def("DecodeSingleFrame", &PyFfmpegDecoder::DecodeSingleFrame)
        .def("GetMotionVectors", &PyFfmpegDecoder::GetMotionVectors,
             py::return_value_policy::move);

    py::class_<PyFFmpegDemuxer>(m, "PyFFmpegDemuxer")
        .def(py::init<const string &>())
        .def(py::init<const string &, const map<string, string> &>())
        .def("DemuxSinglePacket", &PyFFmpegDemuxer::DemuxSinglePacket)
        .def("Width", &PyFFmpegDemuxer::Width)
        .def("Height", &PyFFmpegDemuxer::Height)
        .def("Format", &PyFFmpegDemuxer::Format)
        .def("Framerate", &PyFFmpegDemuxer::Framerate)
        .def("AvgFramerate", &PyFFmpegDemuxer::AvgFramerate)
        .def("IsVFR", &PyFFmpegDemuxer::IsVFR)
        .def("Timebase", &PyFFmpegDemuxer::Timebase)
        .def("Numframes", &PyFFmpegDemuxer::Numframes)
        .def("Codec", &PyFFmpegDemuxer::Codec)
        .def("LastPacketData", &PyFFmpegDemuxer::GetLastPacketData)
        .def("Seek", &PyFFmpegDemuxer::Seek)
        .def("ColorSpace", &PyFFmpegDemuxer::GetColorSpace)
        .def("ColorRange", &PyFFmpegDemuxer::GetColorRange);

    py::class_<PyNvDecoder>(m, "PyNvDecoder")
        .def(py::init<uint32_t, uint32_t, Pixel_Format, cudaVideoCodec,
                      uint32_t>())
        .def(py::init<const string &, int, const map<string, string> &>())
        .def(py::init<const string &, int>())
        .def(py::init<uint32_t, uint32_t, Pixel_Format, cudaVideoCodec,
                      size_t, size_t>())
        .def(py::init<const string &, size_t , size_t ,
                      const map<string, string> &>())
        .def(py::init<const string &, size_t , size_t >())
        .def("Width", &PyNvDecoder::Width)
        .def("Height", &PyNvDecoder::Height)
        .def("ColorSpace", &PyNvDecoder::GetColorSpace)
        .def("ColorRange", &PyNvDecoder::GetColorRange)
        .def("LastPacketData", &PyNvDecoder::LastPacketData)
        .def("Framerate", &PyNvDecoder::Framerate)
        .def("AvgFramerate", &PyNvDecoder::AvgFramerate)
        .def("IsVFR", &PyNvDecoder::IsVFR)
        .def("Timebase", &PyNvDecoder::Timebase)
        .def("Framesize", &PyNvDecoder::Framesize)
        .def("Numframes", &PyNvDecoder::Numframes)
        .def("Format", &PyNvDecoder::GetPixelFormat)
        .def("DecodeSingleSurface",
             py::overload_cast<PacketData &>(
                 &PyNvDecoder::DecodeSingleSurface),
             py::arg("packet_data"),
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleSurface",
             py::overload_cast<py::array_t<uint8_t> &>(
                 &PyNvDecoder::DecodeSingleSurface),
             py::arg("sei"), py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleSurface",
             py::overload_cast<py::array_t<uint8_t> &, PacketData &>(
                 &PyNvDecoder::DecodeSingleSurface),
             py::arg("sei"), py::arg("packet_data"),
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleSurface",
             py::overload_cast<py::array_t<uint8_t> &, SeekContext &>(
                 &PyNvDecoder::DecodeSingleSurface),
             py::arg("sei"), py::arg("seek_context"),
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleSurface",
             py::overload_cast<py::array_t<uint8_t> &, SeekContext &, PacketData &>(
                 &PyNvDecoder::DecodeSingleSurface),
             py::arg("sei"), py::arg("seek_context"),
             py::arg("packet_data"),
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleSurface",
             py::overload_cast<>(&PyNvDecoder::DecodeSingleSurface),
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleSurface",
             py::overload_cast<SeekContext &>(&PyNvDecoder::DecodeSingleSurface),
             py::arg("seek_context"), py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleSurface",
             py::overload_cast<SeekContext &, PacketData &>(
                 &PyNvDecoder::DecodeSingleSurface),
             py::arg("seek_context"), py::arg("packet_data"),
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSurfaceFromPacket",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
                 &PyNvDecoder::DecodeSurfaceFromPacket),
             py::arg("packet"), py::arg("sei"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSurfaceFromPacket",
             py::overload_cast<py::array_t<uint8_t> &>(
                 &PyNvDecoder::DecodeSurfaceFromPacket),
             py::arg("packet"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
                 &PyNvDecoder::DecodeSingleFrame),
             py::arg("frame"), py::arg("sei"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                               PacketData &>(
                 &PyNvDecoder::DecodeSingleFrame),
             py::arg("frame"), py::arg("sei"), py::arg("packet_data"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                               SeekContext &>(
                 &PyNvDecoder::DecodeSingleFrame),
             py::arg("frame"), py::arg("sei"), py::arg("seek_context"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                               SeekContext &, PacketData &>(
                 &PyNvDecoder::DecodeSingleFrame),
             py::arg("frame"), py::arg("sei"), py::arg("seek_context"),
             py::arg("packet_data"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &>(
                 &PyNvDecoder::DecodeSingleFrame),
             py::arg("frame"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, PacketData &>(
                 &PyNvDecoder::DecodeSingleFrame),
             py::arg("frame"), py::arg("packet_data"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, SeekContext &>(
                 &PyNvDecoder::DecodeSingleFrame),
             py::arg("frame"), py::arg("seek_context"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeSingleFrame",
             py::overload_cast<py::array_t<uint8_t> &, SeekContext &,
                               PacketData &>(
                 &PyNvDecoder::DecodeSingleFrame),
             py::arg("frame"), py::arg("seek_context"), py::arg("packet_data"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeFrameFromPacket",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                               py::array_t<uint8_t> &>(
                 &PyNvDecoder::DecodeFrameFromPacket),
             py::arg("frame"), py::arg("packet"), py::arg("sei"),
             py::call_guard<py::gil_scoped_release>())
        .def("DecodeFrameFromPacket",
             py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
                 &PyNvDecoder::DecodeFrameFromPacket),
             py::arg("frame"), py::arg("packet"),
             py::call_guard<py::gil_scoped_release>())
        .def("Numframes", &PyNvDecoder::Numframes)
        .def("FlushSingleSurface", &PyNvDecoder::FlushSingleSurface,
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>())
        .def("FlushSingleFrame", &PyNvDecoder::FlushSingleFrame,
             py::arg("frame"),
             py::call_guard<py::gil_scoped_release>());

    py::class_<PyFrameUploader>(m, "PyFrameUploader")
        .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
        .def(py::init<uint32_t, uint32_t, Pixel_Format, size_t , size_t >())
        .def("Format", &PyFrameUploader::GetFormat)
        .def("UploadSingleFrame", &PyFrameUploader::UploadSingleFrame,
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>());

    py::class_<PySurfaceDownloader>(m, "PySurfaceDownloader")
        .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
        .def(py::init<uint32_t, uint32_t, Pixel_Format, size_t , size_t >())
        .def("Format", &PySurfaceDownloader::GetFormat)
        .def("DownloadSingleSurface",
             &PySurfaceDownloader::DownloadSingleSurface,
             py::call_guard<py::gil_scoped_release>());

    py::class_<PySurfaceConverter>(m, "PySurfaceConverter")
        .def(py::init<uint32_t, uint32_t, Pixel_Format, Pixel_Format, uint32_t>())
        .def(py::init<uint32_t, uint32_t, Pixel_Format, Pixel_Format, size_t , size_t >())        
        .def("Format", &PySurfaceConverter::GetFormat)
        .def("Execute", &PySurfaceConverter::Execute,
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>());

    py::class_<PySurfaceResizer>(m, "PySurfaceResizer")
        .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
        .def(py::init<uint32_t, uint32_t, Pixel_Format, size_t , size_t >())
        .def("Format", &PySurfaceResizer::GetFormat)
        .def("Execute", &PySurfaceResizer::Execute,
             py::return_value_policy::take_ownership,
             py::call_guard<py::gil_scoped_release>());

    m.def("GetNumGpus", &CudaResMgr::GetNumGpus);
}
