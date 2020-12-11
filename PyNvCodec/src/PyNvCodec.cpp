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
  CudaResMgr() {
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

public:
  static CudaResMgr &Instance() {
    static CudaResMgr instance;
    return instance;
  }

  CUcontext GetCtx(size_t idx) {
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

  CUstream GetStream(size_t idx) {
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

  /* Also a static function as we want to keep all the
   * CUDA stuff within one Python module;
   */
  ~CudaResMgr() {
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

  static size_t GetNumGpus() { return Instance().g_Contexts.size(); }

  vector<CUcontext> g_Contexts;
  vector<CUstream> g_Streams;
};

PyFrameUploader::PyFrameUploader(uint32_t width, uint32_t height,
                                 Pixel_Format format, uint32_t gpu_ID) {
  gpuID = gpu_ID;
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  uploader.reset(CudaUploadFrame::Make(CudaResMgr::Instance().GetStream(gpuID),
                                       CudaResMgr::Instance().GetCtx(gpuID),
                                       surfaceWidth, surfaceHeight,
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
  gpuID = gpu_ID;
  surfaceWidth = width;
  surfaceHeight = height;
  surfaceFormat = format;

  upDownloader.reset(
      CudaDownloadSurface::Make(CudaResMgr::Instance().GetStream(gpuID),
                                CudaResMgr::Instance().GetCtx(gpuID),
                                surfaceWidth, surfaceHeight, surfaceFormat));
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
    : gpuID(gpuID), outputFormat(outFormat) {
  upConverter.reset(ConvertSurface::Make(
      width, height, inFormat, outFormat, CudaResMgr::Instance().GetCtx(gpuID),
      CudaResMgr::Instance().GetStream(gpuID)));
}

shared_ptr<Surface> PySurfaceConverter::Execute(shared_ptr<Surface> surface) {
  if (!surface) {
    return shared_ptr<Surface>(Surface::Make(outputFormat));
  }

  upConverter->SetInput(surface.get(), 0U);
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
    : outputFormat(format), gpuID(gpuID) {
  upResizer.reset(ResizeSurface::Make(width, height, format,
                                      CudaResMgr::Instance().GetCtx(gpuID),
                                      CudaResMgr::Instance().GetStream(gpuID)));
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
      return false;
    }
    elementaryVideo = (Buffer *)upDemuxer->GetOutput(0U);
  } while (!elementaryVideo);

  packet.resize({elementaryVideo->GetRawMemSize()}, false);
  memcpy(packet.mutable_data(), elementaryVideo->GetDataAs<void>(),
         elementaryVideo->GetRawMemSize());

  return true;
}

uint32_t PyFFmpegDemuxer::Width() const {
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.width;
}

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

PyNvDecoder::PyNvDecoder(const string &pathToFile, int gpuOrdinal)
    : PyNvDecoder(pathToFile, gpuOrdinal, map<string, string>()) {}

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

Buffer *PyNvDecoder::getElementaryVideo(DemuxFrame *demuxer, bool needSEI) {
  Buffer *elementaryVideo = nullptr;
  do {
    if (needSEI) {
      demuxer->SetInput((Token*)0xdeadbeef, 0U);
    }
    if (TASK_EXEC_FAIL == demuxer->Execute()) {
      return nullptr;
    }
    elementaryVideo = (Buffer *)demuxer->GetOutput(0U);
  } while (!elementaryVideo);

  return elementaryVideo;
};

Surface *PyNvDecoder::getDecodedSurface(NvdecDecodeFrame *decoder,
                                        DemuxFrame *demuxer,
                                        bool &hw_decoder_failure,
                                        bool needSEI) {
  hw_decoder_failure = false;
  Surface *surface = nullptr;
  do {
    auto elementaryVideo = getElementaryVideo(demuxer, needSEI);

    decoder->SetInput(elementaryVideo, 0U);
    try {
      if (TASK_EXEC_FAIL == decoder->Execute()) {
        break;
      }
    } catch (exception &e) {
      cerr << "Exception thrown during decoding process: " << e.what() << endl;
      cerr << "HW decoder will be reset." << endl;
      hw_decoder_failure = true;
      break;
    }

    surface = (Surface *)decoder->GetOutput(0U);
  } while (!surface);

  return surface;
};

Surface *PyNvDecoder::getDecodedSurfaceFromPacket(py::array_t<uint8_t> *pPacket,
                                                  bool &hw_decoder_failure) {
  hw_decoder_failure = false;
  Surface *surface = nullptr;
  unique_ptr<Buffer> elementaryVideo = nullptr;

  if (pPacket && pPacket->size()) {
    elementaryVideo = unique_ptr<Buffer>(
        Buffer::MakeOwnMem(pPacket->size(), pPacket->data()));
  }

  upDecoder->SetInput(elementaryVideo ? elementaryVideo.get() : nullptr, 0U);
  try {
    if (TASK_EXEC_FAIL == upDecoder->Execute()) {
      return nullptr;
    }
  } catch (exception &e) {
    cerr << "Exception thrown during decoding process: " << e.what() << endl;
    cerr << "HW decoder will be reset." << endl;
    hw_decoder_failure = true;
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
  auto mp_buffer = (Buffer *)upDemuxer->GetOutput(1U);
  if (mp_buffer) {
    auto mp = mp_buffer->GetDataAs<MuxingParams>();
    packetData = mp->videoContext.packetData;
  }
}

uint32_t PyNvDecoder::Height() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.height;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get width from demuxer instead");
  }
}

double PyNvDecoder::Framerate() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.frameRate;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get width from demuxer instead");
  }
}

double PyNvDecoder::Timebase() const {
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.timeBase;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get width from demuxer instead");
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
                        "Please get width from demuxer instead");
  }
}

Pixel_Format PyNvDecoder::GetPixelFormat() const { return format; }

struct DecodeContext {
  std::shared_ptr<Surface> pSurface;
  py::array_t<uint8_t> *pSei;
  py::array_t<uint8_t> *pPacket;
  bool usePacket;

  DecodeContext(py::array_t<uint8_t> *sei, py::array_t<uint8_t> *packet)
      : pSurface(nullptr), pSei(sei), pPacket(packet), usePacket(true) {}

  DecodeContext(py::array_t<uint8_t> *sei)
      : pSurface(nullptr), pSei(sei), pPacket(nullptr), usePacket(false) {}

  DecodeContext()
      : pSurface(nullptr), pSei(nullptr), pPacket(nullptr), usePacket(false) {}
};

bool PyNvDecoder::DecodeSurface(struct DecodeContext &ctx) {
  bool hw_decoder_failure = false;

  auto pRawSurf =
      ctx.usePacket
          ? getDecodedSurfaceFromPacket(ctx.pPacket, hw_decoder_failure)
          : getDecodedSurface(upDecoder.get(), upDemuxer.get(),
                              hw_decoder_failure, ctx.pSei != nullptr);

  if (hw_decoder_failure && upDemuxer) {
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
  } else if (hw_decoder_failure) {
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

  if (pRawSurf) {
    ctx.pSurface = shared_ptr<Surface>(pRawSurf->Clone());
    return true;
  } else {
    return false;
  }
}

shared_ptr<Surface>
PyNvDecoder::DecodeSingleSurface(py::array_t<uint8_t> &sei) {
  DecodeContext ctx(&sei);
  if (DecodeSurface(ctx)) {
    return ctx.pSurface;
  } else {
    auto pixFmt = GetPixelFormat();
    auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
    return shared_ptr<Surface>(pSurface->Clone());
  }
}

shared_ptr<Surface> PyNvDecoder::DecodeSingleSurface() {
  DecodeContext ctx;
  if (DecodeSurface(ctx)) {
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
  auto spRawSufrace = DecodeSingleSurface(sei);
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
  auto spRawSufrace = DecodeSingleSurface();
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
                         int gpuOrdinal, Pixel_Format format, bool verbose)
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

  if (gpuOrdinal < 0 || gpuOrdinal >= CudaResMgr::Instance().GetNumGpus()) {
    gpuOrdinal = 0U;
  }
  gpuID = gpuOrdinal;
  cout << "Encoding on GPU " << gpuID << endl;

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
        CudaResMgr::Instance().GetStream(gpuID),
        CudaResMgr::Instance().GetCtx(gpuID), cli_interface,
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
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, gpuID));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                              py::array_t<uint8_t> &packet,
                              const py::array_t<uint8_t> &messageSEI) {
  if (!uploader) {
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, gpuID));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet,
                       messageSEI);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                              py::array_t<uint8_t> &packet,
                              const py::array_t<uint8_t> &messageSEI,
                              bool sync) {
  if (!uploader) {
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, gpuID));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet,
                       messageSEI, sync);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                              py::array_t<uint8_t> &packet, bool sync) {
  if (!uploader) {
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, gpuID));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet, sync);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t> &inRawFrame,
                              py::array_t<uint8_t> &packet,
                              const py::array_t<uint8_t> &messageSEI, bool sync,
                              bool append) {
  if (!uploader) {
    uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, gpuID));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet,
                       messageSEI, sync, append);
}

bool PyNvEncoder::Flush(py::array_t<uint8_t> &packets) {
  uint32_t num_packets = 0U;
  do {
    /* Keep feeding encoder with null input until it returns zero-size
     * surface; */
    EncodeContext ctx(nullptr, &packets, nullptr, true, true);
    auto success = EncodeSingleSurface(ctx);
    if (!success) {
      break;
    }
    num_packets++;
  } while (true);

  return (num_packets > 0U);
}

auto CopySurface = [](shared_ptr<Surface> self, shared_ptr<Surface> other,
                      int gpuID) {
  auto cudaCtx = CudaResMgr::Instance().GetCtx(gpuID);
  CUstream cudaStream = CudaResMgr::Instance().GetStream(gpuID);

  for (auto plane = 0U; plane < self->NumPlanes(); plane++) {
    auto srcPlanePtr = self->PlanePtr(plane);
    auto dstPlanePtr = other->PlanePtr(plane);

    if (!srcPlanePtr || !dstPlanePtr) {
      break;
    }

    CudaCtxPush ctxPush(cudaCtx);

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

PYBIND11_MODULE(PyNvCodec, m) {
  m.doc() = "Python bindings for Nvidia-accelerated video processing";

  PYBIND11_NUMPY_DTYPE_EX(MotionVector, source, "source", w, "w", h, "h", src_x,
                          "src_x", src_y, "src_y", dst_x, "dst_x", dst_y,
                          "dst_y", motion_x, "motion_x", motion_y, "motion_y",
                          motion_scale, "motion_scale");

  py::class_<MotionVector>(m, "MotionVector");

  py::register_exception<HwResetException>(m, "HwResetException");

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

  py::enum_<cudaVideoCodec>(m, "CudaVideoCodec")
      .value("H264", cudaVideoCodec::cudaVideoCodec_H264)
      .value("HEVC", cudaVideoCodec::cudaVideoCodec_HEVC)
      .export_values();

  py::class_<SurfacePlane, shared_ptr<SurfacePlane>>(m, "SurfacePlane")
      .def("Width", &SurfacePlane::Width)
      .def("Height", &SurfacePlane::Height)
      .def("Pitch", &SurfacePlane::Pitch)
      .def("GpuMem", &SurfacePlane::GpuMem)
      .def("ElemSize", &SurfacePlane::ElemSize)
      .def("HostFrameSize", &SurfacePlane::GetHostMemSize);

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
             int gpuID) {
            auto pNewSurf = shared_ptr<Surface>(
                Surface::Make(format, newWidth, newHeight,
                              CudaResMgr::Instance().GetCtx(gpuID)));
            return pNewSurf;
          },
          py::return_value_policy::take_ownership)
      .def(
          "PlanePtr",
          [](shared_ptr<Surface> self, int planeNumber) {
            auto pPlane = self->GetSurfacePlane(planeNumber);
            return make_shared<SurfacePlane>(*pPlane);
          },
          // Integral part of Surface, only reference it;
          py::arg("planeNumber") = 0U, py::return_value_policy::reference)
      .def("CopyFrom",
           [](shared_ptr<Surface> self, shared_ptr<Surface> other, int gpuID) {
             if (self->PixelFormat() != other->PixelFormat()) {
               throw runtime_error("Surfaces have different pixel formats");
             }

             if (self->Width() != other->Width() ||
                 self->Height() != other->Height()) {
               throw runtime_error("Surfaces have different size");
             }

             CopySurface(self, other, gpuID);
           })
      .def(
          "Clone",
          [](shared_ptr<Surface> self, int gpuID) {
            auto pNewSurf = shared_ptr<Surface>(Surface::Make(
                self->PixelFormat(), self->Width(), self->Height(),
                CudaResMgr::Instance().GetCtx(gpuID)));

            CopySurface(self, pNewSurf, gpuID);
            return pNewSurf;
          },
          py::return_value_policy::take_ownership);

  py::class_<PyNvEncoder>(m, "PyNvEncoder")
      .def(py::init<const map<string, string> &, int, Pixel_Format, bool>(),
           py::arg("settings"), py::arg("gpu_id"), py::arg("format") = NV12,
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
           py::arg("sync"), py::arg("append"))
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::arg("sync"))
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sync"))
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"))
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t> &>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"))
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &, bool, bool>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"),
           py::arg("append"))
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &, bool>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"))
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             bool>(&PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sync"))
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             const py::array_t<uint8_t> &>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"))
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"))
      .def("Flush", &PyNvEncoder::Flush, py::arg("packets"));

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
      .def("Codec", &PyFFmpegDemuxer::Codec);

  py::class_<PacketData>(m, "PacketData")
      .def(py::init<>())
      .def_readonly("pts", &PacketData::pts)
      .def_readonly("dts", &PacketData::dts)
      .def_readonly("pos", &PacketData::pos)
      .def_readonly("duration", &PacketData::duration);

  py::class_<PyNvDecoder>(m, "PyNvDecoder")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, cudaVideoCodec,
                    uint32_t>())
      .def(py::init<const string &, int, const map<string, string> &>())
      .def(py::init<const string &, int>())
      .def("Width", &PyNvDecoder::Width)
      .def("Height", &PyNvDecoder::Height)
      .def("LastPacketData", &PyNvDecoder::LastPacketData)
      .def("Framerate", &PyNvDecoder::Framerate)
      .def("Timebase", &PyNvDecoder::Timebase)
      .def("Framesize", &PyNvDecoder::Framesize)
      .def("Format", &PyNvDecoder::GetPixelFormat)
      .def("DecodeSingleSurface",
           py::overload_cast<py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSingleSurface),
           py::arg("sei"), py::return_value_policy::take_ownership)
      .def("DecodeSingleSurface",
           py::overload_cast<>(&PyNvDecoder::DecodeSingleSurface),
           py::return_value_policy::take_ownership)
      .def("DecodeSurfaceFromPacket",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSurfaceFromPacket),
           py::arg("packet"), py::arg("sei"))
      .def("DecodeSurfaceFromPacket",
           py::overload_cast<py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSurfaceFromPacket),
           py::arg("packet"))
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"), py::arg("sei"))
      .def("DecodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeSingleFrame),
           py::arg("frame"))
      .def("DecodeFrameFromPacket",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &,
                             py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeFrameFromPacket),
           py::arg("frame"), py::arg("packet"), py::arg("sei"))
      .def("DecodeFrameFromPacket",
           py::overload_cast<py::array_t<uint8_t> &, py::array_t<uint8_t> &>(
               &PyNvDecoder::DecodeFrameFromPacket),
           py::arg("frame"), py::arg("packet"))
      .def("FlushSingleSurface", &PyNvDecoder::FlushSingleSurface,
           py::return_value_policy::take_ownership)
      .def("FlushSingleFrame", &PyNvDecoder::FlushSingleFrame,
           py::arg("frame"));

  py::class_<PyFrameUploader>(m, "PyFrameUploader")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
      .def("Format", &PyFrameUploader::GetFormat)
      .def("UploadSingleFrame", &PyFrameUploader::UploadSingleFrame,
           py::return_value_policy::take_ownership);

  py::class_<PySurfaceDownloader>(m, "PySurfaceDownloader")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
      .def("Format", &PySurfaceDownloader::GetFormat)
      .def("DownloadSingleSurface",
           &PySurfaceDownloader::DownloadSingleSurface);

  py::class_<PySurfaceConverter>(m, "PySurfaceConverter")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, Pixel_Format, uint32_t>())
      .def("Format", &PySurfaceConverter::GetFormat)
      .def("Execute", &PySurfaceConverter::Execute,
           py::return_value_policy::take_ownership);

  py::class_<PySurfaceResizer>(m, "PySurfaceResizer")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>())
      .def("Format", &PySurfaceResizer::GetFormat)
      .def("Execute", &PySurfaceResizer::Execute,
           py::return_value_policy::take_ownership);

  m.def("GetNumGpus", &CudaResMgr::GetNumGpus);
}
