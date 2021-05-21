/*
 * Copyright 2021 NVIDIA Corporation
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

Surface *
PyNvDecoder::getDecodedSurfaceFromPacket(py::array_t<uint8_t> *pPacket) {
  Surface *surface = nullptr;
  unique_ptr<Buffer> elementaryVideo = nullptr;

  if (pPacket && pPacket->size()) {
    elementaryVideo = unique_ptr<Buffer>(
        Buffer::MakeOwnMem(pPacket->size(), pPacket->data()));
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

    // Flush decoder;
    Surface *p_surf = nullptr;
    do {
      try {
        p_surf = getDecodedSurfaceFromPacket(nullptr);
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