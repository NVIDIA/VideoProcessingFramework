/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Kognia Sports Intelligence
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

#include "PyNvCodec.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PyNvDecoder::PyNvDecoder(const string& pathToFile, int gpuOrdinal)
    : PyNvDecoder(pathToFile, gpuOrdinal, map<string, string>())
{
}

PyNvDecoder::PyNvDecoder(const string& pathToFile, CUcontext ctx, CUstream str)
    : PyNvDecoder(pathToFile, ctx, str, map<string, string>())
{
}

PyNvDecoder::PyNvDecoder(const string& pathToFile, int gpuOrdinal,
                         const map<string, string>& ffmpeg_options)
{
  if (gpuOrdinal < 0 || gpuOrdinal >= CudaResMgr::Instance().GetNumGpus()) {
    gpuOrdinal = 0U;
  }
  gpuID = gpuOrdinal;
  cout << "Decoding on GPU " << gpuID << endl;

  vector<const char*> options;
  for (auto& pair : ffmpeg_options) {
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

PyNvDecoder::PyNvDecoder(const string& pathToFile, CUcontext ctx, CUstream str,
                         const map<string, string>& ffmpeg_options)
{
  vector<const char*> options;
  for (auto& pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));

  MuxingParams params;
  upDemuxer->GetParams(params);
  format = params.videoContext.format;

  upDecoder.reset(NvdecDecodeFrame::Make(
      str, ctx, params.videoContext.codec, poolFrameSize,
      params.videoContext.width, params.videoContext.height, format));
}

PyNvDecoder::PyNvDecoder(uint32_t width, uint32_t height,
                         Pixel_Format new_format, cudaVideoCodec codec,
                         uint32_t gpuOrdinal)
    : format(new_format)
{
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
  upDecoder.reset(NvdecDecodeFrame::Make(str, ctx, codec, poolFrameSize, width,
                                         height, format));
}

Buffer* PyNvDecoder::getElementaryVideo(DemuxFrame* demuxer,
                                        SeekContext* seek_ctx, bool needSEI)
{
  Buffer* elementaryVideo = nullptr;
  Buffer* pktData = nullptr;
  shared_ptr<Buffer> pSeekCtxBuf = nullptr;

  do {
    // Set 1st demuxer input to any non-zero value if we need SEI;
    if (needSEI) {
      demuxer->SetInput((Token*)0xdeadbeef, 0U);
    }

    // Set 2nd demuxer input to seek context if we need to seek;
    if (seek_ctx->use_seek) {
      pSeekCtxBuf =
          shared_ptr<Buffer>(Buffer::MakeOwnMem(sizeof(SeekContext), seek_ctx));
      demuxer->SetInput((Token*)pSeekCtxBuf.get(), 1U);
    }
    if (TASK_EXEC_FAIL == demuxer->Execute()) {
      return nullptr;
    }
    elementaryVideo = (Buffer*)demuxer->GetOutput(0U);

    /* Clear inputs and set down seek flag or we will seek
     * for one and the same frame multiple times. */
    seek_ctx->use_seek = false;
    demuxer->ClearInputs();
  } while (!elementaryVideo);

  auto pktDataBuf = (Buffer*)demuxer->GetOutput(3U);
  if (pktDataBuf) {
    auto pPktData = pktDataBuf->GetDataAs<PacketData>();
    seek_ctx->out_frame_pts = pPktData->pts;
    seek_ctx->out_frame_duration = pPktData->duration;
  }

  return elementaryVideo;
};

Surface* PyNvDecoder::getDecodedSurface(NvdecDecodeFrame* decoder,
                                        DemuxFrame* demuxer,
                                        SeekContext* seek_ctx, bool needSEI)
{
  decoder->ClearInputs();
  decoder->ClearOutputs();

  Surface* surface = nullptr;
  do {
    auto elementaryVideo = getElementaryVideo(demuxer, seek_ctx, needSEI);
    auto pktData = (Buffer*)demuxer->GetOutput(3U);

    decoder->SetInput(elementaryVideo, 0U);
    decoder->SetInput(pktData, 1U);
    if (TASK_EXEC_FAIL == decoder->Execute()) {
      break;
    }

    surface = (Surface*)decoder->GetOutput(0U);
  } while (!surface);

  return surface;
};

Surface*
PyNvDecoder::getDecodedSurfaceFromPacket(const py::array_t<uint8_t>* pPacket,
                                         const PacketData* p_packet_data,
                                         bool no_eos)
{
  upDecoder->ClearInputs();
  upDecoder->ClearOutputs();

  Surface* surface = nullptr;
  unique_ptr<Buffer> packetData = nullptr;
  unique_ptr<Buffer> elementaryVideo = nullptr;

  if (pPacket && pPacket->size()) {
    elementaryVideo = unique_ptr<Buffer>(
        Buffer::MakeOwnMem(pPacket->size(), pPacket->data()));
  }

  if (no_eos) {
    upDecoder->SetInput((Token*)0xbaddf00d, 2U);
  }

  if (p_packet_data) {
    packetData = unique_ptr<Buffer>(
        Buffer::MakeOwnMem(sizeof(PacketData), p_packet_data));
    upDecoder->SetInput(packetData.get(), 1U);
  }

  upDecoder->SetInput(elementaryVideo ? elementaryVideo.get() : nullptr, 0U);
  if (TASK_EXEC_FAIL == upDecoder->Execute()) {
    return nullptr;
  }

  return (Surface*)upDecoder->GetOutput(0U);
};

uint32_t PyNvDecoder::Width() const
{
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.width;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get width from demuxer instead");
  }
}

void PyNvDecoder::LastPacketData(PacketData& packetData) const
{
  if (upDemuxer) {
    auto mp_buffer = (Buffer*)upDemuxer->GetOutput(3U);
    if (mp_buffer) {
      auto mp = mp_buffer->GetDataAs<PacketData>();
      packetData = *mp;
    }
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get packet data from demuxer instead");
  }
}

ColorSpace PyNvDecoder::GetColorSpace() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.color_space;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get color space from demuxer instead");
  }
}

ColorRange PyNvDecoder::GetColorRange() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.color_range;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get color range from demuxer instead");
  }
}

uint32_t PyNvDecoder::Height() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.height;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get height from demuxer instead");
  }
}

double PyNvDecoder::Framerate() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.frameRate;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get framerate from demuxer instead");
  }
}

double PyNvDecoder::AvgFramerate() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.avgFrameRate;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get avg framerate from demuxer instead");
  }
}

bool PyNvDecoder::IsVFR() const
{
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.is_vfr;
  } else {
    throw runtime_error(
        "Decoder was created without built-in demuxer support. "
        "Please check variable framerate flag from demuxer instead");
  }
}

double PyNvDecoder::Timebase() const
{
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.timeBase;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get time base from demuxer instead");
  }
}

uint32_t PyNvDecoder::Framesize() const
{
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

uint32_t PyNvDecoder::Numframes() const
{
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

class DecodeContext
{
private:
  shared_ptr<Surface> pSurface;

  py::array_t<uint8_t>* pSei;
  py::array_t<uint8_t>* pPacket;

  PacketData* pInPktData;
  PacketData* pOutPktData;

  SeekContext* pSeekCtx;

  bool flush;

public:
  DecodeContext(py::array_t<uint8_t>* sei, py::array_t<uint8_t>* packet,
                PacketData* in_pkt_data, PacketData* out_pkt_data,
                SeekContext* seek_ctx, bool is_flush = false)
  {
    if (seek_ctx && packet) {
      throw runtime_error("Can't use seek in standalone mode.");
    }

    pSurface = nullptr;
    pSei = sei;
    pPacket = packet;
    pInPktData = in_pkt_data;
    pOutPktData = out_pkt_data;
    pSeekCtx = seek_ctx;
    flush = is_flush;
  }

  bool IsSeek() const { return (nullptr != pSeekCtx) && (nullptr == pPacket); }

  bool IsStandalone() const { return (nullptr != pPacket); }

  bool IsFlush() const { return flush; }

  bool HasSEI() const { return nullptr != pSei; }

  bool HasOutPktData() const { return nullptr != pOutPktData; }

  bool HasInPktData() const { return nullptr != pInPktData; }

  const py::array_t<uint8_t>* GetPacket() const { return pPacket; }

  const PacketData* GetInPacketData() const { return pInPktData; }

  const SeekContext* GetSeekContext() const { return pSeekCtx; }

  SeekContext* GetSeekContextMutable() { return pSeekCtx; }

  shared_ptr<Surface> GetSurfaceMutable() { return pSurface; }

  void SetOutPacketData(PacketData* out_pkt_data)
  {
    if (!out_pkt_data || !pOutPktData) {
      throw runtime_error("Invalid data pointer");
    }

    memcpy(pOutPktData, out_pkt_data, sizeof(PacketData));
  }

  void SetSei(Buffer* sei)
  {
    if (!pSei) {
      throw runtime_error("Invalid data pointer");
    }

    if (!sei) {
      pSei->resize({0}, false);
      return;
    }

    pSei->resize({sei->GetRawMemSize()}, false);
    memcpy(pSei->mutable_data(), sei->GetRawMemPtr(), sei->GetRawMemSize());
  }

  void SetCloneSurface(Surface* p_surface)
  {
    if (!p_surface) {
      throw runtime_error("Invalid data pointer");
    }
    pSurface = shared_ptr<Surface>(p_surface->Clone());
  }
};

bool PyNvDecoder::DecodeSurface(DecodeContext& ctx)
{
  bool loop_end = false;
  // If we feed decoder with Annex.B from outside we can't seek;
  bool const use_seek = ctx.IsSeek();
  bool dec_error = false, dmx_error = false;

  Surface* pRawSurf = nullptr;

  // Check seek params & flush decoder if we need to seek;
  if (use_seek) {
    MuxingParams params;
    upDemuxer->GetParams(params);

    if (PREV_KEY_FRAME != ctx.GetSeekContext()->mode) {
      throw runtime_error(
          "Decoder can only seek to closest previous key frame");
    }

    // Flush decoder without setting eos flag;
    Surface* p_surf = nullptr;
    do {
      try {
        auto const no_eos = true;
        p_surf = getDecodedSurfaceFromPacket(nullptr, nullptr, no_eos);
      } catch (decoder_error& dec_exc) {
        dec_error = true;
        cerr << dec_exc.what() << endl;
      } catch (cuvid_parser_error& cvd_exc) {
        dmx_error = true;
        cerr << cvd_exc.what() << endl;
      }
    } while (p_surf && !p_surf->Empty());
    upDecoder->ClearOutputs();

    // Set number of decoded frames to zero before the loop;
    ctx.GetSeekContextMutable()->num_frames_decoded = 0U;
  }

  /* Decode frames in loop if seek was done.
   * Otherwise will return after 1st iteration. */
  do {
    try {
      if (ctx.IsFlush()) {
        pRawSurf = getDecodedSurfaceFromPacket(nullptr, nullptr);
      } else if (ctx.IsStandalone()) {
        pRawSurf =
            getDecodedSurfaceFromPacket(ctx.GetPacket(), ctx.GetInPacketData());
      } else {
        pRawSurf = getDecodedSurface(upDecoder.get(), upDemuxer.get(),
                                     ctx.GetSeekContextMutable(), ctx.HasSEI());
      }

      if (!pRawSurf) {
        break;
      }
    } catch (decoder_error& dec_exc) {
      dec_error = true;
      cerr << dec_exc.what() << endl;
    } catch (cuvid_parser_error& cvd_exc) {
      dmx_error = true;
      cerr << cvd_exc.what() << endl;
    }

    // Increase the counter;
    if (use_seek)
      ctx.GetSeekContextMutable()->num_frames_decoded++;

    /* Get timestamp from decoder.
     * However, this doesn't contain anything beside pts. */
    auto pktDataBuf = (Buffer*)upDecoder->GetOutput(1U);
    if (pktDataBuf && ctx.HasOutPktData()) {
      ctx.SetOutPacketData(pktDataBuf->GetDataAs<PacketData>());
    }

    auto is_seek_done = [&](DecodeContext const& ctx, double time_base) {
      auto seek_ctx = ctx.GetSeekContext();
      if (!seek_ctx)
        throw runtime_error("No seek context.");

      auto in_pkt_data = ctx.GetInPacketData();
      if (!in_pkt_data)
        throw runtime_error("No input packet data.");

      int64_t seek_pts = 0;

      switch (seek_ctx->crit) {
      case BY_NUMBER:
        seek_pts = seek_ctx->seek_frame * seek_ctx->out_frame_duration;
        break;

      case BY_TIMESTAMP:
        seek_pts = seek_ctx->seek_frame / time_base;
        break;

      default:
        throw runtime_error("Invalid seek criteria.");
        break;
      }

      return (in_pkt_data->pts >= seek_pts);
    };

    /* Check if seek is done. */
    if (!use_seek) {
      loop_end = true;
    } else {
      MuxingParams params;
      upDemuxer->GetParams(params);
      loop_end = is_seek_done(ctx, params.videoContext.timeBase);
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

    if (ctx.HasSEI()) {
      auto seiBuffer = (Buffer*)upDemuxer->GetOutput(2U);
      ctx.SetSei(seiBuffer);
    }

  } while (use_seek && !loop_end);

  if (pRawSurf) {
    ctx.SetCloneSurface(pRawSurf);
    return true;
  } else {
    return false;
  }
}

auto make_empty_surface = [](Pixel_Format pixFmt) {
  auto pSurface = shared_ptr<Surface>(Surface::Make(pixFmt));
  return shared_ptr<Surface>(pSurface->Clone());
};

void PyNvDecoder::DownloaderLazyInit()
{
  if (!upDownloader) {
    uint32_t width, height, elem_size;
    upDecoder->GetDecodedFrameParams(width, height, elem_size);
    upDownloader.reset(new PySurfaceDownloader(width, height, format, gpuID));
  }
}

bool PyNvDecoder::DecodeFrame(class DecodeContext& ctx,
                              py::array_t<uint8_t>& frame)
{
  if (!DecodeSurface(ctx))
    return false;

  DownloaderLazyInit();
  return upDownloader->DownloadSingleSurface(ctx.GetSurfaceMutable(), frame);
}

void Init_PyNvDecoder(py::module& m)
{
  py::class_<PyNvDecoder, shared_ptr<PyNvDecoder>>(m, "PyNvDecoder")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, cudaVideoCodec,
                    uint32_t>())
      .def(py::init<const string&, int, const map<string, string>&>())
      .def(py::init<const string&, int>())
      .def(py::init<uint32_t, uint32_t, Pixel_Format, cudaVideoCodec, size_t,
                    size_t>())
      .def(
          py::init<const string&, size_t, size_t, const map<string, string>&>())
      .def(py::init<const string&, size_t, size_t>())
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
      .def("Numframes", &PyNvDecoder::Numframes)
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, nullptr, nullptr, &out_pkt_data, nullptr,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("packet_data"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& sei) {
            DecodeContext ctx(&sei, nullptr, nullptr, nullptr, nullptr, false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("sei"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& sei,
             PacketData& out_pkt_data) {
            DecodeContext ctx(&sei, nullptr, nullptr, &out_pkt_data, nullptr,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("sei"), py::arg("packet_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& sei,
             SeekContext& seek_ctx) {
            DecodeContext ctx(&sei, nullptr, nullptr, nullptr, &seek_ctx,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("sei"), py::arg("seek_context"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& sei,
             SeekContext& seek_ctx, PacketData& out_pkt_data) {
            DecodeContext ctx(&sei, nullptr, nullptr, &out_pkt_data, &seek_ctx,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("sei"), py::arg("seek_context"), py::arg("packet_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self) {
            DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, nullptr,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, SeekContext& seek_ctx) {
            DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, &seek_ctx,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("seek_context"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, SeekContext& seek_ctx,
             PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, nullptr, nullptr, &out_pkt_data,
                              &seek_ctx, false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("seek_context"), py::arg("packet_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& packet,
             py::array_t<uint8_t>& sei) {
            DecodeContext ctx(&sei, &packet, nullptr, nullptr, nullptr, false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("packet"), py::arg("sei"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, PacketData& in_pkt_data,
             py::array_t<uint8_t>& packet, py::array_t<uint8_t>& sei) {
            DecodeContext ctx(&sei, &packet, &in_pkt_data, nullptr, nullptr,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("enc_packet_data"), py::arg("packet"), py::arg("sei"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& packet) {
            DecodeContext ctx(nullptr, &packet, nullptr, nullptr, nullptr,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("packet"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, PacketData& in_packet_data,
             py::array_t<uint8_t>& packet) {
            DecodeContext ctx(nullptr, &packet, &in_packet_data, nullptr,
                              nullptr, false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("enc_packet_data"), py::arg("packet"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& packet,
             py::array_t<uint8_t>& sei, PacketData& out_pkt_data) {
            DecodeContext ctx(&sei, &packet, nullptr, &out_pkt_data, nullptr,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("packet"), py::arg("sei"), py::arg("packet_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, PacketData& in_packet_data,
             py::array_t<uint8_t>& packet, py::array_t<uint8_t>& sei,
             PacketData& out_pkt_data) {
            DecodeContext ctx(&sei, &packet, &in_packet_data, &out_pkt_data,
                              nullptr, false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("enc_packet_data"), py::arg("packet"), py::arg("sei"),
          py::arg("packet_data"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& packet,
             PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, &packet, nullptr, &out_pkt_data, nullptr,
                              false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("packet"), py::arg("packet_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, PacketData& in_pkt_data,
             py::array_t<uint8_t>& packet, PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, &packet, &in_pkt_data, &out_pkt_data,
                              nullptr, false);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("enc_packet_data"), py::arg("packet"), py::arg("packet_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "FlushSingleSurface",
          [](shared_ptr<PyNvDecoder> self) {
            DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, nullptr,
                              true);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "FlushSingleSurface",
          [](shared_ptr<PyNvDecoder> self, PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, nullptr, nullptr, &out_pkt_data, nullptr,
                              true);
            if (self->DecodeSurface(ctx))
              return ctx.GetSurfaceMutable();
            else
              return make_empty_surface(self->GetPixelFormat());
          },
          py::arg("packet_data"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             py::array_t<uint8_t>& sei) {
            DecodeContext ctx(&sei, nullptr, nullptr, nullptr, nullptr, false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("sei"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             py::array_t<uint8_t>& sei, PacketData& out_pkt_data) {
            DecodeContext ctx(&sei, nullptr, nullptr, &out_pkt_data, nullptr,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("sei"), py::arg("packet_data"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             py::array_t<uint8_t>& sei, SeekContext& seek_ctx) {
            DecodeContext ctx(&sei, nullptr, nullptr, nullptr, &seek_ctx,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("sei"), py::arg("seek_context"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             py::array_t<uint8_t>& sei, SeekContext& seek_ctx,
             PacketData& out_pkt_data) {
            DecodeContext ctx(&sei, nullptr, nullptr, &out_pkt_data, &seek_ctx,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("sei"), py::arg("seek_context"),
          py::arg("packet_data"), py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame) {
            DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, nullptr,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, nullptr, nullptr, &out_pkt_data, nullptr,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("packet_data"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             SeekContext& seek_ctx) {
            DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, &seek_ctx,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("seek_context"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             SeekContext& seek_ctx, PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, nullptr, nullptr, &out_pkt_data,
                              &seek_ctx, false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("seek_context"), py::arg("packet_data"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeFrameFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             py::array_t<uint8_t>& packet, py::array_t<uint8_t>& sei) {
            DecodeContext ctx(&sei, &packet, nullptr, nullptr, nullptr, false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("packet"), py::arg("sei"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeFrameFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             PacketData& in_pkt_data, py::array_t<uint8_t>& packet,
             py::array_t<uint8_t>& sei) {
            DecodeContext ctx(&sei, &packet, &in_pkt_data, nullptr, nullptr,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("enc_packet_data"), py::arg("packet"),
          py::arg("sei"), py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeFrameFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             py::array_t<uint8_t>& packet) {
            DecodeContext ctx(nullptr, &packet, nullptr, nullptr, nullptr,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("packet"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeFrameFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             PacketData& in_pkt_data, py::array_t<uint8_t>& packet) {
            DecodeContext ctx(nullptr, &packet, &in_pkt_data, nullptr, nullptr,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("enc_packet_data"), py::arg("packet"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeFrameFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             py::array_t<uint8_t>& packet, py::array_t<uint8_t>& sei,
             PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, &packet, nullptr, &out_pkt_data, nullptr,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("packet"), py::arg("sei"),
          py::arg("packet_data"), py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeFrameFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             PacketData& in_pkt_data, py::array_t<uint8_t>& packet,
             py::array_t<uint8_t>& sei, PacketData& out_pkt_data) {
            DecodeContext ctx(&sei, &packet, &in_pkt_data, &out_pkt_data,
                              nullptr, false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("enc_packet_data"), py::arg("packet"),
          py::arg("sei"), py::arg("packet_data"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeFrameFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             PacketData& in_pkt_data, py::array_t<uint8_t>& packet,
             PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, &packet, &in_pkt_data, &out_pkt_data,
                              nullptr, false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("enc_packet_data"), py::arg("packet"),
          py::arg("packet_data"), py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeFrameFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             py::array_t<uint8_t>& packet, PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, &packet, nullptr, &out_pkt_data, nullptr,
                              false);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("packet"), py::arg("packet_data"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "FlushSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame) {
            DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, nullptr,
                              true);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::call_guard<py::gil_scoped_release>())
      .def(
          "FlushSingleFrame",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& frame,
             PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, nullptr, nullptr, &out_pkt_data, nullptr,
                              true);
            return self->DecodeFrame(ctx, frame);
          },
          py::arg("frame"), py::arg("packet_data"),
          py::call_guard<py::gil_scoped_release>());
}