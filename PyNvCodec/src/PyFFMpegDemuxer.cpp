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
#include <streambuf>

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PyFFmpegDemuxer::PyFFmpegDemuxer(const string& pathToFile)
    : PyFFmpegDemuxer(pathToFile, map<string, string>())
{
}

PyFFmpegDemuxer::PyFFmpegDemuxer(const string& pathToFile,
                                 const map<string, string>& ffmpeg_options)
{
  vector<const char*> options;
  for (auto& pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));
}

bool PyFFmpegDemuxer::DemuxSinglePacket(py::array_t<uint8_t>& packet,
                                        py::array_t<uint8_t>* sei)
{
  upDemuxer->ClearInputs();
  upDemuxer->ClearOutputs();

  Buffer* elementaryVideo = nullptr;
  do {
    if (nullptr != sei) {
      upDemuxer->SetInput((Token*)0xdeadbeef, 0U);
    }

    if (TASK_EXEC_FAIL == upDemuxer->Execute()) {
      upDemuxer->ClearInputs();
      return false;
    }
    elementaryVideo = (Buffer*)upDemuxer->GetOutput(0U);
  } while (!elementaryVideo);

  packet.resize({elementaryVideo->GetRawMemSize()}, false);
  memcpy(packet.mutable_data(), elementaryVideo->GetDataAs<void>(),
         elementaryVideo->GetRawMemSize());

  auto seiBuffer = (Buffer*)upDemuxer->GetOutput(2U);
  if (seiBuffer && sei) {
    sei->resize({seiBuffer->GetRawMemSize()}, false);
    memcpy(sei->mutable_data(), seiBuffer->GetDataAs<void>(),
           seiBuffer->GetRawMemSize());
  }

  upDemuxer->ClearInputs();
  return true;
}

void PyFFmpegDemuxer::GetLastPacketData(PacketData& pkt_data)
{
  auto pkt_data_buf = (Buffer*)upDemuxer->GetOutput(3U);
  if (pkt_data_buf) {
    auto pkt_data_ptr = pkt_data_buf->GetDataAs<PacketData>();
    pkt_data = *pkt_data_ptr;
  }
}

uint32_t PyFFmpegDemuxer::Width() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.width;
}

ColorSpace PyFFmpegDemuxer::GetColorSpace() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.color_space;
};

ColorRange PyFFmpegDemuxer::GetColorRange() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.color_range;
};

uint32_t PyFFmpegDemuxer::Height() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.height;
}

Pixel_Format PyFFmpegDemuxer::Format() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.format;
}

cudaVideoCodec PyFFmpegDemuxer::Codec() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.codec;
}

double PyFFmpegDemuxer::Framerate() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.frameRate;
}

double PyFFmpegDemuxer::AvgFramerate() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.avgFrameRate;
}

bool PyFFmpegDemuxer::IsVFR() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.is_vfr;
}

double PyFFmpegDemuxer::Timebase() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.timeBase;
}

uint32_t PyFFmpegDemuxer::Numframes() const
{
  MuxingParams params;
  upDemuxer->GetParams(params);
  return params.videoContext.num_frames;
}

void PyFFmpegDemuxer::GetDecoderInitParams(CUVIDEOFORMAT* params) const
{
  upDemuxer->GetCuVideoFormat(params);
}

bool PyFFmpegDemuxer::Seek(SeekContext& ctx, py::array_t<uint8_t>& packet)
{
  Buffer* elementaryVideo = nullptr;
  auto pSeekCtxBuf = shared_ptr<Buffer>(Buffer::MakeOwnMem(sizeof(ctx), &ctx));
  do {
    upDemuxer->SetInput((Token*)pSeekCtxBuf.get(), 1U);
    if (TASK_EXEC_FAIL == upDemuxer->Execute()) {
      upDemuxer->ClearInputs();
      return false;
    }
    elementaryVideo = (Buffer*)upDemuxer->GetOutput(0U);
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

void Init_PyFFMpegDemuxer(py::module& m)
{
  py::class_<PyFFmpegDemuxer, shared_ptr<PyFFmpegDemuxer>>(m, "PyFFmpegDemuxer")
      .def(py::init<const string&, const map<string, string>&>())
      .def(py::init<const string&>())
      .def(
          "DemuxSinglePacket",
          [](shared_ptr<PyFFmpegDemuxer> self, py::array_t<uint8_t>& packet) {
            return self->DemuxSinglePacket(packet, nullptr);
          },
          py::arg("packet"), py::call_guard<py::gil_scoped_release>())
      .def(
          "DemuxSinglePacket",
          [](shared_ptr<PyFFmpegDemuxer> self, py::array_t<uint8_t>& packet,
             py::array_t<uint8_t>& sei) {
            return self->DemuxSinglePacket(packet, &sei);
          },
          py::arg("packet"), py::arg("sei"),
          py::call_guard<py::gil_scoped_release>())
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
      .def("ColorRange", &PyFFmpegDemuxer::GetColorRange)
      .def("GetDecoderInitParams", &PyFFmpegDemuxer::GetDecoderInitParams);

  py::class_<CUVIDEOFORMAT, shared_ptr<CUVIDEOFORMAT>>(m, "DecoderInitParams")
      .def(py::init<>());

  m.attr("NO_PTS") = py::int_(AV_NOPTS_VALUE);
}