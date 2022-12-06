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
      upDemuxer->SetInput((Token*)0xdeadbeefull, 0U);
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
      .def(py::init<const string&, const map<string, string>&>(),
           py::arg("input"), py::arg("opts"),
           R"pbdoc(
        Constructor method.

        :param input: path to input file
        :param opts: AVDictionary options that will be passed to AVFormat context.
    )pbdoc")
      .def(py::init<const string&>(), py::arg("input"),
           R"pbdoc(
        Constructor method.

        :param input: path to input file
    )pbdoc")
      .def(
          "DemuxSinglePacket",
          [](shared_ptr<PyFFmpegDemuxer> self, py::array_t<uint8_t>& packet) {
            return self->DemuxSinglePacket(packet, nullptr);
          },
          py::arg("packet"), py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Extract single compressed video packet from input file.

        :param packet: encoded packet
        :return: True in case of success, False otherwise
    )pbdoc")
      .def(
          "DemuxSinglePacket",
          [](shared_ptr<PyFFmpegDemuxer> self, py::array_t<uint8_t>& packet,
             py::array_t<uint8_t>& sei) {
            return self->DemuxSinglePacket(packet, &sei);
          },
          py::arg("packet"), py::arg("sei"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Extract single compressed video packet and SEI data from input file.

        :param packet: encoded packet
        :param packet: SEI data
        :return: True in case of success, False otherwise
    )pbdoc")
      .def("Width", &PyFFmpegDemuxer::Width,
           R"pbdoc(
        Return encoded video stream width in pixels.
    )pbdoc")
      .def("Height", &PyFFmpegDemuxer::Height,
           R"pbdoc(
        Return encoded video stream height in pixels.
    )pbdoc")
      .def("Format", &PyFFmpegDemuxer::Format,
           R"pbdoc(
        Return encoded video stream pixel format.
    )pbdoc")
      .def("Framerate", &PyFFmpegDemuxer::Framerate,
           R"pbdoc(
        Return encoded video stream framerate.
    )pbdoc")
      .def("AvgFramerate", &PyFFmpegDemuxer::AvgFramerate,
           R"pbdoc(
        Return encoded video stream average framerate.
    )pbdoc")
      .def("IsVFR", &PyFFmpegDemuxer::IsVFR,
           R"pbdoc(
        Tell if video stream has variable frame rate.
        :return: True in case video stream has variable frame rate, False otherwise
    )pbdoc")
      .def("Timebase", &PyFFmpegDemuxer::Timebase,
           R"pbdoc(
        Return encoded video stream time base.
    )pbdoc")
      .def("Numframes", &PyFFmpegDemuxer::Numframes,
           R"pbdoc(
        Return number of video frames in encoded video stream.
        Please note that some video containers doesn't store this infomation.
    )pbdoc")
      .def("Codec", &PyFFmpegDemuxer::Codec,
           R"pbdoc(
        Return video codec used in encoded video stream.
    )pbdoc")
      .def("LastPacketData", &PyFFmpegDemuxer::GetLastPacketData,
           py::arg("pkt_data"),
           R"pbdoc(
        Get last demuxed packet data.
        :param pkt_data: packet data structure.
    )pbdoc")
      .def("Seek", &PyFFmpegDemuxer::Seek, py::arg("seek_ctx"), py::arg("pkt"),
           R"pbdoc(
        Perform seek operation.
        :param seek_ctx: seek context structure.
        :param pkt: compressed video packet.
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("ColorSpace", &PyFFmpegDemuxer::GetColorSpace,
           R"pbdoc(
        Get color space information stored in video stream.
        Please not that some video containers may not store this information.

        :return: color space information
    )pbdoc")
      .def("ColorRange", &PyFFmpegDemuxer::GetColorRange,
           R"pbdoc(
        Get color range information stored in video stream.
        Please not that some video containers may not store this information.

        :return: color range information
    )pbdoc");

  m.attr("NO_PTS") = py::int_(AV_NOPTS_VALUE);
}
