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

namespace VPF
{
class pythonbuf : public std::streambuf
{
private:
  char d_buffer[1024U];
  py::bytes read_buffer;
  off_type rbuf_end;

  py::object pywrite;
  py::object pyflush;
  py::object pyread;

  int_type underflow()
  {
    read_buffer = pyread(1024U);
    char* read_buffer_data;
    py::ssize_t py_n_read;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(read_buffer.ptr(), &read_buffer_data,
                                          &py_n_read) == -1) {
      setg(0, 0, 0);
      throw std::invalid_argument("The method 'read' of the Python file object "
                                  "did not return a string.");
    }
    off_type n_read = (off_type)py_n_read;
    rbuf_end += n_read;
    setg(read_buffer_data, read_buffer_data, read_buffer_data + n_read);
    if (n_read == 0) {
      return traits_type::eof();
    }
    return traits_type::to_int_type(read_buffer_data[0]);
  }

  std::streamsize showmanyc()
  {
    int_type status = underflow();
    if (traits_type::eof() == status) {
      return -1;
    }
    return egptr() - gptr();
  }

  int overflow(int c)
  {
    if (!traits_type::eq_int_type(c, traits_type::eof())) {
      *pptr() = traits_type::to_char_type(c);
      pbump(1);
    }
    return sync() == 0 ? traits_type::not_eof(c) : traits_type::eof();
  }

  int sync()
  {
    if (pbase() != pptr()) {
      py::str line(pbase(), static_cast<size_t>(pptr() - pbase()));

      pywrite(line);
      pyflush();

      setp(pbase(), epptr());
    }
    return 0;
  }

public:
  pythonbuf(py::object py_stream)
      : pywrite(py_stream.attr("write")), pyflush(py_stream.attr("flush")),
        pyread(py_stream.attr("read"))
  {
    setp(d_buffer, d_buffer + sizeof(d_buffer) - 1);
  }

  ~pythonbuf() { sync(); }
};
} // namespace VPF

static void write_to_byte_io(py::object fileHandle, std::string& line)
{
  VPF::pythonbuf buf(fileHandle);
  std::ostream stream(&buf);
  stream << line << endl;
}

static std::string read_from_byte_io(py::object fileHandle)
{
  VPF::pythonbuf buf(fileHandle);
  std::istream stream(&buf);

  std::string line;
  std::getline(stream, line);

  return line;
}

PyFFmpegDemuxer::PyFFmpegDemuxer(py::object fileHandle)
{
  VPF::pythonbuf buf(fileHandle);
  std::istream i_str(&buf);

  map<string, string> ffmpeg_options;
  vector<const char*> options;
  for (auto& pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(DemuxFrame::Make(i_str, options.data(), options.size()));
}

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

bool PyFFmpegDemuxer::DemuxSinglePacket(py::array_t<uint8_t>& packet)
{
  Buffer* elementaryVideo = nullptr;
  do {
    if (TASK_EXEC_FAIL == upDemuxer->Execute()) {
      upDemuxer->ClearInputs();
      return false;
    }
    elementaryVideo = (Buffer*)upDemuxer->GetOutput(0U);
  } while (!elementaryVideo);

  packet.resize({elementaryVideo->GetRawMemSize()}, false);
  memcpy(packet.mutable_data(), elementaryVideo->GetDataAs<void>(),
         elementaryVideo->GetRawMemSize());

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
  py::class_<PyFFmpegDemuxer>(m, "PyFFmpegDemuxer")
      .def(py::init<py::object>())
      .def(py::init<const string&>())
      .def(py::init<const string&, const map<string, string>&>())
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

  m.def("write_to_byte_io", &write_to_byte_io);
  m.def("read_from_byte_io", &read_from_byte_io);
}