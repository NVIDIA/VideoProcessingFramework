/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Kognia Sports Intelligence
 * Copyright 2021 Videonetics Technology Private Limited
 * Copyright 2023 VisionLabs LLC
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

PyFfmpegDecoder::PyFfmpegDecoder(const string& pathToFile,
                                 const map<string, string>& ffmpeg_options,
                                 uint32_t gpuID)
{
  gpu_id = gpuID;
  NvDecoderClInterface cli_iface(ffmpeg_options);
  upDecoder.reset(FfmpegDecodeFrame::Make(pathToFile.c_str(), cli_iface));
}

bool PyFfmpegDecoder::DecodeSingleFrame(py::array_t<uint8_t>& frame)
{
  UpdateState();

  if (TASK_EXEC_SUCCESS == upDecoder->Execute()) {
    auto pRawFrame = (Buffer*)upDecoder->GetOutput(0U);
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

std::shared_ptr<Surface> PyFfmpegDecoder::DecodeSingleSurface()
{
  // Don't call UploadSingleFrame(DecodeSingleFrame()).
  // On some platforms py::array_t ctor causes segfault within python land.
  UploaderLazyInit();
  UpdateState();

  if (TASK_EXEC_SUCCESS == upDecoder->Execute()) {
    auto pRawFrame = (Buffer*)upDecoder->GetOutput(0U);
    if (pRawFrame) {
      return upUploader->UploadBuffer(pRawFrame);
    }
  }

  return shared_ptr<Surface>(Surface::Make(PixelFormat()));
}

void* PyFfmpegDecoder::GetSideData(AVFrameSideDataType data_type,
                                   size_t& raw_size)
{
  if (TASK_EXEC_SUCCESS == upDecoder->GetSideData(data_type)) {
    auto pSideData = (Buffer*)upDecoder->GetOutput(1U);
    if (pSideData) {
      raw_size = pSideData->GetRawMemSize();
      return pSideData->GetDataAs<void>();
    }
  }
  return nullptr;
}

void PyFfmpegDecoder::UpdateState()
{
  last_h = Height();
  last_w = Width();
}

bool PyFfmpegDecoder::IsResolutionChanged()
{
  if (last_h != Height()) {
    return true;
  }

  if (last_w != Width()) {
    return true;
  }

  return false;
}

void PyFfmpegDecoder::UploaderLazyInit()
{
  if (IsResolutionChanged() && upUploader) {
    upUploader.reset();
    upUploader = nullptr;
  }

  if (!upUploader) {
    upUploader.reset(
        new PyFrameUploader(Width(), Height(), PixelFormat(), gpu_id));
  }
}

py::array_t<MotionVector> PyFfmpegDecoder::GetMotionVectors()
{
  size_t size = 0U;
  auto ptr = (AVMotionVector*)GetSideData(AV_FRAME_DATA_MOTION_VECTORS, size);
  size /= sizeof(*ptr);

  if (ptr && size) {
    py::array_t<MotionVector> mv(static_cast<int64_t>(size));
    auto req = mv.request(true);
    auto mvc = static_cast<MotionVector*>(req.ptr);

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

    return std::move(mv);
  }

  return std::move(py::array_t<MotionVector>(0));
}

uint32_t PyFfmpegDecoder::Width() const
{
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.width;
};

uint32_t PyFfmpegDecoder::Height() const
{
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.height;
};

double PyFfmpegDecoder::Framerate() const
{
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.frameRate;
};

ColorSpace PyFfmpegDecoder::Color_Space() const
{
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.color_space;
};

ColorRange PyFfmpegDecoder::Color_Range() const
{
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.color_range;
};

cudaVideoCodec PyFfmpegDecoder::Codec() const
{
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.codec;
};

Pixel_Format PyFfmpegDecoder::PixelFormat() const
{
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.format;
};

void Init_PyFFMpegDecoder(py::module& m)
{
  py::class_<PyFfmpegDecoder>(m, "PyFfmpegDecoder")
      .def(py::init<const string&, const map<string, string>&, uint32_t>(),
           py::arg("input"), py::arg("opts"), py::arg("gpu_id") = 0,
           R"pbdoc(
        Constructor method.

        :param input: path to input file
        :param opts: AVDictionary options that will be passed to AVFormat context.
    )pbdoc")
      .def("DecodeSingleFrame", &PyFfmpegDecoder::DecodeSingleFrame,
           py::arg("frame"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Decode single video frame from input file.

        :param frame: decoded video frame
        :return: True in case of success, False otherwise
    )pbdoc")
      .def("DecodeSingleSurface", &PyFfmpegDecoder::DecodeSingleSurface,
           py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Decode single video frame from input file and upload to GPU memory.

        :return: Surface allocated in GPU memory. It's Empty() in case of failure,
        non-empty otherwise.
    )pbdoc")
      .def("GetMotionVectors", &PyFfmpegDecoder::GetMotionVectors,
           py::return_value_policy::move,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Return motion vectors of last decoded video frame.

        :return: numpy array with motion vectors.
    )pbdoc")
      .def("Codec", &PyFfmpegDecoder::Codec,
           R"pbdoc(
        Return video codec used in encoded video stream.
    )pbdoc")
      .def("Width", &PyFfmpegDecoder::Width,
           R"pbdoc(
        Return encoded video file width in pixels.
    )pbdoc")
      .def("Height", &PyFfmpegDecoder::Height,
           R"pbdoc(
        Return encoded video file height in pixels.
    )pbdoc")
      .def("Framerate", &PyFfmpegDecoder::Framerate,
           R"pbdoc(
        Return encoded video file framerate.
    )pbdoc")
      .def("ColorSpace", &PyFfmpegDecoder::Color_Space,
           R"pbdoc(
        Get color space information stored in video file.
        Please not that some video containers may not store this information.

        :return: color space information
    )pbdoc")
      .def("ColorRange", &PyFfmpegDecoder::Color_Range,
           R"pbdoc(
        Get color range information stored in video file.
        Please not that some video containers may not store this information.

        :return: color range information
    )pbdoc")
      .def("Format", &PyFfmpegDecoder::PixelFormat,
           R"pbdoc(
        Return encoded video file pixel format.
    )pbdoc");
}
