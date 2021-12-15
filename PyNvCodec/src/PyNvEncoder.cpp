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

struct EncodeContext {
  std::shared_ptr<Surface> rawSurface;
  py::array_t<uint8_t>* pPacket;
  const py::array_t<uint8_t>* pMessageSEI;
  bool sync;
  bool append;

  EncodeContext(std::shared_ptr<Surface> spRawSurface,
                py::array_t<uint8_t>* packet,
                const py::array_t<uint8_t>* messageSEI, bool is_sync,
                bool is_append)
      : rawSurface(spRawSurface), pPacket(packet), pMessageSEI(messageSEI),
        sync(is_sync), append(is_append)
  {
  }
};

bool PyNvEncoder::Reconfigure(const map<string, string>& encodeOptions,
                              bool force_idr, bool reset_enc, bool verbose)
{

  if (upEncoder) {
    NvEncoderClInterface cli_interface(encodeOptions);
    return upEncoder->Reconfigure(cli_interface, force_idr, reset_enc, verbose);
  }

  return true;
}

PyNvEncoder::PyNvEncoder(const map<string, string>& encodeOptions, int gpuID,
                         Pixel_Format format, bool verbose)
    : PyNvEncoder(encodeOptions, CudaResMgr::Instance().GetCtx(gpuID),
                  CudaResMgr::Instance().GetStream(gpuID), format, verbose)
{
}

PyNvEncoder::PyNvEncoder(const map<string, string>& encodeOptions,
                         CUcontext ctx, CUstream str, Pixel_Format format,
                         bool verbose)
    : upEncoder(nullptr), uploader(nullptr), options(encodeOptions),
      verbose_ctor(verbose), eFormat(format)
{

  // Parse resolution;
  auto ParseResolution = [&](const string& res_string, uint32_t& width,
                             uint32_t& height) {
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

bool PyNvEncoder::EncodeSingleSurface(EncodeContext& ctx)
{
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
    upEncoder->SetInput((Token*)0xdeadbeef, 1U);
  }

  if (ctx.pMessageSEI && ctx.pMessageSEI->size()) {
    /* Set 3rd input in case we have SEI message;
     */
    upEncoder->SetInput(spSEI.get(), 2U);
  }

  if (TASK_EXEC_FAIL == upEncoder->Execute()) {
    throw runtime_error("Error while encoding frame");
  }

  auto encodedFrame = (Buffer*)upEncoder->GetOutput(0U);
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

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t>& inRawFrame,
                              py::array_t<uint8_t>& packet)
{
  if (!uploader) {
    uploader.reset(
        new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t>& inRawFrame,
                              py::array_t<uint8_t>& packet,
                              const py::array_t<uint8_t>& messageSEI)
{
  if (!uploader) {
    uploader.reset(
        new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet,
                       messageSEI);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t>& inRawFrame,
                              py::array_t<uint8_t>& packet,
                              const py::array_t<uint8_t>& messageSEI, bool sync)
{
  if (!uploader) {
    uploader.reset(
        new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet,
                       messageSEI, sync);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t>& inRawFrame,
                              py::array_t<uint8_t>& packet, bool sync)
{
  if (!uploader) {
    uploader.reset(
        new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet, sync);
}

bool PyNvEncoder::EncodeFrame(py::array_t<uint8_t>& inRawFrame,
                              py::array_t<uint8_t>& packet,
                              const py::array_t<uint8_t>& messageSEI, bool sync,
                              bool append)
{
  if (!uploader) {
    uploader.reset(
        new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx, cuda_str));
  }

  return EncodeSurface(uploader->UploadSingleFrame(inRawFrame), packet,
                       messageSEI, sync, append);
}

bool PyNvEncoder::FlushSinglePacket(py::array_t<uint8_t>& packet)
{
  /* Keep feeding encoder with null input until it returns zero-size
   * surface; */
  shared_ptr<Surface> spRawSurface = nullptr;
  const py::array_t<uint8_t>* messageSEI = nullptr;
  auto const is_sync = true;
  auto const is_append = false;
  EncodeContext ctx(spRawSurface, &packet, messageSEI, is_sync, is_append);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::Flush(py::array_t<uint8_t>& packets)
{
  uint32_t num_packets = 0U;
  do {
    if (!FlushSinglePacket(packets)) {
      break;
    }
    num_packets++;
  } while (true);
  return (num_packets > 0U);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t>& packet,
                                const py::array_t<uint8_t>& messageSEI,
                                bool sync, bool append)
{
  EncodeContext ctx(rawSurface, &packet, &messageSEI, sync, append);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t>& packet,
                                const py::array_t<uint8_t>& messageSEI,
                                bool sync)
{
  EncodeContext ctx(rawSurface, &packet, &messageSEI, sync, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t>& packet, bool sync)
{
  EncodeContext ctx(rawSurface, &packet, nullptr, sync, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t>& packet,
                                const py::array_t<uint8_t>& messageSEI)
{
  EncodeContext ctx(rawSurface, &packet, &messageSEI, false, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array_t<uint8_t>& packet)
{
  EncodeContext ctx(rawSurface, &packet, nullptr, false, false);
  return EncodeSingleSurface(ctx);
}

void Init_PyNvEncoder(py::module& m)
{
  py::class_<PyNvEncoder>(m, "PyNvEncoder")
      .def(py::init<const map<string, string>&, int, Pixel_Format, bool>(),
           py::arg("settings"), py::arg("gpu_id"), py::arg("format") = NV12,
           py::arg("verbose") = false)
      .def(py::init<const map<string, string>&, size_t, size_t, Pixel_Format,
                    bool>(),
           py::arg("settings"), py::arg("cuda_context"), py::arg("cuda_stream"),
           py::arg("format") = NV12, py::arg("verbose") = false)
      .def("Reconfigure", &PyNvEncoder::Reconfigure, py::arg("settings"),
           py::arg("force_idr") = false, py::arg("reset_encoder") = false,
           py::arg("verbose") = false)
      .def("Width", &PyNvEncoder::Width)
      .def("Height", &PyNvEncoder::Height)
      .def("Format", &PyNvEncoder::GetPixelFormat)
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&, bool, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::arg("sync"), py::arg("append"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::arg("sync"), py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sync"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&, bool, bool>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"),
           py::arg("append"), py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&, bool>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "EncodeSingleFrame",
          py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&, bool>(
              &PyNvEncoder::EncodeFrame),
          py::arg("frame"), py::arg("packet"), py::arg("sync"),
          py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"),
           py::call_guard<py::gil_scoped_release>())
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"),
           py::call_guard<py::gil_scoped_release>())
      .def("Flush", &PyNvEncoder::Flush, py::arg("packets"),
           py::call_guard<py::gil_scoped_release>())
      .def("FlushSinglePacket", &PyNvEncoder::FlushSinglePacket,
           py::arg("packets"), py::call_guard<py::gil_scoped_release>());
}