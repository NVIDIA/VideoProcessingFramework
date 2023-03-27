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

int nvcvImagePitch = 0;

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

uint32_t PyNvEncoder::Width() const { return encWidth; }

uint32_t PyNvEncoder::Height() const { return encHeight; }

Pixel_Format PyNvEncoder::GetPixelFormat() const { return eFormat; }

std::map<NV_ENC_CAPS, int> PyNvEncoder::Capabilities()
{
  if (!upEncoder) {
    NvEncoderClInterface cli_interface(options);

    upEncoder.reset(NvencEncodeFrame::Make(
        cuda_str, cuda_ctx, cli_interface,
        NV12 == eFormat ? NV_ENC_BUFFER_FORMAT_NV12
                        : YUV444 == eFormat ? NV_ENC_BUFFER_FORMAT_YUV444
                                            : NV_ENC_BUFFER_FORMAT_UNDEFINED,
        encWidth, encHeight, verbose_ctor));
  }

  std::map<NV_ENC_CAPS, int> capabilities;
  capabilities.erase(capabilities.begin(), capabilities.end());
  for (int cap = NV_ENC_CAPS_NUM_MAX_BFRAMES; cap < NV_ENC_CAPS_EXPOSED_COUNT;
       cap++) {
    auto val = upEncoder->GetCapability((NV_ENC_CAPS)cap);
    capabilities[(NV_ENC_CAPS)cap] = val;
  }

  return capabilities;
}

int PyNvEncoder::GetFrameSizeInBytes() const
{ 
     switch (GetPixelFormat()) {
        
        case NV12:
            return Width() * (Height() + (Height() + 1) / 2);
        case YUV420_10bit:
        case YUV444:
            return Width() * Height() * 3;
        case YUV444_10bit:
            return 2 * Width() * Height() * 3;
        default:
            throw invalid_argument("Invalid Buffer format");
        return 0;
  }
}

bool PyNvEncoder::Reconfigure(const map<string, string>& encodeOptions,
                              bool force_idr, bool reset_enc, bool verbose)
{
  if (upEncoder) {
    NvEncoderClInterface cli_interface(encodeOptions);
    auto ret =
        upEncoder->Reconfigure(cli_interface, force_idr, reset_enc, verbose);
    if (!ret) {
      return ret;
    } else {
      encWidth = upEncoder->GetWidth();
      encHeight = upEncoder->GetHeight();
      uploader.reset(new PyFrameUploader(encWidth, encHeight, eFormat, cuda_ctx,
                                         cuda_str));
    }
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
  case YUV444_10bit:
    fmt_string = "YUV444_10bit";
    break;
  case YUV420_10bit:
    fmt_string = "YUV420_10bit";
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

    NV_ENC_BUFFER_FORMAT encoderFormat;
    

    switch (eFormat) {
    case VPF::NV12:
      encoderFormat = NV_ENC_BUFFER_FORMAT_NV12;
      break;
    case VPF::YUV444:
      encoderFormat = NV_ENC_BUFFER_FORMAT_YUV444;
      break;
    case VPF::YUV420_10bit: //P12 already has memory representation similar to 10 bit yuv420, hence reusing the same class
    case VPF::P12:
      encoderFormat = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
      break;
    case VPF::YUV444_10bit:
      encoderFormat = NV_ENC_BUFFER_FORMAT_YUV444_10BIT;
      break;
    default:
      throw invalid_argument("Input buffer format not supported by VPF currently.");
        break;
    }

    upEncoder.reset(NvencEncodeFrame::Make(
        cuda_str, cuda_ctx, 
        cli_interface,
        encoderFormat,
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
    upEncoder->SetInput((Token*)0xdeadbeefull, 1U);
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

bool PyNvEncoder::EncodeFromNVCVImage(py::object nvcvImage,
                                py::array_t<uint8_t>& packet, bool bIsnvcvImage)
{
  struct NVCVImageMapper {
    int nWidth[3];
    int nHeight[3];
    int nStride[3];
    CUdeviceptr ptrToData[3];
  };

  if (!bIsnvcvImage) {
    std::cerr << "Please set the boolean to true" << std::endl;
    return false;
  }

  NVCVImageMapper nv12Mapper;

  memset(&nv12Mapper, 0, sizeof(NVCVImageMapper));

  nvcvImage  = nvcvImage.attr("cuda")();
 

    if (py::hasattr(nvcvImage, "__cuda_array_interface__")) {
      py::dict dict =
          (nvcvImage).attr("__cuda_array_interface__").cast<py::dict>();
      if (!dict.contains("shape") || !dict.contains("typestr") ||
          !dict.contains("data") || !dict.contains("version")) {
        return false;
      }
      int version = dict["version"].cast<int>();
      if (version < 2) {
        return false;
      }

      py::tuple tdata = dict["data"].cast<py::tuple>();
      void     *ptr   = reinterpret_cast<void *>(tdata[0].cast<long>());
      PyNvEncoder::CheckValidCUDABuffer(ptr);
      
      nv12Mapper.ptrToData[0 ] =(CUdeviceptr) ptr;

      py::tuple shape = dict["shape"].cast<py::tuple>();

      nv12Mapper.nWidth[0 ] = shape[1].cast<long>();
      nv12Mapper.nHeight[0 ] = shape[0].cast<long>();

      std::string dtype = dict["typestr"].cast<std::string>();
     
    }

    int width = nv12Mapper.nWidth[0];
    int height = nv12Mapper.nHeight[0];
    int stride = nvcvImagePitch;
    CUdeviceptr lumaDataPtr = nv12Mapper.ptrToData[0];
    CUdeviceptr chromaDataPtr = lumaDataPtr + (width * height);
    shared_ptr<SurfaceNV12> nv12Planar =
        make_shared<SurfaceNV12>(
        width, 
        height,
        stride, 
        lumaDataPtr);

    EncodeContext ctx(nv12Planar, &packet, nullptr, false, false);
    bool bResult = EncodeSingleSurface(ctx);

  
  return bResult;
}

void Init_PyNvEncoder(py::module& m)
{
  py::enum_<NV_ENC_CAPS>(m, "NV_ENC_CAPS")
      .value("NUM_MAX_BFRAMES", NV_ENC_CAPS_NUM_MAX_BFRAMES)
      .value("SUPPORTED_RATECONTROL_MODES", NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES)
      .value("SUPPORT_FIELD_ENCODING", NV_ENC_CAPS_SUPPORT_FIELD_ENCODING)
      .value("SUPPORT_MONOCHROME", NV_ENC_CAPS_SUPPORT_MONOCHROME)
      .value("SUPPORT_FMO", NV_ENC_CAPS_SUPPORT_FMO)
      .value("SUPPORT_QPELMV", NV_ENC_CAPS_SUPPORT_QPELMV)
      .value("SUPPORT_BDIRECT_MODE", NV_ENC_CAPS_SUPPORT_BDIRECT_MODE)
      .value("SUPPORT_CABAC", NV_ENC_CAPS_SUPPORT_CABAC)
      .value("SUPPORT_ADAPTIVE_TRANSFORM", NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM)
      .value("SUPPORT_STEREO_MVC", NV_ENC_CAPS_SUPPORT_STEREO_MVC)
      .value("NUM_MAX_TEMPORAL_LAYERS", NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS)
      .value("SUPPORT_HIERARCHICAL_PFRAMES", NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES)
      .value("SUPPORT_HIERARCHICAL_BFRAMES", NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES)
      .value("LEVEL_MAX", NV_ENC_CAPS_LEVEL_MAX)
      .value("LEVEL_MIN", NV_ENC_CAPS_LEVEL_MIN)
      .value("SEPARATE_COLOUR_PLANE", NV_ENC_CAPS_SEPARATE_COLOUR_PLANE)
      .value("WIDTH_MAX", NV_ENC_CAPS_WIDTH_MAX)
      .value("HEIGHT_MAX", NV_ENC_CAPS_HEIGHT_MAX)
      .value("SUPPORT_TEMPORAL_SVC", NV_ENC_CAPS_SUPPORT_TEMPORAL_SVC)
      .value("SUPPORT_DYN_RES_CHANGE", NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE)
      .value("SUPPORT_DYN_BITRATE_CHANGE", NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE)
      .value("SUPPORT_DYN_FORCE_CONSTQP", NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP)
      .value("SUPPORT_DYN_RCMODE_CHANGE", NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE)
      .value("SUPPORT_SUBFRAME_READBACK", NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK)
      .value("SUPPORT_CONSTRAINED_ENCODING", NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING)
      .value("SUPPORT_INTRA_REFRESH", NV_ENC_CAPS_SUPPORT_INTRA_REFRESH)
      .value("SUPPORT_CUSTOM_VBV_BUF_SIZE", NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)
      .value("SUPPORT_DYNAMIC_SLICE_MODE", NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE)
      .value("SUPPORT_REF_PIC_INVALIDATION", NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION)
      .value("PREPROC_SUPPORT", NV_ENC_CAPS_PREPROC_SUPPORT)
      .value("ASYNC_ENCODE_SUPPORT", NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT)
      .value("MB_NUM_MAX", NV_ENC_CAPS_MB_NUM_MAX)
      .value("MB_PER_SEC_MAX", NV_ENC_CAPS_MB_PER_SEC_MAX)
      .value("SUPPORT_YUV444_ENCODE", NV_ENC_CAPS_SUPPORT_YUV444_ENCODE)
      .value("SUPPORT_LOSSLESS_ENCODE", NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE)
      .value("SUPPORT_SAO", NV_ENC_CAPS_SUPPORT_SAO)
      .value("SUPPORT_MEONLY_MODE", NV_ENC_CAPS_SUPPORT_MEONLY_MODE)
      .value("SUPPORT_LOOKAHEAD", NV_ENC_CAPS_SUPPORT_LOOKAHEAD)
      .value("SUPPORT_TEMPORAL_AQ", NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ)
      .value("SUPPORT_10BIT_ENCODE", NV_ENC_CAPS_SUPPORT_10BIT_ENCODE)
      .value("NUM_MAX_LTR_FRAMES", NV_ENC_CAPS_NUM_MAX_LTR_FRAMES)
      .value("SUPPORT_WEIGHTED_PREDICTION", NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION)
      .value("DYNAMIC_QUERY_ENCODER_CAPACITY", NV_ENC_CAPS_DYNAMIC_QUERY_ENCODER_CAPACITY)
      .value("SUPPORT_BFRAME_REF_MODE", NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE)
      .value("SUPPORT_EMPHASIS_LEVEL_MAP", NV_ENC_CAPS_SUPPORT_EMPHASIS_LEVEL_MAP)
      .value("WIDTH_MIN", NV_ENC_CAPS_WIDTH_MIN)
      .value("HEIGHT_MIN", NV_ENC_CAPS_HEIGHT_MIN)
      .value("SUPPORT_MULTIPLE_REF_FRAMES", NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES)
      .value("SUPPORT_ALPHA_LAYER_ENCODING", NV_ENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING)
      .value("EXPOSED_COUNT", NV_ENC_CAPS_EXPOSED_COUNT)
      .export_values();

  py::class_<PyNvEncoder>(m, "PyNvEncoder")
      .def(py::init<const map<string, string>&, int, Pixel_Format, bool>(),
           py::arg("settings"), py::arg("gpu_id"), py::arg("format") = NV12,
           py::arg("verbose") = false,
           R"pbdoc(
        Constructor method.

        :param settings: Dictionary with nvenc settings
        :param gpu_id: what GPU to run encode on
        :param format: pixel format to use by codec
        :param verbose: output verbose information to log
    )pbdoc")
      .def(py::init<const map<string, string>&, size_t, size_t, Pixel_Format,
                    bool>(),
           py::arg("settings"), py::arg("context"), py::arg("stream"),
           py::arg("format") = NV12, py::arg("verbose") = false,
           R"pbdoc(
        Constructor method.

        :param settings: Dictionary with nvenc settings
        :param context: CUDA context to use
        :param stream: CUDA stream to use
        :param format: pixel format to use by codec
        :param verbose: output verbose information to log
    )pbdoc")
      .def("Reconfigure", &PyNvEncoder::Reconfigure, py::arg("settings"),
           py::arg("force_idr") = false, py::arg("reset_encoder") = false,
           py::arg("verbose") = false,
           R"pbdoc(
        DESC.

        :param settings: Dictionary with nvenc settings
        :param force_idr: force next encoded frame to be IDR key frame
        :param reset_encoder: force encoder reset
        :param verbose: output verbose information to log
        :return:
    )pbdoc")
      .def("Width", &PyNvEncoder::Width,
           R"pbdoc(
        Return encoded video stream width in pixels.
    )pbdoc")
      .def("Height", &PyNvEncoder::Height,
           R"pbdoc(
        Return encoded video stream height in pixels.
    )pbdoc")
      .def("Format", &PyNvEncoder::GetPixelFormat,
           R"pbdoc(
        Return encoded video stream pixel format.
    )pbdoc")
      .def("GetFrameSizeInBytes", &PyNvEncoder::GetFrameSizeInBytes,
          R"pbdoc(
        This function is used to get the current frame size based on pixel format.
    )pbdoc")
      .def("Capabilities", &PyNvEncoder::Capabilities,
           py::return_value_policy::move,
           R"pbdoc(
        Return dictionary with Nvenc capabilities.
    )pbdoc")
      .def(
          "EncodeSingleSurface",
           (&PyNvEncoder::EncodeFromNVCVImage),
           py::arg("nvcvimage"), py::arg("packet"),
           py::arg("testbool"),
          R"pbdoc(
        Encode single Surface. Please note that this function may not return
        compressed video packet.

        :param surface: raw input Surface
        :param packet: output compressed packet
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&, bool, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::arg("sync"), py::arg("append"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Encode single Surface. Please not that this function may not return
        compressed video packet.

        :param surface: raw input Surface
        :param packet: output compressed packet
        :param sei: unregistered user data SEI information to be attached to encoded bitstream
        :param sync: run function in sync mode, will ensure encoded packet is returned when function returns
        :param append: append encoded packet to input packet
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::arg("sync"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Encode single Surface. Please not that this function may not return
        compressed video packet.

        :param surface: raw input Surface
        :param packet: output compressed packet
        :param sei: unregistered user data SEI information to be attached to encoded bitstream
        :param sync: run function in sync mode, will ensure encoded packet is returned when function returns
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sync"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Encode single Surface. Please not that this function may not return
        compressed video packet.

        :param surface: raw input Surface
        :param packet: output compressed packet
        :param sync: run function in sync mode, will ensure encoded packet is returned when function returns
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Encode single Surface. Please not that this function may not return
        compressed video packet.

        :param surface: raw input Surface
        :param packet: output compressed packet
        :param sei: unregistered user data SEI information to be attached to encoded bitstream
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array_t<uint8_t>&>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Encode single Surface. Please not that this function may not return
        compressed video packet.

        :param surface: raw input Surface
        :param packet: output compressed packet
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&, bool, bool>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"),
           py::arg("append"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Combination of UploadSingleFrame + EncodeSingleSurface.

        :param frame: raw video frame
        :param packet: output compressed packet
        :param sei: unregistered user data SEI information to be attached to encoded bitstream
        :param sync: run function in sync mode, will ensure encoded packet is returned when function returns
        :param append: append encoded packet to input packet
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&, bool>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"), py::arg("sync"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Combination of UploadSingleFrame + EncodeSingleSurface.

        :param frame: raw video frame
        :param packet: output compressed packet
        :param sei: unregistered user data SEI information to be attached to encoded bitstream
        :param sync: run function in sync mode, will ensure encoded packet is returned when function returns
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def(
          "EncodeSingleFrame",
          py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&, bool>(
              &PyNvEncoder::EncodeFrame),
          py::arg("frame"), py::arg("packet"), py::arg("sync"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Combination of UploadSingleFrame + EncodeSingleSurface.

        :param frame: raw video frame
        :param packet: output compressed packet
        :param sync: run function in sync mode, will ensure encoded packet is returned when function returns
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&,
                             const py::array_t<uint8_t>&>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"), py::arg("sei"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Combination of UploadSingleFrame + EncodeSingleSurface.

        :param frame: raw video frame
        :param packet: output compressed packet
        :param sei: unregistered user data SEI information to be attached to encoded bitstream
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("EncodeSingleFrame",
           py::overload_cast<py::array_t<uint8_t>&, py::array_t<uint8_t>&>(
               &PyNvEncoder::EncodeFrame),
           py::arg("frame"), py::arg("packet"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Combination of UploadSingleFrame + EncodeSingleSurface.

        :param frame: raw video frame
        :param packet: output compressed packet
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("Flush", &PyNvEncoder::Flush, py::arg("packets"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Flush encoder.
        Use this method in the end of encoding session to obtain all remaining
        compressed frames.

        :param packets: one or multiple compressed packets squashed together.
        :return: True in case of success, False otherwise.
    )pbdoc")
      .def("FlushSinglePacket", &PyNvEncoder::FlushSinglePacket,
           py::arg("packets"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Flush encoder.
        Use this method in the end of encoding session to obtain single remaining
        compressed frame. TO flush encoder completely you need to call this
        method multiple times.

        :param packets: single compressed packet.
        :return: True in case of success, False otherwise.
    )pbdoc");
}
