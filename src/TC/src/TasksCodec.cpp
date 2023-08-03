/*
 * Copyright 2019 NVIDIA Corporation
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

#include <chrono>
#include <fstream>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "CodecsSupport.hpp"
#include "MemoryInterfaces.hpp"
#include "NppCommon.hpp"
#include "Tasks.hpp"

#include "NvCodecCLIOptions.h"
#include "NvCodecUtils.h"
#include "NvEncoderCuda.h"

#include "FFmpegDemuxer.h"
#include "NvDecoder.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

using namespace VPF;
using namespace std;
using namespace chrono;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

namespace VPF
{
struct NvencEncodeFrame_Impl {
  using packet = vector<uint8_t>;

  NV_ENC_BUFFER_FORMAT enc_buffer_format;
  queue<packet> packetQueue;
  vector<uint8_t> lastPacket;
  Buffer* pElementaryVideo;
  NvEncoderCuda* pEncoderCuda = nullptr;
  CUcontext context = nullptr;
  CUstream stream = 0;
  bool didEncode = false;
  bool didFlush = false;
  NV_ENC_RECONFIGURE_PARAMS recfg_params;
  NV_ENC_INITIALIZE_PARAMS& init_params;
  NV_ENC_CONFIG encodeConfig;
  std::map<NV_ENC_CAPS, int> capabilities;

  NvencEncodeFrame_Impl() = delete;
  NvencEncodeFrame_Impl(const NvencEncodeFrame_Impl& other) = delete;
  NvencEncodeFrame_Impl& operator=(const NvencEncodeFrame_Impl& other) = delete;

  uint32_t GetWidth() const { return pEncoderCuda->GetEncodeWidth(); };
  uint32_t GetHeight() const { return pEncoderCuda->GetEncodeHeight(); };
  int GetCap(NV_ENC_CAPS cap) const
  {
    auto it = capabilities.find(cap);
    if (it != capabilities.end()) {
      return it->second;
    }

    return -1;
  }

  NvencEncodeFrame_Impl(NV_ENC_BUFFER_FORMAT format,
                        NvEncoderClInterface& cli_iface, CUcontext ctx,
                        CUstream str, int32_t width, int32_t height,
                        bool verbose)
      : init_params(recfg_params.reInitEncodeParams)
  {
    pElementaryVideo = Buffer::Make(0U);

    context = ctx;
    stream = str;
    pEncoderCuda = new NvEncoderCuda(context, width, height, format);
    enc_buffer_format = format;

    init_params = {NV_ENC_INITIALIZE_PARAMS_VER};
    encodeConfig = {NV_ENC_CONFIG_VER};
    init_params.encodeConfig = &encodeConfig;

    cli_iface.SetupInitParams(init_params, false, pEncoderCuda->GetApi(),
                              pEncoderCuda->GetEncoder(), capabilities,
                              verbose);

    pEncoderCuda->CreateEncoder(&init_params);

    pEncoderCuda->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&stream,
                                   (NV_ENC_CUSTREAM_PTR)&stream);

  }

  bool Reconfigure(NvEncoderClInterface& cli_iface, bool force_idr,
                   bool reset_enc, bool verbose)
  {
    recfg_params.version = NV_ENC_RECONFIGURE_PARAMS_VER;
    recfg_params.resetEncoder = reset_enc;
    recfg_params.forceIDR = force_idr;

    cli_iface.SetupInitParams(init_params, true, pEncoderCuda->GetApi(),
                              pEncoderCuda->GetEncoder(), capabilities,
                              verbose);

    return pEncoderCuda->Reconfigure(&recfg_params);
  }

  ~NvencEncodeFrame_Impl()
  {
    pEncoderCuda->DestroyEncoder();
    delete pEncoderCuda;
    delete pElementaryVideo;
  }
};    
}

NvencEncodeFrame* NvencEncodeFrame::Make(CUstream cuStream, CUcontext cuContext,
                                         NvEncoderClInterface& cli_iface,
                                         NV_ENC_BUFFER_FORMAT format,
                                         uint32_t width, uint32_t height,
                                         bool verbose)
{
  return new NvencEncodeFrame(cuStream, cuContext, cli_iface, format, width,
                              height, verbose);
}

bool VPF::NvencEncodeFrame::Reconfigure(NvEncoderClInterface& cli_iface,
                                        bool force_idr, bool reset_enc,
                                        bool verbose)
{
  return pImpl->Reconfigure(cli_iface, force_idr, reset_enc, verbose);
}

NvencEncodeFrame::NvencEncodeFrame(CUstream cuStream, CUcontext cuContext,
                                   NvEncoderClInterface& cli_iface,
                                   NV_ENC_BUFFER_FORMAT format, uint32_t width,
                                   uint32_t height, bool verbose)
    :

      Task("NvencEncodeFrame", NvencEncodeFrame::numInputs,
           NvencEncodeFrame::numOutputs, nullptr, nullptr)
{
  pImpl = new NvencEncodeFrame_Impl(format, cli_iface, cuContext, cuStream,
                                    width, height, verbose);
}

NvencEncodeFrame::~NvencEncodeFrame() { delete pImpl; };

TaskExecStatus NvencEncodeFrame::Run()
{
  NvtxMark tick(GetName());
  SetOutput(nullptr, 0U);

  try {
    auto& pEncoderCuda = pImpl->pEncoderCuda;
    auto& didFlush = pImpl->didFlush;
    auto& didEncode = pImpl->didEncode;
    auto& context = pImpl->context;
    auto input = (Surface*)GetInput(0U);
    vector<vector<uint8_t>> encPackets;

    if (input) {
      auto& stream = pImpl->stream;
      const NvEncInputFrame* encoderInputFrame =
          pEncoderCuda->GetNextInputFrame();
      auto width = input->Width(), height = input->Height(),
           pitch = input->Pitch();

      bool is_resize_needed = (pEncoderCuda->GetEncodeWidth() != width) ||
                              (pEncoderCuda->GetEncodeHeight() != height);

      if (is_resize_needed) {
        return TASK_EXEC_FAIL;
      } else {
        NvEncoderCuda::CopyToDeviceFrame(
            context, stream, (void*)input->PlanePtr(), pitch,
            (CUdeviceptr)encoderInputFrame->inputPtr,
            (int32_t)encoderInputFrame->pitch, pEncoderCuda->GetEncodeWidth(),
            pEncoderCuda->GetEncodeHeight(), CU_MEMORYTYPE_DEVICE,
            encoderInputFrame->bufferFormat, encoderInputFrame->chromaOffsets,
            encoderInputFrame->numChromaPlanes);
      }
      

      auto pSEI = (Buffer*)GetInput(2U);
      NV_ENC_SEI_PAYLOAD payload = {0};
      if (pSEI) {
        payload.payloadSize = pSEI->GetRawMemSize();
        // Unregistered user data for H.265 and H.264 both;
        payload.payloadType = 5;
        payload.payload = pSEI->GetDataAs<uint8_t>();
      }

      auto const seiNumber = pSEI ? 1U : 0U;
      auto pPayload = pSEI ? &payload : nullptr;

      auto sync = GetInput(1U);
      if (sync) {
        pEncoderCuda->EncodeFrame(encPackets, nullptr, false, seiNumber,
                                  pPayload);
      } else {
        pEncoderCuda->EncodeFrame(encPackets, nullptr, true, seiNumber,
                                  pPayload);
      }
      didEncode = true;
    } else if (didEncode && !didFlush) {
      // No input after a while means we're flushing;
      pEncoderCuda->EndEncode(encPackets);
      didFlush = true;
    }

    /* Push encoded packets into queue;
     */
    for (auto& packet : encPackets) {
      pImpl->packetQueue.push(packet);
    }

    /* Then return least recent packet;
     */
    pImpl->lastPacket.clear();
    if (!pImpl->packetQueue.empty()) {
      pImpl->lastPacket = pImpl->packetQueue.front();
      pImpl->pElementaryVideo->Update(pImpl->lastPacket.size(),
                                      (void*)pImpl->lastPacket.data());
      pImpl->packetQueue.pop();
      SetOutput(pImpl->pElementaryVideo, 0U);
    }

    return TASK_EXEC_SUCCESS;
  } catch (exception& e) {
    cerr << e.what() << endl;
    return TASK_EXEC_FAIL;
  }
}

uint32_t NvencEncodeFrame::GetWidth() const { return pImpl->GetWidth(); }

uint32_t NvencEncodeFrame::GetHeight() const { return pImpl->GetHeight(); }

int NvencEncodeFrame::GetCapability(NV_ENC_CAPS cap) const
{
  return pImpl->GetCap(cap);
}

namespace VPF
{
struct NvdecDecodeFrame_Impl {
  NvDecoder nvDecoder;
  Surface* pLastSurface = nullptr;
  Buffer* pPacketData = nullptr;
  CUstream stream = 0;
  CUcontext context = nullptr;
  bool didDecode = false;

  NvdecDecodeFrame_Impl() = delete;
  NvdecDecodeFrame_Impl(const NvdecDecodeFrame_Impl& other) = delete;
  NvdecDecodeFrame_Impl& operator=(const NvdecDecodeFrame_Impl& other) = delete;

  NvdecDecodeFrame_Impl(CUstream cuStream, CUcontext cuContext,
                        cudaVideoCodec videoCodec, Pixel_Format format)
      : stream(cuStream), context(cuContext),
        nvDecoder(cuStream, cuContext, videoCodec)
  {
    pLastSurface = Surface::Make(format);
    pPacketData = Buffer::MakeOwnMem(sizeof(PacketData));
  }

  ~NvdecDecodeFrame_Impl()
  {
    delete pLastSurface;
    delete pPacketData;
  }
};
} // namespace VPF

NvdecDecodeFrame* NvdecDecodeFrame::Make(CUstream cuStream, CUcontext cuContext,
                                         cudaVideoCodec videoCodec,
                                         uint32_t decodedFramesPoolSize,
                                         uint32_t coded_width,
                                         uint32_t coded_height,
                                         Pixel_Format format)
{
  return new NvdecDecodeFrame(cuStream, cuContext, videoCodec,
                              decodedFramesPoolSize, coded_width, coded_height,
                              format);
}

NvdecDecodeFrame::NvdecDecodeFrame(CUstream cuStream, CUcontext cuContext,
                                   cudaVideoCodec videoCodec,
                                   uint32_t decodedFramesPoolSize,
                                   uint32_t coded_width, uint32_t coded_height,
                                   Pixel_Format format)
    :

      Task("NvdecDecodeFrame", NvdecDecodeFrame::numInputs,
           NvdecDecodeFrame::numOutputs, nullptr, nullptr)
{
  pImpl = new NvdecDecodeFrame_Impl(cuStream, cuContext, videoCodec, format);
}

NvdecDecodeFrame::~NvdecDecodeFrame()
{
  auto lastSurface = pImpl->pLastSurface->PlanePtr();
  pImpl->nvDecoder.UnlockSurface(lastSurface);
  delete pImpl;
}

TaskExecStatus NvdecDecodeFrame::Run()
{
  NvtxMark tick(GetName());
  ClearOutputs();
  try {
    auto& decoder = pImpl->nvDecoder;
    auto pEncFrame = (Buffer*)GetInput();

    if (!pEncFrame && !pImpl->didDecode) {
      /* Empty input given + we've never did decoding means something went
       * wrong; Otherwise (no input + we did decode) means we're flushing;
       */
      return TASK_EXEC_FAIL;
    }

    bool isSurfaceReturned = false;
    uint64_t timestamp = 0U;
    auto pPktData = (Buffer*)GetInput(1U);
    if (pPktData) {
      auto p_pkt_data = pPktData->GetDataAs<PacketData>();
      timestamp = p_pkt_data->pts;
      pImpl->pPacketData->Update(sizeof(*p_pkt_data), p_pkt_data);
    }

    auto const no_eos = nullptr != GetInput(2);

    /* This will feed decoder with input timestamp.
     * It will also return surface + it's timestamp.
     * So timestamp is input + output parameter. */
    DecodedFrameContext dec_ctx;
    if (no_eos) {
      dec_ctx.no_eos = true;
    }

    {
      /* Do this in separate scope because we don't want to measure
       * DecodeLockSurface() function run time;
       */
      stringstream ss;
      ss << "Start decode for frame with pts " << timestamp;
      NvtxMark decode_k_off(ss.str().c_str());
    }

    PacketData in_pkt_data = {0};
    if (pPktData) {
      auto p_pkt_data = pPktData->GetDataAs<PacketData>();
      in_pkt_data = *p_pkt_data;
    }

    isSurfaceReturned =
        decoder.DecodeLockSurface(pEncFrame, in_pkt_data, dec_ctx);
    pImpl->didDecode = true;

    if (isSurfaceReturned) {
      // Unlock last surface because we will use it later;
      auto lastSurface = pImpl->pLastSurface->PlanePtr();
      decoder.UnlockSurface(lastSurface);

      // Update the reconstructed frame data;
      auto rawW = decoder.GetWidth();
      auto rawH = decoder.GetHeight() + decoder.GetChromaHeight();
      auto rawP = decoder.GetDeviceFramePitch();

      // Element size for different bit depth;
      auto elem_size = 0U;
      switch (pImpl->nvDecoder.GetBitDepth()) {
      case 8U:
        elem_size = sizeof(uint8_t);
        break;
      case 10U:
        elem_size = sizeof(uint16_t);
        break;
      case 12U:
        elem_size = sizeof(uint16_t);
        break;
      default:
        return TASK_EXEC_FAIL;
      }

      SurfacePlane tmpPlane(rawW, rawH, rawP, elem_size, dec_ctx.mem);
      pImpl->pLastSurface->Update(&tmpPlane, 1);
      SetOutput(pImpl->pLastSurface, 0U);

      // Update the reconstructed frame timestamp;
      auto p_packet_data = pImpl->pPacketData->GetDataAs<PacketData>();
      memset(p_packet_data, 0, sizeof(*p_packet_data));
      *p_packet_data = dec_ctx.out_pdata;
      SetOutput(pImpl->pPacketData, 1U);

      {
        stringstream ss;
        ss << "End decode for frame with pts " << dec_ctx.pts;
        NvtxMark display_ready(ss.str().c_str());
      }

      return TASK_EXEC_SUCCESS;
    }

    /* If we have input and don't get output so far that's fine.
     * Otherwise input is NULL and we're flusing so we shall get frame.
     */
    return pEncFrame ? TASK_EXEC_SUCCESS : TASK_EXEC_FAIL;
  } catch (exception& e) {
    cerr << e.what() << endl;
    return TASK_EXEC_FAIL;
  }
  
}

void NvdecDecodeFrame::GetDecodedFrameParams(uint32_t& width, uint32_t& height,
                                             uint32_t& elem_size)
{
  width = pImpl->nvDecoder.GetWidth();
  height = pImpl->nvDecoder.GetHeight();
  elem_size = (pImpl->nvDecoder.GetBitDepth() + 7) / 8;
}

uint32_t NvdecDecodeFrame::GetDeviceFramePitch()
{
  return uint32_t(pImpl->nvDecoder.GetDeviceFramePitch());
}

int NvdecDecodeFrame::GetCapability(NV_DEC_CAPS cap) const
{
  CUVIDDECODECAPS decode_caps;
  memset((void*)&decode_caps, 0, sizeof(decode_caps));

  decode_caps.eCodecType = pImpl->nvDecoder.GetCodec();
  decode_caps.eChromaFormat = pImpl->nvDecoder.GetChromaFormat();
  decode_caps.nBitDepthMinus8 = pImpl->nvDecoder.GetBitDepth() - 8;

  auto ret = pImpl->nvDecoder._api().cuvidGetDecoderCaps(&decode_caps);
  if (CUDA_SUCCESS != ret) {
    return -1;
  }

  switch (cap) {
  case BIT_DEPTH_MINUS_8:
    return decode_caps.nBitDepthMinus8;
  case IS_CODEC_SUPPORTED:
    return decode_caps.bIsSupported;
  case OUTPUT_FORMAT_MASK:
    return decode_caps.nOutputFormatMask;
  case MAX_WIDTH:
    return decode_caps.nMaxWidth;
  case MAX_HEIGHT:
    return decode_caps.nMaxHeight;
  case MAX_MB_COUNT:
    return decode_caps.nMaxMBCount;
  case MIN_WIDTH:
    return decode_caps.nMinWidth;
  case MIN_HEIGHT:
    return decode_caps.nMinHeight;
#if CHECK_API_VERSION(11, 0)
  case IS_HIST_SUPPORTED:
    return decode_caps.bIsHistogramSupported;
  case HIST_COUNT_BIT_DEPTH:
    return decode_caps.nCounterBitDepth;
  case HIST_COUNT_BINS:
    return decode_caps.nMaxHistogramBins;
#endif
  default:
    return -1;
  }
}