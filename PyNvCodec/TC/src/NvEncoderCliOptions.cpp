/*
 * Copyright 2020 NVIDIA Corporation
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

#include "NvEncoderCLIOptions.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

using namespace std;
using namespace VPF;

namespace VPF {
/* Some encoding parameters shall be passed from upper level
 * configure functions;
 */
struct ParentParams {
  GUID codec_guid;
  uint32_t gop_length;
  bool is_low_latency;
  bool is_lossless;
};
} // namespace VPF

auto GetCapabilityValue = [](GUID guidCodec, NV_ENC_CAPS capsToQuery,
                              NV_ENCODE_API_FUNCTION_LIST api_func,
                              void *encoder) {
  NV_ENC_CAPS_PARAM capsParam = {NV_ENC_CAPS_PARAM_VER};
  capsParam.capsToQuery = capsToQuery;
  int v;
  api_func.nvEncGetEncodeCaps(encoder, guidCodec, &capsParam, &v);
  return v;
};

auto FindAttribute = [](const map<string, string> &options,
                         const string &option) {
  auto it = options.find(option);
  if (it != options.end()) {
    return it->second;
  }

  return string("");
};

auto FindCodecGuid = [](const string &codec_name) {
  static const map<string, GUID> codec_guids = {
      {"h264", NV_ENC_CODEC_H264_GUID}, {"hevc", NV_ENC_CODEC_HEVC_GUID}};

  auto it = codec_guids.find(codec_name);
  if (it != codec_guids.end()) {
    return it->second;
  }

  throw invalid_argument("Invalid codec given.");
};

struct PresetProperties {
  GUID preset_guid;
  bool is_low_latency;
  bool is_lossless;

  PresetProperties(GUID guid, bool ll, bool lossless)
      : preset_guid(guid), is_low_latency(ll), is_lossless(lossless) {}
};

auto FindPresetProperties = [](const string &preset_name) {
  static const map<string, PresetProperties> preset_guids = {
      {"default", PresetProperties(NV_ENC_PRESET_DEFAULT_GUID, false, false)},
      {"hp", PresetProperties(NV_ENC_PRESET_HP_GUID, false, false)},
      {"hq", PresetProperties(NV_ENC_PRESET_HQ_GUID, false, false)},
      {"bd", PresetProperties(NV_ENC_PRESET_BD_GUID, false, false)},
      {"ll",
       PresetProperties(NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID, true, false)},
      {"ll_hp",
       PresetProperties(NV_ENC_PRESET_LOW_LATENCY_HP_GUID, true, false)},
      {"ll_hq",
       PresetProperties(NV_ENC_PRESET_LOW_LATENCY_HQ_GUID, true, false)},
      {"lossless",
       PresetProperties(NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID, false, true)},
      {"lossless_hp",
       PresetProperties(NV_ENC_PRESET_LOSSLESS_HP_GUID, false, true)}};

  auto it = preset_guids.find(preset_name);
  if (it != preset_guids.end()) {
    return it->second;
  } else {
    cerr << "Preset " << preset_name << " not found. Using default." << endl;
    return preset_guids.begin()->second;
  }
};

auto ParseResolution = [](const string &res_string, uint32_t &width,
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

template <typename T> T FromString(const string &value) {}

template <> uint32_t FromString(const string &value) {
  stringstream ss;
  ss << value;

  uint32_t ret;
  ss >> ret;
  return ret;
}

template <> uint16_t FromString(const string &value) {
  stringstream ss;
  ss << value;

  uint16_t ret;
  ss >> ret;
  return ret;
}

template <> int FromString(const string &value) {
  stringstream ss;
  ss << value;

  int ret;
  ss >> ret;
  return ret;
}

auto IsSameGuid = [](const GUID &a, const GUID &b) {
  return 0 == memcmp((const void *)&a, (const void *)&b, sizeof(a));
};

string ToString(const GUID &guid) {
  // Codecs;
  if (IsSameGuid(NV_ENC_CODEC_H264_GUID, guid)) {
    return "H.264";
  } else if (IsSameGuid(NV_ENC_CODEC_HEVC_GUID, guid)) {
    return "H.265";
  }
  // Presets;
  else if (IsSameGuid(NV_ENC_PRESET_DEFAULT_GUID, guid)) {
    return "Default";
  } else if (IsSameGuid(NV_ENC_PRESET_HP_GUID, guid)) {
    return "HP";
  } else if (IsSameGuid(NV_ENC_PRESET_HQ_GUID, guid)) {
    return "HQ";
  } else if (IsSameGuid(NV_ENC_PRESET_BD_GUID, guid)) {
    return "BD";
  } else if (IsSameGuid(NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID, guid)) {
    return "LL";
  } else if (IsSameGuid(NV_ENC_PRESET_LOW_LATENCY_HQ_GUID, guid)) {
    return "LLHQ";
  } else if (IsSameGuid(NV_ENC_PRESET_LOW_LATENCY_HP_GUID, guid)) {
    return "LLHP";
  } else if (IsSameGuid(NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID, guid)) {
    return "Default";
  } else if (IsSameGuid(NV_ENC_PRESET_LOSSLESS_HP_GUID, guid)) {
    return "Lossless";
  }
  // Profiles;
  else if (IsSameGuid(NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID, guid)) {
    return "Auto";
  } else if (IsSameGuid(NV_ENC_H264_PROFILE_BASELINE_GUID, guid)) {
    return "Baseline";
  } else if (IsSameGuid(NV_ENC_H264_PROFILE_MAIN_GUID, guid)) {
    return "Main";
  } else if (IsSameGuid(NV_ENC_H264_PROFILE_HIGH_GUID, guid)) {
    return "High";
  } else if (IsSameGuid(NV_ENC_H264_PROFILE_HIGH_444_GUID, guid)) {
    return "High YUV444";
  } else if (IsSameGuid(NV_ENC_H264_PROFILE_STEREO_GUID, guid)) {
    return "Stereo";
  } else if (IsSameGuid(NV_ENC_H264_PROFILE_SVC_TEMPORAL_SCALABILTY, guid)) {
    return "SVC";
  } else if (IsSameGuid(NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID, guid)) {
    return "Progressive High";
  } else if (IsSameGuid(NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID, guid)) {
    return "Constrained high";
  } else if (IsSameGuid(NV_ENC_HEVC_PROFILE_MAIN_GUID, guid)) {
    return "HEVC Main";
  } else if (IsSameGuid(NV_ENC_HEVC_PROFILE_MAIN10_GUID, guid)) {
    return "HEVC Main 10 bit";
  } else if (IsSameGuid(NV_ENC_HEVC_PROFILE_FREXT_GUID, guid)) {
    return "HEVC YUV444";
  }
  // Default;
  else {
    return "";
  }
}

void PrintNvEncInitializeParams(const NV_ENC_INITIALIZE_PARAMS &params) {
  cout << "NV_ENC_INITIALIZE_PARAMS:         " << endl;
  cout << " version:                         " << params.version << endl;
  cout << " encodeGUID:                      " << ToString(params.encodeGUID)
       << endl;
  cout << " presetGUID:                      " << ToString(params.presetGUID)
       << endl;
  cout << " encodeWidth:                     " << params.encodeWidth << endl;
  cout << " encodeHeight:                    " << params.encodeHeight << endl;
  cout << " darWidth:                        " << params.darWidth << endl;
  cout << " darHeight:                       " << params.darHeight << endl;
  cout << " frameRateNum:                    " << params.frameRateNum << endl;
  cout << " frameRateDen:                    " << params.frameRateDen << endl;
  cout << " enableEncodeAsync:               " << params.enableEncodeAsync
       << endl;
  cout << " enablePTD:                       " << params.enablePTD << endl;
  cout << " reportSliceOffsets:              " << params.reportSliceOffsets
       << endl;
  cout << " enableSubFrameWrite:             " << params.enableSubFrameWrite
       << endl;
  cout << " enableExternalMEHints:           " << params.enableExternalMEHints
       << endl;
  cout << " enableMEOnlyMode:                " << params.enableMEOnlyMode
       << endl;
  cout << " enableWeightedPrediction:        "
       << params.enableWeightedPrediction << endl;
  cout << " enableOutputInVidmem:            " << params.enableOutputInVidmem
       << endl;
  cout << " maxEncodeWidth:                  " << params.maxEncodeWidth << endl;
  cout << " maxEncodeHeight:                 " << params.maxEncodeHeight << endl
       << endl;
}

void NvEncoderClInterface::SetupInitParams(NV_ENC_INITIALIZE_PARAMS &params,
                                           bool is_reconfigure,
                                           NV_ENCODE_API_FUNCTION_LIST api_func,
                                           void *encoder,
                                           bool print_settings) const {
  if (!is_reconfigure) {
    auto enc_config = params.encodeConfig;
    memset(&params, 0, sizeof(params));
    params.encodeConfig = enc_config;

    // Setup default values;
    params.encodeConfig->version = NV_ENC_CONFIG_VER;
    params.version = NV_ENC_INITIALIZE_PARAMS_VER;
    params.frameRateNum = 30;
    params.frameRateDen = 1;
  }

  // Codec;
  auto codec = FindAttribute(options, "codec");
  if (!codec.empty()) {
    params.encodeGUID = FindCodecGuid(codec);
  }
  ParentParams parent_params = {0};
  parent_params.codec_guid = params.encodeGUID;

  // Preset;
  auto preset = FindAttribute(options, "preset");
  if (!preset.empty()) {
    auto props = FindPresetProperties(preset);
    params.presetGUID = props.preset_guid;
    parent_params.is_lossless = props.is_lossless;
    parent_params.is_low_latency = props.is_low_latency;
  }

  // Resolution;
  auto resolution = FindAttribute(options, "s");
  if (!resolution.empty()) {
    uint32_t width = 0U, height = 0U;
    ParseResolution(resolution, width, height);
    params.encodeWidth = width;
    params.encodeHeight = height;
    params.darWidth = params.encodeWidth;
    params.darHeight = params.encodeHeight;
    params.maxEncodeWidth = params.encodeWidth;
    params.maxEncodeHeight = params.encodeHeight;
  }

  // FPS;
  auto fps = FindAttribute(options, "fps");
  if (!fps.empty()) {
    params.frameRateNum = FromString<uint32_t>(fps);
    params.frameRateDen = 1;
  }

    // Async mode capability;
#if defined(_WIN32)
  if (!params.enableOutputInVidmem) {
    params.enableEncodeAsync = GetCapabilityValue(
        params.encodeGUID, NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT, api_func, encoder);
  }
#endif

  // Rest isn't implemented so far, set up as fixed values;
  if (!is_reconfigure) {
    params.enablePTD = 1;
    params.reportSliceOffsets = 0;
    params.enableSubFrameWrite = 0;
    params.enableMEOnlyMode = false;
    params.enableOutputInVidmem = false;
  }

  // Load configuration from preset;
  if (!preset.empty()) {
    NV_ENC_PRESET_CONFIG preset_config = {NV_ENC_PRESET_CONFIG_VER,
                                          {NV_ENC_CONFIG_VER}};

    auto status = api_func.nvEncGetEncodePresetConfig(
        encoder, params.encodeGUID, params.presetGUID, &preset_config);
    if (NV_ENC_SUCCESS != status) {
      stringstream ss;
      ss << "Failed to get preset configuration. Error code " << status << endl;
      throw runtime_error(ss.str());
    }

    memcpy(params.encodeConfig, &preset_config.presetCfg,
           sizeof(preset_config.presetCfg));
  }

  SetupEncConfig(*params.encodeConfig, parent_params, is_reconfigure,
                 print_settings);

  if (print_settings) {
    PrintNvEncInitializeParams(params);
  }
}

void PrintNvEncConfig(const NV_ENC_CONFIG &config) {
  cout << "NV_ENC_CONFIG:                    " << endl;
  cout << " version:                         " << config.version << endl;
  cout << " profileGUID:                     " << ToString(config.profileGUID)
       << endl;
  cout << " gopLength:                       " << config.gopLength << endl;
  cout << " frameIntervalP:                  " << config.frameIntervalP << endl;
  cout << " monoChromeEncoding:              " << config.monoChromeEncoding
       << endl;
  cout << " frameFieldMode:                  " << config.frameFieldMode << endl;
  cout << " mvPrecision:                     " << config.mvPrecision << endl
       << endl;
}



void NvEncoderClInterface::SetupEncConfig(NV_ENC_CONFIG &config,
                                          ParentParams &parent_params,
                                          bool is_reconfigure,
                                          bool print_settings) const {
  if (!is_reconfigure) {
    config.frameIntervalP = 1;
    config.gopLength = NVENC_INFINITE_GOPLENGTH;
    config.profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
  }

  // Consequtive B frames number;
  auto b_frames = FindAttribute(options, "bf");
  if (!b_frames.empty()) {
    config.frameIntervalP = FromString<int>(b_frames);
  }

  // GOP size;
  auto gop_size = FindAttribute(options, "gop");
  if (!gop_size.empty()) {
    config.gopLength = FromString<uint32_t>(gop_size);
  }

  SetupRateControl(config.rcParams, parent_params, is_reconfigure,
                   print_settings);

  parent_params.gop_length = config.gopLength;
  if (IsSameGuid(NV_ENC_CODEC_H264_GUID, parent_params.codec_guid)) {
    SetupH264Config(config.encodeCodecConfig.h264Config, parent_params,
                    is_reconfigure, print_settings);
  } else if (IsSameGuid(NV_ENC_CODEC_HEVC_GUID, parent_params.codec_guid)) {
    SetupHEVCConfig(config.encodeCodecConfig.hevcConfig, parent_params,
                    is_reconfigure, print_settings);
  }
  else
  {
    throw invalid_argument(
        "Invalid codec given. Choose between h.264 and hevc");
  }

  if (print_settings) {
    PrintNvEncConfig(config);
  }
}

auto FindRcMode = [](const string &rc_name) {
  static const map<string, NV_ENC_PARAMS_RC_MODE> rc_modes = {
      {"constqp", NV_ENC_PARAMS_RC_CONSTQP},
      {"vbr", NV_ENC_PARAMS_RC_VBR},
      {"cbr", NV_ENC_PARAMS_RC_CBR},
      {"cbr_ll_hq", NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ},
      {"cbr_hq", NV_ENC_PARAMS_RC_CBR_HQ},
      {"vbr_hq", NV_ENC_PARAMS_RC_VBR_HQ}};

  auto it = rc_modes.find(rc_name);
  if (it != rc_modes.end()) {
    return it->second;
  } else {
    cerr << "Invalid RC mode given. Using cbr as default";
    return NV_ENC_PARAMS_RC_CBR;
  }
};

auto ParseBitrate = [](const string &br_value) {
  static const uint32_t default_value = 10000000U;

  try {
    // Find 'k', 'K', 'm', 'M' suffix;
    auto it = br_value.rbegin();
    auto suffix = *it;
    uint32_t multiplier = 1U;
    if ('K' == suffix || 'k' == suffix) {
      /* Byte doesn't belong to System International so here
       * we follow JEDEC 100B.01 standard which defines
       * kilobyte as 1024 bytes and megabyte as 1024 kilobytes; */
      multiplier = 1024U;
    } else if ('M' == suffix || 'm' == suffix) {
      multiplier = 1024U * 1024U;
    }

    // Value without suffix;
    auto numerical_value = (multiplier > 1)
                               ? string(br_value.begin(), br_value.end() - 1)
                               : string(br_value.begin(), br_value.end());

    // Compose into result value;
    stringstream ss;
    ss << numerical_value;
    uint32_t res;
    ss >> res;
    return res * multiplier;
  } catch (...) {
    cerr << "Can't parse bitrate string. Using default value " << default_value
         << endl;
    return default_value;
  }
};

auto ParseQpMode = [](const string &qp_value, NV_ENC_QP &qp_values) {
  auto split = [&](const string &s, char delimiter) {
    stringstream ss(s);
    string token;
    vector<string> vTokens;
    while (getline(ss, token, delimiter)) {
      vTokens.push_back(token);
    }
    return vTokens;
  };

  auto vQp = split(qp_value, ',');
  try {
    if (vQp.size() == 1) {
      auto qp = (unsigned)stoi(vQp[0]);
      qp_values = {qp, qp, qp};
    } else if (vQp.size() == 3) {
      qp_values = {(unsigned)stoi(vQp[0]), (unsigned)stoi(vQp[1]),
                   (unsigned)stoi(vQp[2])};
    } else {
      cerr << " qp_for_P_B_I or qp_P,qp_B,qp_I (no space is allowed)" << endl;
    }
  } catch (...) {
  }
};

void PrintNvEncRcParams(const NV_ENC_RC_PARAMS &params) {
  cout << "NV_ENC_RC_PARAMS:                 " << endl;
  cout << " version:                         " << params.version << endl;
  cout << " rateControlMode:                 " << params.rateControlMode
       << endl;
  cout << " constQP:                         " << params.constQP.qpInterP
       << ", " << params.constQP.qpInterB << ", " << params.constQP.qpIntra
       << endl;
  cout << " averageBitRate:                  " << params.averageBitRate << endl;
  cout << " maxBitRate:                      " << params.maxBitRate << endl;
  cout << " vbvBufferSize:                   " << params.vbvBufferSize << endl;
  cout << " vbvInitialDelay:                 " << params.vbvInitialDelay
       << endl;
  cout << " enableMinQP:                     " << params.enableMinQP << endl;
  cout << " enableMaxQP:                     " << params.enableMaxQP << endl;
  cout << " enableInitialRCQP:               " << params.enableInitialRCQP
       << endl;
  cout << " enableAQ:                        " << params.enableAQ << endl;
  cout << " enableLookahead:                 " << params.enableLookahead
       << endl;
  cout << " disableIadapt:                   " << params.disableIadapt << endl;
  cout << " disableBadapt:                   " << params.disableBadapt << endl;
  cout << " enableTemporalAQ:                " << params.enableTemporalAQ
       << endl;
  cout << " zeroReorderDelay:                " << params.zeroReorderDelay
       << endl;
  cout << " enableNonRefP:                   " << params.enableNonRefP << endl;
  cout << " strictGOPTarget:                 " << params.strictGOPTarget
       << endl;
  cout << " aqStrength:                      " << params.aqStrength << endl;
  cout << " minQP:                           " << params.minQP.qpInterP << ", "
       << params.minQP.qpInterB << ", " << params.minQP.qpIntra << endl;
  cout << " maxQP:                           " << params.maxQP.qpInterP << ", "
       << params.maxQP.qpInterB << ", " << params.maxQP.qpIntra << endl;
  cout << " initialRCQP:                     " << params.initialRCQP.qpInterP
       << ", " << params.initialRCQP.qpInterB << ", "
       << params.initialRCQP.qpIntra << endl;
  cout << " targetQuality:                   " << (uint32_t)params.targetQuality
       << endl;
  cout << " targetQualityLSB:                "
       << (uint32_t)params.targetQualityLSB << endl;
  cout << " lookaheadDepth:                  " << params.lookaheadDepth << endl;
  cout << " qpMapMode:                       " << params.qpMapMode << endl
       << endl;
}

void NvEncoderClInterface::SetupRateControl(NV_ENC_RC_PARAMS &params,
                                            ParentParams &parent_params,
                                            bool is_reconfigure,
                                            bool print_settings) const {
  if (!is_reconfigure) {
    memset(&params, 0, sizeof(params));

    /* Set up default RC mode and QP values if we're
     * not in lossless mode; */
    params.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    if (!parent_params.is_lossless) {
      params.constQP = {28, 31, 25};
    }
  }

  // Average bitrate;
  auto avg_br = FindAttribute(options, "bitrate");
  if (!avg_br.empty()) {
    params.averageBitRate = ParseBitrate(avg_br);

    /* If bitrate is explicitly provided, set BRC mode
     * to CBR or LL CBR and override later within this function
     * if BRC is also explicitly set; */
    params.rateControlMode = parent_params.is_low_latency
                                 ? NV_ENC_PARAMS_RC_CBR
                                 : NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ;
  }

  // Max bitrate;
  auto max_br = FindAttribute(options, "maxbitrate");
  if (!max_br.empty()) {
    params.maxBitRate = ParseBitrate(max_br);
  }

  // VBV buffer size;
  auto vbv_buf_size = FindAttribute(options, "vbvbufsize");
  if (!vbv_buf_size.empty()) {
    params.vbvBufferSize = ParseBitrate(vbv_buf_size);
  }

  // VBV initial delay;
  auto vbv_init_size = FindAttribute(options, "vbvinit");
  if (!vbv_init_size.empty()) {
    params.vbvInitialDelay = ParseBitrate(vbv_init_size);
  }

  // Constant Quality mode;
  auto cq_mode = FindAttribute(options, "cq");
  if (!cq_mode.empty()) {
    params.targetQuality = FromString<int>(cq_mode);
    // This is done for purpose;
    params.averageBitRate = 0U;
    params.maxBitRate = 0U;
  }

  // Rate Control mode;
  auto rc_mode = FindAttribute(options, "rc");
  if (!rc_mode.empty()) {
    params.rateControlMode = FindRcMode(rc_mode);
  }

  // Initial QP values;
  auto init_qp = FindAttribute(options, "initqp");
  if (!init_qp.empty()) {
    params.enableInitialRCQP = true;
    ParseQpMode(init_qp, params.initialRCQP);
  }

  // Minimal QP values;
  auto min_qp = FindAttribute(options, "qmin");
  if (!min_qp.empty()) {
    params.enableMinQP = true;
    ParseQpMode(init_qp, params.minQP);
  }

  // Maximum QP values;
  auto max_qp = FindAttribute(options, "qmax");
  if (!max_qp.empty()) {
    params.enableMaxQP = true;
    ParseQpMode(max_qp, params.maxQP);
  }

  // Constant QP values;
  auto const_qp = FindAttribute(options, "constqp");
  if (!const_qp.empty()) {
    ParseQpMode(const_qp, params.constQP);
  }

  // Temporal AQ flag;
  auto temporal_aq = FindAttribute(options, "temporalaq");
  if (!temporal_aq.empty()) {
    params.enableTemporalAQ = true;
  }

  // Look-ahead;
  auto look_ahead = FindAttribute(options, "lookahead");
  if (!look_ahead.empty()) {
    params.lookaheadDepth = FromString<uint16_t>(look_ahead);
    params.enableLookahead = (0U != params.lookaheadDepth);
  }

  // Adaptive Quantization strength;
  auto aq_strength = FindAttribute(options, "aq");
  if (!aq_strength.empty()) {
    params.enableAQ = true;
    params.aqStrength = FromString<uint32_t>(aq_strength);
  }

  if (print_settings) {
    PrintNvEncRcParams(params);
  }
}

#if CHECK_API_VERSION(9, 1)
auto ParseNumRefFrames = [](string &value, NV_ENC_NUM_REF_FRAMES &num_frames) {
  auto num_ref_frames = FromString<uint32_t>(value);
  auto valid_range = num_ref_frames > (int)NV_ENC_NUM_REF_FRAMES_AUTOSELECT;
  valid_range = valid_range && (num_ref_frames < (int)NV_ENC_NUM_REF_FRAMES_7);

  if (valid_range) {
    num_frames = (NV_ENC_NUM_REF_FRAMES)num_ref_frames;
  }
};
#endif

void PrintNvEncH264Config(const NV_ENC_CONFIG_H264 &config) {
  cout << "NV_ENC_CONFIG_H264 :              " << endl;
  cout << " enableStereoMVC:                 " << config.enableStereoMVC
       << endl;
  cout << " hierarchicalPFrames:             " << config.hierarchicalPFrames
       << endl;
  cout << " hierarchicalBFrames:             " << config.hierarchicalBFrames
       << endl;
  cout << " outputBufferingPeriodSEI:        "
       << config.outputBufferingPeriodSEI << endl;
  cout << " outputPictureTimingSEI:          " << config.outputPictureTimingSEI
       << endl;
  cout << " outputAUD:                       " << config.outputAUD << endl;
  cout << " disableSPSPPS:                   " << config.disableSPSPPS << endl;
  cout << " outputFramePackingSEI:           " << config.outputFramePackingSEI
       << endl;
  cout << " outputRecoveryPointSEI:          " << config.outputRecoveryPointSEI
       << endl;
  cout << " enableIntraRefresh:              " << config.enableIntraRefresh
       << endl;
  cout << " enableConstrainedEncoding:       "
       << config.enableConstrainedEncoding << endl;
  cout << " repeatSPSPPS:                    " << config.repeatSPSPPS << endl;
  cout << " enableVFR:                       " << config.enableVFR << endl;
  cout << " enableLTR:                       " << config.enableLTR << endl;
  cout << " qpPrimeYZeroTransformBypassFlag: "
       << config.qpPrimeYZeroTransformBypassFlag << endl;
  cout << " useConstrainedIntraPred:         " << config.useConstrainedIntraPred
       << endl;
#if CHECK_API_VERSION(9, 1)
  cout << " enableFillerDataInsertion:       "
       << config.enableFillerDataInsertion << endl;
#endif
  cout << " level:                           " << config.level << endl;
  cout << " idrPeriod:                       " << config.idrPeriod << endl;
  cout << " separateColourPlaneFlag:         " << config.separateColourPlaneFlag
       << endl;
  cout << " disableDeblockingFilterIDC:      "
       << config.disableDeblockingFilterIDC << endl;
  cout << " numTemporalLayers:               " << config.numTemporalLayers
       << endl;
  cout << " spsId:                           " << config.spsId << endl;
  cout << " ppsId:                           " << config.ppsId << endl;
  cout << " adaptiveTransformMode:           " << config.adaptiveTransformMode
       << endl;
  cout << " fmoMode:                         " << config.fmoMode << endl;
  cout << " bdirectMode:                     " << config.bdirectMode << endl;
  cout << " entropyCodingMode:               " << config.entropyCodingMode
       << endl;
  cout << " stereoMode:                      " << config.stereoMode << endl;
  cout << " intraRefreshPeriod:              " << config.intraRefreshPeriod
       << endl;
  cout << " intraRefreshCnt:                 " << config.intraRefreshCnt
       << endl;
  cout << " maxNumRefFrames:                 " << config.maxNumRefFrames
       << endl;
  cout << " sliceMode:                       " << config.sliceMode << endl;
  cout << " sliceModeData:                   " << config.sliceModeData << endl;
  cout << " ltrNumFrames:                    " << config.ltrNumFrames << endl;
  cout << " ltrTrustMode:                    " << config.ltrTrustMode << endl;
  cout << " chromaFormatIDC:                 " << config.chromaFormatIDC
       << endl;
  cout << " maxTemporalLayers:               " << config.maxTemporalLayers
       << endl;
  cout << " useBFramesAsRef:                 " << config.useBFramesAsRef
       << endl;
#if CHECK_API_VERSION(9, 1)
  cout << " numRefL0:                        " << config.numRefL0 << endl;
  cout << " numRefL1:                        " << config.numRefL1 << endl
       << endl;
#endif
}

void NvEncoderClInterface::SetupH264Config(NV_ENC_CONFIG_H264 &config,
                                           ParentParams &parent_params,
                                           bool is_reconfigure,
                                           bool print_settings) const {
  if (!is_reconfigure) {
    memset(&config, 0, sizeof(config));

    config.sliceMode = 3;
    config.sliceModeData = 1;
    config.chromaFormatIDC = 1;
  }

  config.idrPeriod = parent_params.gop_length;

#if CHECK_API_VERSION(9, 1)
  // IDR period;
  auto idr_period = FindAttribute(options, "idrperiod");
  if (!idr_period.empty()) {
    config.idrPeriod = FromString<uint32_t>(idr_period);
  }

  // Number of reference frames in L0 and L1 lists;
  auto num_ref_l0 = FindAttribute(options, "numrefl0");
  if (!num_ref_l0.empty()) {
    ParseNumRefFrames(num_ref_l0, config.numRefL0);
  }

  auto num_ref_l1 = FindAttribute(options, "numrefl1");
  if (!num_ref_l1.empty()) {
    ParseNumRefFrames(num_ref_l1, config.numRefL1);
  }
#endif

  SetupVuiConfig(config.h264VUIParameters, parent_params, is_reconfigure,
                 print_settings);

  if (print_settings) {
    PrintNvEncH264Config(config);
  }
}

void PrintNvEncConfigHevc(const NV_ENC_CONFIG_HEVC &config) {
  cout << "NV_ENC_CONFIG_HEVC:                 " << endl;
  cout << " level:                             " << config.level << endl;
  cout << " tier:                              " << config.tier << endl;
  cout << " minCUSize:                         " << config.minCUSize << endl;
  cout << " maxCUSize:                         " << config.maxCUSize << endl;
  cout << " useConstrainedIntraPred:           "
       << config.useConstrainedIntraPred << endl;
  cout << " disableDeblockAcrossSliceBoundary: "
       << config.disableDeblockAcrossSliceBoundary << endl;
  cout << " outputBufferingPeriodSEI:          "
       << config.outputBufferingPeriodSEI << endl;
  cout << " outputPictureTimingSEI:            "
       << config.outputPictureTimingSEI << endl;
  cout << " outputAUD:                         " << config.outputAUD << endl;
  cout << " enableLTR:                         " << config.enableLTR << endl;
  cout << " disableSPSPPS:                     " << config.disableSPSPPS
       << endl;
  cout << " repeatSPSPPS:                      " << config.repeatSPSPPS << endl;
  cout << " enableIntraRefresh:                " << config.enableIntraRefresh
       << endl;
  cout << " chromaFormatIDC:                   " << config.chromaFormatIDC
       << endl;
  cout << " pixelBitDepthMinus8:               " << config.pixelBitDepthMinus8
       << endl;
#if CHECK_API_VERSION(9, 1)
  cout << " enableFillerDataInsertion:         "
       << config.enableFillerDataInsertion << endl;
#endif
  cout << " idrPeriod:                         " << config.idrPeriod << endl;
  cout << " intraRefreshPeriod:                " << config.intraRefreshPeriod
       << endl;
  cout << " intraRefreshCnt:                   " << config.intraRefreshCnt
       << endl;
  cout << " maxNumRefFramesInDPB:              " << config.maxNumRefFramesInDPB
       << endl;
  cout << " ltrNumFrames:                      " << config.ltrNumFrames << endl;
  cout << " vpsId:                             " << config.vpsId << endl;
  cout << " spsId:                             " << config.spsId << endl;
  cout << " ppsId:                             " << config.ppsId << endl;
  cout << " sliceMode:                         " << config.sliceMode << endl;
  cout << " sliceModeData:                     " << config.sliceModeData
       << endl;
  cout << " maxTemporalLayersMinus1:           "
       << config.maxTemporalLayersMinus1 << endl;
  cout << " ltrTrustMode:                      " << config.ltrTrustMode << endl;
  cout << " useBFramesAsRef:                   " << config.useBFramesAsRef
       << endl;
#if CHECK_API_VERSION(9, 1)
  cout << " numRefL0:                          " << config.numRefL0 << endl;
  cout << " numRefL1:                          " << config.numRefL1 << endl
       << endl;
#endif
}

void NvEncoderClInterface::SetupHEVCConfig(NV_ENC_CONFIG_HEVC &config,
                                           ParentParams &parent_params,
                                           bool is_reconfigure,
                                           bool print_settings) const {
  if (!is_reconfigure) {
    memset(&config, 0, sizeof(config));

    config.chromaFormatIDC = 1;
  }

  config.idrPeriod = parent_params.gop_length;

#if CHECK_API_VERSION(9, 1)
  // IDR period;
  auto idr_period = FindAttribute(options, "idrperiod");
  if (!idr_period.empty()) {
    config.idrPeriod = FromString<uint32_t>(idr_period);
  }

  // Number of reference frames in L0 and L1 lists;
  auto num_ref_l0 = FindAttribute(options, "numrefl0");
  if (!num_ref_l0.empty()) {
    ParseNumRefFrames(num_ref_l0, config.numRefL0);
  }

  auto num_ref_l1 = FindAttribute(options, "numrefl1");
  if (!num_ref_l1.empty()) {
    ParseNumRefFrames(num_ref_l1, config.numRefL1);
  }
#endif

  SetupVuiConfig(config.hevcVUIParameters, parent_params, is_reconfigure,
                 print_settings);

  if (print_settings) {
    PrintNvEncConfigHevc(config);
  }
}

void PrintNvEncVuiParameters(const NV_ENC_CONFIG_H264_VUI_PARAMETERS &params) {
  cout << "NV_ENC_CONFIG_VUI_PARAMETERS:     " << endl;
  cout << " overscanInfoPresentFlag:         " << params.overscanInfoPresentFlag
       << endl;
  cout << " overscanInfo:                    " << params.overscanInfo << endl;
  cout << " videoSignalTypePresentFlag:      "
       << params.videoSignalTypePresentFlag << endl;
  cout << " videoFormat:                     " << params.videoFormat << endl;
  cout << " videoFullRangeFlag:              " << params.videoFullRangeFlag
       << endl;
  cout << " colourDescriptionPresentFlag:    "
       << params.colourDescriptionPresentFlag << endl;
  cout << " colourPrimaries:                 " << params.colourPrimaries
       << endl;
  cout << " transferCharacteristics:         " << params.transferCharacteristics
       << endl;
  cout << " colourMatrix:                    " << params.colourMatrix << endl;
  cout << " chromaSampleLocationFlag:        "
       << params.chromaSampleLocationFlag << endl;
  cout << " chromaSampleLocationTop:         " << params.chromaSampleLocationTop
       << endl;
  cout << " chromaSampleLocationBot:         " << params.chromaSampleLocationBot
       << endl;
  cout << " bitstreamRestrictionFlag:        "
       << params.bitstreamRestrictionFlag << endl
       << endl;
}

void NvEncoderClInterface::SetupVuiConfig(
    NV_ENC_CONFIG_H264_VUI_PARAMETERS &params, ParentParams &parent_params,
    bool is_reconfigure, bool print_settings) const {

  if (!is_reconfigure) {
    memset(&params, 0, sizeof(params));

    params.videoFormat = 5;
    params.colourPrimaries = 2;
    params.transferCharacteristics = 2;
    params.colourMatrix = 2;
  }

  if (print_settings) {
    PrintNvEncVuiParameters(params);
  }
}

NvEncoderClInterface::NvEncoderClInterface(
    const map<string, string> &cli_options)
    : options(cli_options) {}
