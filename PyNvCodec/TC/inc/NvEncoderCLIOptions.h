/*
 * Copyright 2019 NVIDIA Corporation
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

#pragma once
#include "nvEncodeAPI.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_API_VERSION(major, minor)                                    \
  ((major <  NVENCAPI_MAJOR_VERSION) ||                                    \
  ( major == NVENCAPI_MAJOR_VERSION) && (minor <= NVENCAPI_MINOR_VERSION))

#ifndef _WIN32
inline bool operator==(const GUID &guid1, const GUID &guid2) {
  return !memcmp(&guid1, &guid2, sizeof(GUID));
}

inline bool operator!=(const GUID &guid1, const GUID &guid2) {
  return !(guid1 == guid2);
}
#endif

class NvEncoderInitParam {
public:
  explicit NvEncoderInitParam(
      const char *szParam = "",
      std::function<void(NV_ENC_INITIALIZE_PARAMS *pParams)> *pfuncInit =
          nullptr,
      bool _bLowLatency = false)
      : strParam(szParam), bLowLatency(_bLowLatency) {
    if (pfuncInit) {
      funcInit = *pfuncInit;
    }

    std::transform(strParam.begin(), strParam.end(), strParam.begin(), tolower);
    std::istringstream ss(strParam);
    tokens = std::vector<std::string>{std::istream_iterator<std::string>(ss),
                                      std::istream_iterator<std::string>()};

    for (unsigned i = 0; i < tokens.size(); i++) {
      if (tokens[i] == "-codec" && ++i != tokens.size()) {
        ParseString("-codec", tokens[i], vCodec, szCodecNames, &guidCodec);
        continue;
      }
      if (bLowLatency) {
        if (tokens[i] == "-preset" && ++i != tokens.size()) {
          ParseString("-preset", tokens[i], vLowLatencyPreset,
                      szLowLatencyPresetNames, &guidPreset);
          continue;
        }
      } else {
        if (tokens[i] == "-preset" && ++i != tokens.size()) {
          ParseString("-preset", tokens[i], vPreset, szPresetNames,
                      &guidPreset);
          continue;
        }
      }
    }

    if (bLowLatency)
      guidPreset = NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID;
  }
  virtual ~NvEncoderInitParam() = default;

  virtual bool IsCodecH264() {
    return GetEncodeGUID() == NV_ENC_CODEC_H264_GUID;
  }

  virtual bool IsCodecHEVC() {
    return GetEncodeGUID() == NV_ENC_CODEC_HEVC_GUID;
  }

public:
  virtual GUID GetEncodeGUID() { return guidCodec; }

  virtual GUID GetPresetGUID() { return guidPreset; }

  virtual void SetInitParams(NV_ENC_INITIALIZE_PARAMS *pParams,
                             NV_ENC_BUFFER_FORMAT eBufferFormat) {
    NV_ENC_CONFIG &config = *pParams->encodeConfig;

    auto parseTokens = [](NvEncoderInitParam *pInitParam, NV_ENC_CONFIG &config,
                          NV_ENC_INITIALIZE_PARAMS *pParams, unsigned int &i) {
      if (pInitParam->tokens[i] == "-codec") {
        i++;
        return;
      }

      if (pInitParam->tokens[i] == "-preset") {
        i++;
        return;
      }

      if (pInitParam->tokens[i] == "-profile") {
        i++;
        if (i != pInitParam->tokens.size())
          pInitParam->ParseString(
              "-profile", pInitParam->tokens[i],
              pInitParam->IsCodecH264() ? pInitParam->vH264Profile
                                        : pInitParam->vHevcProfile,
              pInitParam->IsCodecH264() ? pInitParam->szH264ProfileNames
                                        : pInitParam->szHevcProfileNames,
              &config.profileGUID);
        return;
      }

      if (pInitParam->tokens[i] == "-rc") {
        i++;
        if (i != pInitParam->tokens.size()) {
          pInitParam->ParseString(
              "-rc", pInitParam->tokens[i], pInitParam->vRcMode,
              pInitParam->szRcModeNames, &config.rcParams.rateControlMode);
          return;
        }
      }

      if (pInitParam->tokens[i] == "-fps") {
        i++;
        if (i != pInitParam->tokens.size()) {
          pInitParam->ParseInt("-fps", pInitParam->tokens[i],
                               &pParams->frameRateNum);
          return;
        }
      }

      if (pInitParam->tokens[i] == "-bf") {
        i++;
        if (i != pInitParam->tokens.size()) {
          pInitParam->ParseInt("-bf", pInitParam->tokens[i],
                               &config.frameIntervalP);
          ++config.frameIntervalP;
          return;
        }
      }

      if (pInitParam->tokens[i] == "-bitrate") {
        i++;
        if (i != pInitParam->tokens.size()) {
          NvEncoderInitParam::ParseBitRate("-bitrate", pInitParam->tokens[i],
                                           &config.rcParams.averageBitRate);
          return;
        }
      }

      if (pInitParam->tokens[i] == "-maxbitrate") {
        i++;
        if (i != pInitParam->tokens.size()) {
          NvEncoderInitParam::ParseBitRate("-maxbitrate", pInitParam->tokens[i],
                                           &config.rcParams.maxBitRate);
          return;
        }
      }

      if (pInitParam->tokens[i] == "-vbvbufsize") {
        i++;
        if (i != pInitParam->tokens.size()) {
          NvEncoderInitParam::ParseBitRate("-vbvbufsize", pInitParam->tokens[i],
                                           &config.rcParams.vbvBufferSize);
          return;
        }
      }

      if (pInitParam->tokens[i] == "-vbvinit") {
        i++;
        if (i != pInitParam->tokens.size()) {
          NvEncoderInitParam::ParseBitRate("-vbvinit", pInitParam->tokens[i],
                                           &config.rcParams.vbvInitialDelay);
          return;
        }
      }

      if (pInitParam->tokens[i] == "-cq") {
        i++;
        if (i != pInitParam->tokens.size()) {
          pInitParam->ParseInt("-cq", pInitParam->tokens[i],
                               &config.rcParams.targetQuality);
          return;
        }
      }

      if (pInitParam->tokens[i] == "-initqp") {
        i++;
        if (i != pInitParam->tokens.size()) {
          NvEncoderInitParam::ParseQp("-initqp", pInitParam->tokens[i],
                                      &config.rcParams.initialRCQP);
          config.rcParams.enableInitialRCQP = true;
          return;
        }
      }

      if (pInitParam->tokens[i] == "-qmin") {
        i++;
        if (i != pInitParam->tokens.size()) {
          NvEncoderInitParam::ParseQp("-qmin", pInitParam->tokens[i],
                                      &config.rcParams.minQP);
          config.rcParams.enableMinQP = true;
          return;
        }
      }

      if (pInitParam->tokens[i] == "-qmax") {
        i++;
        if (i != pInitParam->tokens.size()) {
          NvEncoderInitParam::ParseQp("-qmax", pInitParam->tokens[i],
                                      &config.rcParams.maxQP);
          config.rcParams.enableMinQP = true;
          return;
        }
      }

      if (pInitParam->tokens[i] == "-constqp") {
        i++;
        if (i != pInitParam->tokens.size()) {
          NvEncoderInitParam::ParseQp("-constqp", pInitParam->tokens[i],
                                      &config.rcParams.constQP);
          return;
        }
      }

      if (pInitParam->tokens[i] == "-temporalaq") {
        config.rcParams.enableTemporalAQ = true;
        return;
      }

      if (pInitParam->tokens[i] == "-lookahead") {
        i++;
        if (i != pInitParam->tokens.size()) {
          pInitParam->ParseInt("-lookahead", pInitParam->tokens[i],
                               &config.rcParams.lookaheadDepth);
          return;
        }
      }

      int aqStrength;
      if (pInitParam->tokens[i] == "-aq") {
        i++;
        if (i != pInitParam->tokens.size()) {
          pInitParam->ParseInt("-aq", pInitParam->tokens[i], &aqStrength);
          config.rcParams.enableAQ = true;
          config.rcParams.aqStrength = aqStrength;
          return;
        }
      }

#if CHECK_API_VERSION(9, 1)
      if (pInitParam->tokens[i] == "-idrperiod") {
        i++;
        if (i != pInitParam->tokens.size()) {
          int idrPeriod;
          pInitParam->ParseInt("-idrperiod", pInitParam->tokens[i], &idrPeriod);
          if (pInitParam->IsCodecH264()) {
            config.encodeCodecConfig.h264Config.idrPeriod = idrPeriod;
          } else {
            config.encodeCodecConfig.hevcConfig.idrPeriod = idrPeriod;
          }
          return;
        }
      }

      if (pInitParam->tokens[i] == "-numrefl0") {
        i++;
        if (i != pInitParam->tokens.size()) {
          int numRefL0;
          pInitParam->ParseInt("-numrefl0", pInitParam->tokens[i], &numRefL0);

          auto validRange = numRefL0 > (int)NV_ENC_NUM_REF_FRAMES_AUTOSELECT;
          validRange = validRange && (numRefL0 < (int)NV_ENC_NUM_REF_FRAMES_7);
          if (!validRange) {
            return;
          }

          if (pInitParam->IsCodecH264()) {
            config.encodeCodecConfig.h264Config.numRefL0 =
                (NV_ENC_NUM_REF_FRAMES)numRefL0;
          } else {
            config.encodeCodecConfig.hevcConfig.numRefL0 =
                (NV_ENC_NUM_REF_FRAMES)numRefL0;
          }
          return;
        }
      }

      if (pInitParam->tokens[i] == "-numrefl1") {
        i++;
        if (i != pInitParam->tokens.size()) {
          int numRefL1;
          pInitParam->ParseInt("-numrefl1", pInitParam->tokens[i], &numRefL1);

          auto validRange = numRefL1 > (int)NV_ENC_NUM_REF_FRAMES_AUTOSELECT;
          validRange = validRange && (numRefL1 < (int)NV_ENC_NUM_REF_FRAMES_7);
          if (!validRange) {
            return;
          }

          if (pInitParam->IsCodecH264()) {
            config.encodeCodecConfig.h264Config.numRefL1 =
                (NV_ENC_NUM_REF_FRAMES)numRefL1;
          } else {
            config.encodeCodecConfig.hevcConfig.numRefL1 =
                (NV_ENC_NUM_REF_FRAMES)numRefL1;
          }
          return;
        }
      }
#endif

      if (pInitParam->tokens[i] == "-gop") {
        i++;
        if (i != pInitParam->tokens.size()) {
          pInitParam->ParseInt("-gop", pInitParam->tokens[i],
                               &config.gopLength);
          if (pInitParam->IsCodecH264()) {
            config.encodeCodecConfig.h264Config.idrPeriod = config.gopLength;
          } else {
            config.encodeCodecConfig.hevcConfig.idrPeriod = config.gopLength;
          }
          return;
        }
      }

      if (pInitParam->tokens[i] == "-444") {
        i++;
        if (i != pInitParam->tokens.size()) {
          if (pInitParam->IsCodecH264()) {
            config.encodeCodecConfig.h264Config.chromaFormatIDC = 3;
          } else {
            config.encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
          }
          return;
        }
      }

      std::ostringstream errmessage;
      errmessage << "Incorrect parameter: " << pInitParam->tokens[i]
                 << std::endl;
      errmessage << "Re-run the application with the -h option to get a list "
                    "of the supported options.";
      errmessage << std::endl;

      throw std::invalid_argument(errmessage.str());
    };

    for (unsigned i = 0; i < tokens.size(); i++) {
      parseTokens(this, config, pParams, i);
    }

    if (IsCodecHEVC()) {
      if (eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT ||
          eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
        config.encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 = 2;
      }
    }

    funcInit(pParams);
  }

private:
  template <typename T>
  bool ParseString(const std::string &strName, const std::string &strValue,
                   const std::vector<T> &vValue,
                   const std::string &strValueNames, T *pValue) {
    std::vector<std::string> vstrValueName = split(strValueNames, ' ');
    auto it = std::find(vstrValueName.begin(), vstrValueName.end(), strValue);
    if (it == vstrValueName.end()) {
      std::cerr << strName << " options: " << strValueNames;
      return false;
    }
    *pValue = vValue[it - vstrValueName.begin()];
    return true;
  }

  static bool ParseBitRate(const std::string &strName,
                           const std::string &strValue, unsigned *pBitRate) {
    try {
      size_t l;
      double r = std::stod(strValue, &l);
      char c = strValue[l];
      if (c != 0 && c != 'k' && c != 'm') {
        std::cerr << strName << " units: 1, K, M (lower case also allowed)";
      }
      *pBitRate = (unsigned)((c == 'm' ? 1000000 : (c == 'k' ? 1000 : 1)) * r);
    } catch (...) {
      return false;
    }
    return true;
  }

  template <typename T>
  bool ParseInt(const std::string &strName, const std::string &strValue,
                T *pInt) {
    try {
      *pInt = std::stoi(strValue);
    } catch (...) {
      std::cerr << strName << " need a value of positive number";
      return false;
    }
    return true;
  }

  static bool ParseQp(const std::string &strName, const std::string &strValue,
                      NV_ENC_QP *pQp) {
    std::vector<std::string> vQp = split(strValue, ',');
    try {
      if (vQp.size() == 1) {
        auto qp = (unsigned)std::stoi(vQp[0]);
        *pQp = {qp, qp, qp};
      } else if (vQp.size() == 3) {
        *pQp = {(unsigned)std::stoi(vQp[0]), (unsigned)std::stoi(vQp[1]),
                (unsigned)std::stoi(vQp[2])};
      } else {
        std::cerr << strName
                  << " qp_for_P_B_I or qp_P,qp_B,qp_I (no space is allowed)";
        return false;
      }
    } catch (...) {
      return false;
    }
    return true;
  }

  static std::vector<std::string> split(const std::string &s, char delimiter) {
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> vTokens;
    while (getline(ss, token, delimiter)) {
      vTokens.push_back(token);
    }
    return vTokens;
  }

private:
  std::string strParam;

  std::function<void(NV_ENC_INITIALIZE_PARAMS *pParams)> funcInit =
      [](NV_ENC_INITIALIZE_PARAMS *pParams) {};

  std::vector<std::string> tokens;
  GUID guidCodec = NV_ENC_CODEC_H264_GUID;
  GUID guidPreset = NV_ENC_PRESET_DEFAULT_GUID;
  bool bLowLatency = false;

  const char *szCodecNames = "h264 hevc";
  std::vector<GUID> vCodec =
      std::vector<GUID>{NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID};

  const char *szPresetNames =
      "default hp hq bd ll ll_hp ll_hq lossless lossless_hp";
  const char *szLowLatencyPresetNames = "ll ll_hp ll_hq";
  std::vector<GUID> vPreset =
      std::vector<GUID>{NV_ENC_PRESET_DEFAULT_GUID,
                        NV_ENC_PRESET_HP_GUID,
                        NV_ENC_PRESET_HQ_GUID,
                        NV_ENC_PRESET_BD_GUID,
                        NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID,
                        NV_ENC_PRESET_LOW_LATENCY_HP_GUID,
                        NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,
                        NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID,
                        NV_ENC_PRESET_LOSSLESS_HP_GUID};

  std::vector<GUID> vLowLatencyPreset = std::vector<GUID>{
      NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID,
      NV_ENC_PRESET_LOW_LATENCY_HP_GUID,
      NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,
  };

  const char *szH264ProfileNames = "baseline main high high444";
  std::vector<GUID> vH264Profile = std::vector<GUID>{
      NV_ENC_H264_PROFILE_BASELINE_GUID,
      NV_ENC_H264_PROFILE_MAIN_GUID,
      NV_ENC_H264_PROFILE_HIGH_GUID,
      NV_ENC_H264_PROFILE_HIGH_444_GUID,
  };
  const char *szHevcProfileNames = "main main10 frext";
  std::vector<GUID> vHevcProfile = std::vector<GUID>{
      NV_ENC_HEVC_PROFILE_MAIN_GUID,
      NV_ENC_HEVC_PROFILE_MAIN10_GUID,
      NV_ENC_HEVC_PROFILE_FREXT_GUID,
  };

  const char *szRcModeNames = "constqp vbr cbr cbr_ll_hq cbr_hq vbr_hq";
  std::vector<NV_ENC_PARAMS_RC_MODE> vRcMode =
      std::vector<NV_ENC_PARAMS_RC_MODE>{
          NV_ENC_PARAMS_RC_CONSTQP, NV_ENC_PARAMS_RC_VBR,
          NV_ENC_PARAMS_RC_CBR,     NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ,
          NV_ENC_PARAMS_RC_CBR_HQ,  NV_ENC_PARAMS_RC_VBR_HQ,
      };
};
