/*
 * Copyright 2019-2020 NVIDIA Corporation
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
#include "TC_CORE.hpp"
#include "nvEncodeAPI.h"
#include <list>
#include <map>
#include <string>

#define CHECK_API_VERSION(major, minor)                                        \
  ((major < NVENCAPI_MAJOR_VERSION) ||                                         \
   (major == NVENCAPI_MAJOR_VERSION) && (minor <= NVENCAPI_MINOR_VERSION))

extern "C" {
struct AVDictionary;
}

namespace VPF
{
DllExport std::map<std::string, std::string> GetNvencInitParams();

class DllExport NvEncoderClInterface
{
public:
  explicit NvEncoderClInterface(const std::map<std::string, std::string>&);
  ~NvEncoderClInterface() = default;

  // Will setup the parameters from CLI arguments;
  void SetupInitParams(NV_ENC_INITIALIZE_PARAMS& params, bool is_reconfigure,
                       NV_ENCODE_API_FUNCTION_LIST api_func, void* encoder,
                       std::map<NV_ENC_CAPS, int>& capabilities,
                       bool print_settings = true) const;

private:
  void SetupEncConfig(NV_ENC_CONFIG& config, struct ParentParams& params,
                      bool is_reconfigure, bool print_settings) const;

  void SetupRateControl(NV_ENC_RC_PARAMS& params,
                        struct ParentParams& parent_params, bool is_reconfigure,
                        bool print_settings) const;

  void SetupH264Config(NV_ENC_CONFIG_H264& config, struct ParentParams& params,
                       bool is_reconfigure, bool print_settings) const;

  void SetupHEVCConfig(NV_ENC_CONFIG_HEVC& config, struct ParentParams& params,
                       bool is_reconfigure, bool print_settings) const;

  // H.264 and H.265 has exactly same VUI parameters config;
  void SetupVuiConfig(NV_ENC_CONFIG_H264_VUI_PARAMETERS& params,
                      struct ParentParams& parent_params, bool is_reconfigure,
                      bool print_settings) const;

  std::map<std::string, std::string> options;
};

class DllExport NvDecoderClInterface
{
public:
  explicit NvDecoderClInterface(const std::map<std::string, std::string>&);
  ~NvDecoderClInterface();

  AVDictionary* GetOptions();

  uint32_t GetNumSideDataEntries();

private:
  struct NvDecoderClInterface_Impl* pImpl = nullptr;
};
} // namespace VPF
