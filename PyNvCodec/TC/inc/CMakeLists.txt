#
# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set(TC_HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/MemoryInterfaces.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/CodecsSupport.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/Tasks.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/Version.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/FFmpegDemuxer.h
	${CMAKE_CURRENT_SOURCE_DIR}/NvCodecUtils.h
	${CMAKE_CURRENT_SOURCE_DIR}/NvDecoder.h
	${CMAKE_CURRENT_SOURCE_DIR}/NvEncoder.h
	${CMAKE_CURRENT_SOURCE_DIR}/NvCodecCLIOptions.h
	${CMAKE_CURRENT_SOURCE_DIR}/NvEncoderCuda.h
	${CMAKE_CURRENT_SOURCE_DIR}/NppCommon.hpp
	PARENT_SCOPE
)

set (TC_INC_PATH ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)