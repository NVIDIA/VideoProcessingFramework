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

set(TC_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/MemoryInterfaces.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Tasks.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/TasksColorCvt.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/FFmpegDemuxer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/NvDecoder.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/NvEncoder.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/NvEncoderCuda.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/NppCommon.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/NvCodecCliOptions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FfmpegSwDecoder.cpp
	PARENT_SCOPE
)