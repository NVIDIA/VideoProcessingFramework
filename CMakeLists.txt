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

cmake_minimum_required(VERSION 3.10)

project(Video_Processing_Framework)

set(TRACK_TOKEN_ALLOCATIONS FALSE CACHE BOOL "Debug memory allocations within VPF")

if(TRACK_TOKEN_ALLOCATIONS)
	add_definitions(-DTRACK_TOKEN_ALLOCATIONS)
endif(TRACK_TOKEN_ALLOCATIONS)

#Undef MIN & MAX & C runtime warnings for Windows
if(WIN32)
	add_definitions(-DNOMINMAX)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

if(UNIX)
	set (CMAKE_CXX_STANDARD 11)
	include_directories(/usr/local/cuda/include)
	link_directories(/usr/local/cuda/lib64)
endif(UNIX)

add_subdirectory(PyNvCodec)

#Add C++ sample
include_directories(${TC_CORE_INC_PATH})
include_directories(${TC_INC_PATH})
include_directories(${VIDEO_CODEC_SDK_INCLUDE_DIR})

#Bring all the stuff together
install(FILES $<TARGET_FILE:TC_CORE>									DESTINATION bin)
install(FILES $<TARGET_FILE:TC>											DESTINATION bin)

if(GENERATE_PYTHON_BINDINGS)
	install(FILES $<TARGET_FILE:PyNvCodec>								DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleDecode.py			DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleEncode.py			DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleTranscode.py		DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleFrameUpload.py		DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleColorConversion.py	DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleSufraceDownload.py	DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleTranscodeOneToN.py	DESTINATION bin)
endif(GENERATE_PYTHON_BINDINGS)