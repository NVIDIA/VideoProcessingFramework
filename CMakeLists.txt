#
# Copyright 2019 NVIDIA Corporation
# Copyright 2021 Kognia Sports Intelligence
# Copyright 2021 Videonetics Technology Private Limited
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

cmake_minimum_required(VERSION 3.17)

project(Video_Processing_Framework LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (WIN32)
set(VIDEO_CODEC_SDK_DIR ${CMAKE_SOURCE_DIR}/Video_Codec_SDK_11.1.5 CACHE PATH "Path to Nvidia Video Codec SDK")
set(FFMPEG_DIR ${CMAKE_SOURCE_DIR}/ffmpeg CACHE PATH "Path to FFMpeg")
endif()

if (DEFINED ENV{PATH_TO_SDK})
    set(VIDEO_CODEC_SDK_DIR $ENV{PATH_TO_SDK})
endif()

set(TRACK_TOKEN_ALLOCATIONS FALSE CACHE BOOL "Debug memory allocations within VPF")

if(TRACK_TOKEN_ALLOCATIONS)
	add_definitions(-DTRACK_TOKEN_ALLOCATIONS)
endif(TRACK_TOKEN_ALLOCATIONS)

#Undef MIN & MAX & C runtime warnings for Windows
if(WIN32)
	add_definitions(-DNOMINMAX)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)
add_subdirectory(PyNvCodec/pybind11-2.1.1)
add_subdirectory(PyNvCodec)
# add_subdirectory(PytorchNvCodec)

#Bring all the stuff together
install(FILES $<TARGET_FILE:TC_CORE>										DESTINATION bin)
install(FILES $<TARGET_FILE:TC>												DESTINATION bin)

if(GENERATE_PYTHON_BINDINGS)
	install(FILES $<TARGET_FILE:PyNvCodec>									DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleDecode.py				DESTINATION bin)	
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleEncode.py				DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleDecodeSw.py				DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleDecodeMultiThread.py	DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleEncodeMultiThread.py	DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleDemuxDecode.py			DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SamplePyTorch.py				DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleTensorRTResnet.py		DESTINATION bin)
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/SampleTorchResnet.py			DESTINATION bin)	
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/Tests.py						DESTINATION bin)
endif(GENERATE_PYTHON_BINDINGS)

