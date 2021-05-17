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

project(TC)
enable_language(CUDA)
add_subdirectory(TC_CORE)

set(USE_NVTX FALSE CACHE BOOL "Use NVTX for profiling")
if(USE_NVTX)
	add_definitions(-DUSE_NVTX)
endif()

set(VIDEO_CODEC_SDK_DIR "" CACHE PATH "Path to Nvidia Video Codec SDK")
set(FFMPEG_DIR "" CACHE PATH "Path to FFMpeg")

if (DEFINED FFMPEG_INCLUDE_DIR)
	set(FFMPEG_INCLUDE_DIR "${FFMPEG_INCLUDE_DIR}" CACHE PATH "Path to FFmpeg includes")
else()
	set(FFMPEG_INCLUDE_DIR "${FFMPEG_DIR}/include")
endif()

if (DEFINED FFMPEG_LIB_DIR)
	set(FFMPEG_LIB_DIR "${FFMPEG_LIB_DIR}" CACHE PATH "Path to FFmpeg libraries")
else()
	set(FFMPEG_LIB_DIR "${FFMPEG_DIR}/lib")
endif()

message(STATUS "Searching for FFmpeg libs in ${FFMPEG_LIB_DIR}")
message(STATUS "Searching for FFmpeg headers in ${FFMPEG_INCLUDE_DIR}")

find_path( AVCODEC_INCLUDE_DIR libavcodec/avcodec.h ${FFMPEG_INCLUDE_DIR})
find_path( AVFORMAT_INCLUDE_DIR libavformat/avformat.h ${FFMPEG_INCLUDE_DIR})
find_path( AVUTIL_INCLUDE_DIR libavutil/avutil.h ${FFMPEG_INCLUDE_DIR})

message(STATUS "Searching for Video Codec SDK headers in ${VIDEO_CODEC_SDK_DIR}/include folder")
find_path( VIDEO_CODEC_SDK_INCLUDE_DIR nvEncodeAPI.h ${VIDEO_CODEC_SDK_DIR}/include)
# Video Codec SDK 10 has headers in folder with different name
if (VIDEO_CODEC_SDK_INCLUDE_DIR-NOTFOUND STREQUAL ${VIDEO_CODEC_SDK_INCLUDE_DIR})
	message(STATUS "Searching for Video Codec SDK headers in ${VIDEO_CODEC_SDK_DIR}/Interface folder")
	find_path( VIDEO_CODEC_SDK_INCLUDE_DIR nvEncodeAPI.h ${VIDEO_CODEC_SDK_DIR}/Interface)
endif()

find_library( SWRESAMPLE_LIBRARY swresample ${FFMPEG_LIB_DIR})
find_library( AVFORMAT_LIBRARY avformat ${FFMPEG_LIB_DIR})
find_library( AVCODEC_LIBRARY avcodec ${FFMPEG_LIB_DIR})
find_library( AVUTIL_LIBRARY avutil ${FFMPEG_LIB_DIR})

if(UNIX)
	find_library( NVCUVID_LIBRARY nvcuvid ${VIDEO_CODEC_SDK_DIR}/Lib/linux/stubs/x86_64)
	find_library( NVENCODE_LIBRARY nvidia-encode ${VIDEO_CODEC_SDK_DIR}/Lib/linux/stubs/x86_64)
elseif(WIN32)
	find_library( NVCUVID_LIBRARY nvcuvid ${VIDEO_CODEC_SDK_DIR}/Lib/x64)
	find_library( NVENCODE_LIBRARY nvencodeapi ${VIDEO_CODEC_SDK_DIR}/Lib/x64)
endif(UNIX)

include_directories(${AVCODEC_INCLUDE_DIR})
include_directories(${AVFORMAT_INCLUDE_DIR})
include_directories(${AVUTIL_INCLUDE_DIR})
include_directories(${VIDEO_CODEC_SDK_INCLUDE_DIR})

#Do version stuff;
set (TC_VERSION_MAJOR 1)
set (TC_VERSION_MINOR 0)

set (inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_subdirectory(${inc_dir})

configure_file(
	"${inc_dir}/Version.hpp.in"
	"${inc_dir}/Version.hpp"
)

#Add library to targets;
set (src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${src_dir})

#Add target;
add_library(TC SHARED ${TC_HEADERS} ${TC_SOURCES})
IF(UNIX)
	set_property(TARGET TC PROPERTY CUDA_STANDARD 11)
endif(UNIX)

#Add TC_CORE & TC headers;
include_directories(${TC_CORE_INC_PATH})
include_directories(${TC_INC_PATH})

#Link against libraries;
if(UNIX)
	target_link_libraries(TC PUBLIC pthread)
endif(UNIX)

target_link_libraries(TC PUBLIC ${NVCUVID_LIBRARY})
target_link_libraries(TC PUBLIC ${NVENCODE_LIBRARY})
target_link_libraries(TC PUBLIC ${AVUTIL_LIBRARY})
target_link_libraries(TC PUBLIC ${AVCODEC_LIBRARY})
target_link_libraries(TC PUBLIC ${AVFORMAT_LIBRARY})
target_link_libraries(TC PUBLIC ${SWRESAMPLE_LIBRARY})
target_link_libraries(TC PUBLIC cudart)
target_link_libraries(TC PUBLIC cuda)
target_link_libraries(TC PUBLIC npps)
target_link_libraries(TC PUBLIC nppig)
target_link_libraries(TC PUBLIC nppicc)
target_link_libraries(TC PUBLIC nppidei)
target_link_libraries(TC PUBLIC TC_CORE)

#Promote variables to parent & global scope;
set (TC_CORE_INC_PATH             ${TC_CORE_INC_PATH}             PARENT_SCOPE)
set (TC_INC_PATH                  ${TC_INC_PATH}                  PARENT_SCOPE)

set (AVUTIL_INCLUDE_DIR           ${AVUTIL_INCLUDE_DIR}           PARENT_SCOPE)
set (AVCODEC_INCLUDE_DIR          ${AVCODEC_INCLUDE_DIR}          PARENT_SCOPE)
set (AVFORMAT_INCLUDE_DIR         ${AVFORMAT_INCLUDE_DIR}         PARENT_SCOPE)

set (FFMPEG_DIR                   ${FFMPEG_DIR}                   PARENT_SCOPE)
set (FFMPEG_INCLUDE_DIR           ${FFMPEG_INCLUDE_DIR}           PARENT_SCOPE)
set (FFMPEG_LIB_DIR               ${FFMPEG_LIB_DIR}               PARENT_SCOPE)
set (AVUTIL_LIBRARY               ${AVUTIL_LIBRARY}               PARENT_SCOPE)
set (AVCODEC_LIBRARY              ${AVCODEC_LIBRARY}              PARENT_SCOPE)
set (AVFORMAT_LIBRARY             ${AVFORMAT_LIBRARY}             PARENT_SCOPE)
set (SWRESAMPLE_LIBRARY           ${SWRESAMPLE_LIBRARY}           PARENT_SCOPE)

set (VIDEO_CODEC_SDK_INCLUDE_DIR  ${VIDEO_CODEC_SDK_INCLUDE_DIR}  PARENT_SCOPE)
set (NVENCODE_LIBRARY             ${NVENCODE_LIBRARY}             PARENT_SCOPE)
set (NVCUVID_LIBRARY              ${NVCUVID_LIBRARY}              PARENT_SCOPE)
