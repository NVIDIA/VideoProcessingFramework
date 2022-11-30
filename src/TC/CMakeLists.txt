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

cmake_minimum_required(VERSION 3.20)

# This must be set before enabling CUDA. Otherwise it will be set to the default ARCH by nvcc (e.g. 52 by nvcc 11.7)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Define Volta / Turing / Ampere / Ada if not set by the user
    set(CMAKE_CUDA_ARCHITECTURES
        60
        70
        72
        75
        80
        86
        #89 # Require CUDA 11.8 for CC 89 or cmake 3.23 for CMAKE_CUDA_ARCHITECTURES_{ALL,ALL_MAJOR}
    )
endif()

project(TC LANGUAGES CXX CUDA)

add_subdirectory(TC_CORE)

find_package(CUDAToolkit 11.2 REQUIRED)
include(GenerateExportHeader)

add_library(
    TC
    src/MemoryInterfaces.cpp
    src/Tasks.cpp
    src/TasksColorCvt.cpp
    src/FFmpegDemuxer.cpp
    src/NvDecoder.cpp
    src/NvEncoder.cpp
    src/NvEncoderCuda.cpp
    src/NppCommon.cpp
    src/NvCodecCliOptions.cpp
    src/FfmpegSwDecoder.cpp
    src/Resize.cu
)

set(TC_VERSION_MAJOR 1)
set(TC_VERSION_MINOR 0)

configure_file(inc/Version.hpp.in tc_version.h)

option(USE_NVTX "Use NVTX for profiling" FALSE)
if(USE_NVTX)
    add_definitions(-DUSE_NVTX)
    ## Replace when requiring CMAKE 3.25:
    # target_link_libraries(TC PUBLIC CUDA::nvtx3)
    target_link_libraries(TC PUBLIC ${CMAKE_DL_LIBRARIES})
endif()

generate_export_header(TC)

target_link_libraries(
    TC
    PUBLIC
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::npps
        CUDA::nppig
        CUDA::nppial
        CUDA::nppicc
        CUDA::nppidei
        TC_CORE
)

target_compile_features(TC PRIVATE cxx_std_17)
set_property(
    TARGET TC
    PROPERTY
        # required for shared Python modules in case library is build statically on Unix
        POSITION_INDEPENDENT_CODE
        ON
)

set(TC_VIDEO_CODEC_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    CACHE PATH "Path to Video Codec SDK interface headers"
)
target_include_directories(
    TC
    PUBLIC
        inc
        ${TC_VIDEO_CODEC_INTERFACE_DIR} # TODO: check if it can be made private!
        ${CMAKE_CURRENT_BINARY_DIR}
)
if(NOT EXISTS ${TC_VIDEO_CODEC_INTERFACE_DIR}/nvEncodeAPI.h)
    message(
        FATAL_ERROR
        "Could not find nvEncodeAPI.h! "
        "Please set TC_VIDEO_CODEC_INTERFACE_DIR=\"${VIDEO_CODEC_INTERFACE_DIR}\" to a location that contains nvEncodeAPI.h!"
    )
endif()

if(UNIX)
    target_link_libraries(TC PUBLIC pthread)
endif(UNIX)

# TODO: avoid cuvidParseVideoData! cuvid is deprecated: https://docs.nvidia.com/cuda/video-decoder/index.html
set(TC_NVCUVID_LIBRARY_DIR "" CACHE STRING "Where to find nvcuvid library")
find_library(NVCUVID_LIBRARY NAMES nvcuvid HINTS TC_NVCUVID_LIBRARY_DIR)
if(NOT NVCUVID_LIBRARY)
    message(
        FATAL_ERROR
        "Could not find NVCUVID_LIBRARY! Please set TC_NVCUVID_LIBRARY_DIR to indicate its location!"
    )
endif()
target_link_libraries(TC PUBLIC ${NVCUVID_LIBRARY})

find_package(PkgConfig)
if(PKG_CONFIG_FOUND AND (NOT MSVC))
    pkg_check_modules(
        LIBAV
        REQUIRED
        IMPORTED_TARGET
        libavfilter
        libavformat
        libavcodec
        libswresample
        libavutil
    )
    target_link_libraries(TC_CORE PUBLIC PkgConfig::LIBAV)
else()
    set(TC_FFMPEG_INCLUDE_DIR "" CACHE STRING "Where to find ffmpeg includes")
    set(TC_FFMPEG_LIBRARY_DIR "" CACHE STRING "Where to find ffmpeg libraries")
    macro(link_av_component target lib_name)
        find_path(
            ${lib_name}_INCLUDE_DIR
            NAMES "${lib_name}/${lib_name}.h" "lib${lib_name}/${lib_name}.h"
            HINTS TC_FFMPEG_INCLUDE_DIR
        )
        find_library(
            ${lib_name}_LIBRARY
            NAMES "${lib_name}"
            HINTS TC_FFMPEG_LIBRARY_DIR
        )
        if(NOT ${lib_name}_LIBRARY)
            message(
                FATAL_ERROR
                "Could not find ${lib_name}_LIBRARY! Please set TC_FFMPEG_LIBRARY_DIR to indicate its location!"
            )
        endif()
        if(NOT ${lib_name}_INCLUDE_DIR)
            message(
                FATAL_ERROR
                "Could not find ${lib_name}_INCLUDE_DIR! Please set TC_FFMPEG_INCLUDE_DIR to indicate its location!"
            )
        endif()
        target_link_libraries(TC PUBLIC ${${lib_name}_LIBRARY})
        target_include_directories(TC PUBLIC ${${lib_name}_INCLUDE_DIR})
    endmacro()
    link_av_component(TC avfilter)
    link_av_component(TC avformat)
    link_av_component(TC avcodec)
    link_av_component(TC swresample)
    link_av_component(TC avutil)
endif()
