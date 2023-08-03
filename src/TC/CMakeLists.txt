#
# Copyright 2019 NVIDIA Corporation
# Copyright 2023 VisionLabs LLC
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

project(TC LANGUAGES C CXX CUDA)

add_subdirectory(TC_CORE)

find_package(CUDAToolkit 11.2 REQUIRED)
include(GenerateExportHeader)

add_library(
    TC
    src/MemoryInterfaces.cpp
    src/Tasks.cpp
    src/TasksColorCvt.cpp
    src/FFmpegDemuxer.cpp  
    src/NppCommon.cpp
    src/NvCodecCliOptions.cpp
    src/FfmpegDecoder.cpp
)

if (NOT BUILD_TEGRA)
    message(STATUS "**** Linking against VC SDK ****")
    # Only add sources which depend on Video Codec SDK if not on Tegra since  
    # it doesn't support it and relies on V4L2 driver interface instead.
    target_sources(
        TC PRIVATE
        src/NvDecoder.cpp
        src/NvEncoder.cpp
        src/NvEncoderCuda.cpp
        src/TasksCodec.cpp
    )
endif()

if (WIN32)
    target_sources(TC PRIVATE
        src/tc_dlopen_windows.cpp
    )
else()
    target_sources(TC PRIVATE
        src/tc_dlopen_unix.cpp
    )
endif()

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
        "Please set TC_VIDEO_CODEC_INTERFACE_DIR=\"${TC_VIDEO_CODEC_INTERFACE_DIR}\" to a location that contains nvEncodeAPI.h!"
    )
endif()

if(UNIX)
    target_link_libraries(TC PUBLIC pthread)
endif(UNIX)

find_package(PkgConfig)
# pkgconfig works on Windows, but can not handle runtime dependencies
if(PKG_CONFIG_FOUND AND (NOT WIN32)) 
    pkg_check_modules(
        LIBAV
        REQUIRED
        IMPORTED_TARGET
        # versions taken from CI 22.04
        #libavfilter>=7.110.100
        #libavformat>=58.76.100
        #libavcodec>=58.134.100
        #libswresample>=3.9.100
        #libavutil>=56.70.100
        # versions taken from CI. Ubuntu 20.04 system packages
        libavfilter>=7.57.100
        libavformat>=58.29.100
        libavcodec>=58.54.100
        libswresample>=3.5.100
        libavutil>=56.31.100
    )
    target_link_libraries(TC_CORE PUBLIC PkgConfig::LIBAV)
    try_compile(HAS_BSF ${PROJECT_BINARY_DIR} SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/has_bsf_compile_test.c LINK_LIBRARIES PkgConfig::LIBAV)

    if(BUILD_TEGRA)
        message(STATUS "**** Linking against V4L2 ****")
        pkg_check_modules(
            V4L2
            REQUIRED
            IMPORTED_TARGET
            libv4l2>=1.18.0
        )
        target_link_libraries(TC PUBLIC PkgConfig::V4L2)
    endif()
else()
    set(TC_FFMPEG_INCLUDE_DIR "" CACHE PATH "Where to find ffmpeg includes")
    set(TC_FFMPEG_LIBRARY_DIR "" CACHE PATH "Where to find ffmpeg libraries")
    message(STATUS "TC_FFMPEG_INCLUDE_DIR=\"${TC_FFMPEG_INCLUDE_DIR}\"")
    message(STATUS "TC_FFMPEG_ROOT=\"${TC_FFMPEG_INCLUDE_DIR}\"")
    message(STATUS "TC_FFMPEG_LIBRARY_DIR=\"${TC_FFMPEG_LIBRARY_DIR}\"")

    macro(link_av_component target lib_name)
        find_path(
            ${lib_name}_INCLUDE_DIR
            NAMES "${lib_name}/${lib_name}.h" "lib${lib_name}/${lib_name}.h"
            HINTS ${TC_FFMPEG_INCLUDE_DIR} "${TC_FFMPEG_ROOT}/include"
        )
        find_library(
            ${lib_name}_LIBRARY
            NAMES "${lib_name}"
            HINTS ${TC_FFMPEG_LIBRARY_DIR} "${TC_FFMPEG_ROOT}/lib"
        )
        if(NOT ${lib_name}_LIBRARY)
            message(
                FATAL_ERROR
                "Could not find ${lib_name}_LIBRARY! Please set TC_FFMPEG_ROOT to indicate its location!"
            )
        endif()
        if(NOT ${lib_name}_INCLUDE_DIR)
            message(
                FATAL_ERROR
                "Could not find ${lib_name}_INCLUDE_DIR! Please set TC_FFMPEG_ROOT to indicate its location!"
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
    find_path(
        BSF_INCLUDE_DIR
        NAMES "libavcodec/bsf.h"
        HINTS ${TC_FFMPEG_INCLUDE_DIR} "${TC_FFMPEG_ROOT}/include"
    )
    if(BSF_INCLUDE_DIR)
        set(HAS_BSF TRUE)
    else()
        set(HAS_BSF FALSE)
        message(WARNING "Could not find \"libavcodec/bsf.h\" while other ffmpeg includes could be found."
            "This likely means that your FFMPEG installation is old. Still trying to compile VPF!")
    endif()
endif()

if(HAS_BSF)
    target_compile_definitions(TC PUBLIC -DHAS_BSF=1)
endif()
message(STATUS "HAS_BSF=${HAS_BSF}")
