#
# Copyright 2019 NVIDIA Corporation
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

project(PyNvCodec)
#Add src & inc directories;
set (src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/inc)

#Do version stuff;
set (PYNVCODEC_VERSION_MAJOR 1)
set (PYNVCODEC_VERSION_MINOR 0)

configure_file(
	"${inc_dir}/Version.hpp.in"
	"${inc_dir}/Version.hpp"
)

# add_subdirectory(pybind11-2.1.1)
add_subdirectory(TC)

pybind11_add_module(PyNvCodec 
    src/PyNvCodec.cpp)

target_include_directories(PyNvCodec 
    PUBLIC inc)

target_link_libraries(PyNvCodec
    PUBLIC TC)

target_compile_definitions(PyNvCodec
    PRIVATE GENERATE_PYTHON_BINDINGS)

set_property(TARGET PyNvCodec PROPERTY CUDA_ARCHITECTURES 52 53 60 61 62 70 72 75)

install(FILES $<TARGET_FILE:PyNvCodec>	                                DESTINATION bin)