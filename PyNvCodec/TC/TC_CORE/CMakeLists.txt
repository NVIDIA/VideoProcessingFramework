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

project(TC_CORE)

#Do version stuff;
set (TC_CORE_VERSION_MAJOR 1)
set (TC_CORE_VERSION_MINOR 0)

set (inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/inc)
set (src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)

configure_file(
	"${inc_dir}/Version.hpp.in"
	"${inc_dir}/Version.hpp"
)

add_library(TC_CORE STATIC
	${src_dir}/Task.cpp
	${src_dir}/Token.cpp
)
target_include_directories(TC_CORE
	PUBLIC ${inc_dir}
)

set_property(TARGET TC_CORE PROPERTY CUDA_ARCHITECTURES 52 53 60 61 62 70 72 75)

install(FILES $<TARGET_FILE:TC_CORE>									DESTINATION bin)