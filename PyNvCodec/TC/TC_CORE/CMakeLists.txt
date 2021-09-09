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

project(TC_CORE)

#Do version stuff;
set (TC_CORE_VERSION_MAJOR 1)
set (TC_CORE_VERSION_MINOR 0)

set (inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_subdirectory(${inc_dir})

configure_file(
	"${inc_dir}/Version.hpp.in"
	"${inc_dir}/Version.hpp"
)

#Add library to targets;
set (src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${src_dir})

add_library(TC_CORE SHARED ${TC_CORE_HEADERS} ${TC_CORE_SOURCES})
include_directories(${TC_CORE_INC_PATH})

set(TC_CORE_INC_PATH ${TC_CORE_INC_PATH} PARENT_SCOPE)