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

project(TC_CORE VERSION 1.0.0 LANGUAGES CXX)
include(GenerateExportHeader)

set(TC_CORE_VERSION_MAJOR 1)
set(TC_CORE_VERSION_MINOR 0)

configure_file(inc/Version.hpp.in tc_core_version.h)

add_library(TC_CORE src/Task.cpp src/Token.cpp)
target_include_directories(TC_CORE PUBLIC inc ${CMAKE_CURRENT_BINARY_DIR})

generate_export_header(TC_CORE)
target_compile_features(TC_CORE PRIVATE cxx_std_17)
set_property(
    TARGET TC_CORE
    PROPERTY
        # required for shared Python modules in case library is build statically on Unix
        POSITION_INDEPENDENT_CODE
        ON
)

#install(TARGETS TC_CORE) # must be installed when shared library
