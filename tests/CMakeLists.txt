#
# Copyright 2021 NVIDIA Corporation
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

set(ENABLE_TESTS TRUE CACHE BOOL "Enable unit tests")

set (TEST_SUITES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_PySurface.py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_PyNvDecoder.py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_PyNvEncoder.py
    ${CMAKE_CURRENT_SOURCE_DIR}/test_PyFfmpegDemuxer.py
    PARENT_SCOPE
)

set (TEST_ASSETS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_res_change.h264
    ${CMAKE_CURRENT_SOURCE_DIR}/test.mp4
    PARENT_SCOPE
)

set (TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)