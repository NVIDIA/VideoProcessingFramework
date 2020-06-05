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

project(TESTS)

set(GENERATE_GTESTS FALSE CACHE BOOL "Generate GoogleTest tests")

if(GENERATE_GTESTS)
	# Find Gtest;
	set(GTEST_DIR "" CACHE PATH "Path to GoogleTest")

	set (GTEST_INCLUDE_DIR ${GTEST_DIR}/include)
	if(WIN32)
		set (GTEST_LIBRARY ${GTEST_DIR}/lib/gtest.lib)
		set (GTEST_MAIN_LIBRARY ${GTEST_DIR}/lib/gtest_main.lib)
	endif(WIN32)
	find_package(GTest REQUIRED)

	# Tests sources;
	set (TESTS_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/test_Main.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/test_Task.cpp
	)

	# Add target;
	include_directories(${GTEST_INCLUDE_DIR})
	include_directories(${TC_CORE_INC_PATH})
	add_executable(TESTS ${TESTS_SOURCES})
	target_link_libraries(TESTS GTest::GTest)
	target_link_libraries(TESTS TC_CORE)

	# Add tests;
	add_test(TESTS TESTS)

endif(GENERATE_GTESTS)


#Promote variables to parent scope;
set (GENERATE_GTESTS ${GENERATE_GTESTS} PARENT_SCOPE)
