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

project(PyNvCodec)
add_subdirectory(TC)

#Add src & inc directories;
set (src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_subdirectory(${src_dir})
add_subdirectory(${inc_dir})

#Do version stuff;
set (PYNVCODEC_VERSION_MAJOR 1)
set (PYNVCODEC_VERSION_MINOR 0)

configure_file(
	"${inc_dir}/Version.hpp.in"
	"${inc_dir}/Version.hpp"
)

#Add TC, TC_CORE and avutil includes;
include_directories(${AVUTIL_INCLUDE_DIR})
include_directories(${TC_CORE_INC_PATH})
include_directories(${TC_INC_PATH})
include_directories(${VIDEO_CODEC_SDK_INCLUDE_DIR})
include_directories(${inc_dir})

#Set up python bindings;
set(GENERATE_PYTHON_BINDINGS FALSE CACHE BOOL "Generate VPF Python bindings")

if(GENERATE_PYTHON_BINDINGS)
	find_package(PythonLibs 3.5 REQUIRED)
	if(PYTHONLIBS_FOUND)
		#Adding pybind11 stuff;
		add_subdirectory(pybind11-2.1.1)

		#Add target;
		pybind11_add_module(PyNvCodec SHARED ${PYNVCODEC_SOURCES} ${PYNVCODEC_HEADERS})
		set_property(TARGET PyNvCodec PROPERTY CXX_STANDARD 14)
		
		#Link libs;
		target_link_libraries(PyNvCodec PUBLIC TC_CORE TC ${PYTHON_LIBRARIES} ${NVCUVID_LIBRARY} ${NVENCODE_LIBRARY})
	endif(PYTHONLIBS_FOUND)
endif(GENERATE_PYTHON_BINDINGS)


#Promote variables to parent scope;
set (TC_CORE_INC_PATH			${TC_CORE_INC_PATH}				PARENT_SCOPE)
set (TC_INC_PATH				${TC_INC_PATH}					PARENT_SCOPE)
        
set (AVUTIL_INCLUDE_DIR			${AVUTIL_INCLUDE_DIR}			PARENT_SCOPE)
set (AVCODEC_INCLUDE_DIR		${AVCODEC_INCLUDE_DIR}			PARENT_SCOPE)
set (AVFORMAT_INCLUDE_DIR		${AVFORMAT_INCLUDE_DIR}			PARENT_SCOPE)
        
set (FFMPEG_DIR					${FFMPEG_DIR}					PARENT_SCOPE)
set (FFMPEG_INCLUDE_DIR			${FFMPEG_INCLUDE_DIR}			PARENT_SCOPE)
set (FFMPEG_LIB_DIR				${FFMPEG_LIB_DIR}				PARENT_SCOPE)
set (AVUTIL_LIBRARY				${AVUTIL_LIBRARY}				PARENT_SCOPE)
set (AVCODEC_LIBRARY			${AVCODEC_LIBRARY}				PARENT_SCOPE)
set (AVFORMAT_LIBRARY			${AVFORMAT_LIBRARY}				PARENT_SCOPE)
set (SWRESAMPLE_LIBRARY			${SWRESAMPLE_LIBRARY}			PARENT_SCOPE)

set (GENERATE_PYTHON_BINDINGS	${GENERATE_PYTHON_BINDINGS}		PARENT_SCOPE)
