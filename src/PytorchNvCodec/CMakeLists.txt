#
# Copyright 2020 NVIDIA Corporation
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

project(PytorchNvCodec)

#Add src directory;
set (src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)

#Do version stuff;
set (PYTORCH_NVCODEC_VERSION_MAJOR 1)
set (PYTORCH_NVCODEC_VERSION_MINOR 0)

#Set up python bindings;
set(GENERATE_PYTORCH_EXTENSION FALSE CACHE BOOL "Generate VPF Pytorch extension")

if(GENERATE_PYTHON_BINDINGS)
	if(GENERATE_PYTORCH_EXTENSION)
		set(PYTORCH_EXTENSION_SOURCES_DIR ${src_dir} PARENT_SCOPE)
	endif(GENERATE_PYTORCH_EXTENSION)
endif(GENERATE_PYTHON_BINDINGS)


#Promote variables to parent scope;
set (GENERATE_PYTORCH_EXTENSION	${GENERATE_PYTORCH_EXTENSION} PARENT_SCOPE)
