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

set(PYNVCODEC_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/PyNvCodec.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PyFrameUploader.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PyBufferUploader.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PySurfaceDownloader.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PyCudaBufferDownloader.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PySurfaceConverter.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PySurfaceResizer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PySurfaceRemaper.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PyFFMpegDecoder.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PyFFMpegDemuxer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PyNvDecoder.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PyNvEncoder.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PySurface.cpp
	PARENT_SCOPE
)
