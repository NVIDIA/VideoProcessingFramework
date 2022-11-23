cmake_minimum_required(VERSION 3.17)

project(Video_Processing_Framework)

option(TRACK_TOKEN_ALLOCATIONS "Debug memory allocations within VPF" FALSE )

if(TRACK_TOKEN_ALLOCATIONS)
	add_definitions(-DTRACK_TOKEN_ALLOCATIONS)
endif(TRACK_TOKEN_ALLOCATIONS)

#Undef MIN & MAX & C runtime warnings for Windows
if(WIN32)
	add_definitions(-DNOMINMAX)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

add_subdirectory(PyNvCodec)
# TODO: rename TC to NvCodec or something more descriptive
add_subdirectory(TC)
#add_subdirectory(PytorchNvCodec)
