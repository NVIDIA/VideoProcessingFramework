#!/usr/bin/bash
#
# @sandhawalia (Harsimrat Singh Sandhwalia)
# Switching between --build-args passed from docker-compose
#

if [ "$GEN_PYTORCH_EXT" = "1" ] ; then
    # Extract PyTorch installatin
    make -f docker/Makefile build_env EXTRAS=vpf-pytorch
elif [ "$GEN_OPENGL_EXT" = "1" ] ; then
    # pycuda doesn't build with OpenGL support by default have to build form source
    make -f docker/Makefile build_env EXTRAS=vpf-opengl && make -f docker/Makefile pycuda_built PYTHON_BINARY="$PYTHON_BINARY"
else
    # Barebone VPF dependencies
    make -f docker/Makefile build_env EXTRAS=vpf
fi
