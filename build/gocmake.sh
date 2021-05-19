#!/bin/bash

# Export paths to Video Codec SDK and FFMpeg
export PATH_TO_SDK=/opt/Video_Codec_SDK_11.0.10
export PATH_TO_FFMPEG=/opt/ffmpeg/build_x64_release_shared
export CUDACXX=/usr/local/cuda/bin/nvcc
export INSTALL_PREFIX=$(pwd)/install
cmake .. \
  -DFFMPEG_DIR:PATH="$PATH_TO_FFMPEG" \
  -DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
  -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
  -DGENERATE_PYTORCH_EXTENSION:BOOL="0" \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
  -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES:STRING="75"
