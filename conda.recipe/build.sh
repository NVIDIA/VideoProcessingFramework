# TODO: 
# 1. download video codec sdk from nvidia website(requires login) or s3
# 2. Build / Test for Windows 

export VIDEO_CODEC_SDK_DIR="/home/Video_Codec_SDK_10.0.26"
export GENERATE_PYTORCH_EXTENSION="1"

# ffmpeg
export PATH_TO_FFMPEG="$(which ffmpeg)"
# NOTE: cmake find package is not able to locate ffmpeg libs and includes unless explicitly provided
export FFMPEG_LIB_DIR="$PREFIX/lib"
export FFMPEG_INCLUDE_DIR="$PREFIX/include"

# cuda
export CUDA_LIB_DIR="$PREFIX/lib"
export CUDA_INCLUDE_DIR="$PREFIX/include"

$PYTHON -m pip install . -vvv

mv build/*/lib*.so $PREFIX/lib


