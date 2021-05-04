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

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
import sys
import os

if os.name == 'nt':
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file = sys.stderr)
        print("Can't set CUDA DLLs search path.", file = sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file = sys.stderr)
        exit(1)

import PyNvCodec as nvc
import numpy as np

total_num_frames = 128

def dump_motion_vectors(mvcFile, nvDec):
    motionVectors = nvDec.GetMotionVectors()
    np.savetxt(mvcFile, motionVectors, delimiter=',')

def decode(encFilePath, decFilePath, mvcFilePath):
    decFile = open(decFilePath, "wb")
    mvcFile = open(mvcFilePath, "wb")

    nvDec = nvc.PyFfmpegDecoder(encFilePath, {'flags2' : '+export_mvs'})
    rawFrameYUV = np.ndarray(shape=(0), dtype=np.uint8)
    
    dec_frame = 0
    while (dec_frame < total_num_frames):
        success = nvDec.DecodeSingleFrame(rawFrameYUV)
        if not (success):
            print("Frame not decoded")
        else:
            print("Frame decoded") 
            bits = bytearray(rawFrameYUV)
            decFile.write(bits)
            dump_motion_vectors(mvcFile, nvDec)

        dec_frame += 1

if __name__ == "__main__":

    print("This sample decodes first ", total_num_frames, " frames from input video to raw YUV420 file using FFmpeg CPU-based decoder.")
    print("It also extracts motion vectors using ffmpeg AVDictionary export_mvs entry")
    print("Usage: SampleDecode.py $input_file $output_file $motion_vectors_file")

    if(len(sys.argv) < 4):
        print("Provide path to input and output files (YUV420 frames and motion vectos)")
        exit(1)

    encFilePath = sys.argv[1]
    decFilePath = sys.argv[2]
    mvcFilePath = sys.argv[3]

    decode(encFilePath, decFilePath, mvcFilePath)