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

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
import numpy as np
from enum import Enum
import PyNvCodec as nvc
import sys
import os
import logging

logger = logging.getLogger(__file__)

if os.name == "nt":
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        logger.error("CUDA_PATH environment variable is not set.")
        logger.error("Can't set CUDA DLLs search path.")
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(";")
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        logger.error("PATH environment variable is not set.")
        exit(1)


total_num_frames = 444


def encode(gpuID, decFilePath, encFilePath, width, height):
    decFile = open(decFilePath, "rb")
    encFile = open(encFilePath, "wb")
    res = str(width) + 'x' + str(height)

    nvEnc = nvc.PyNvEncoder({'preset': 'P5', 'tuning_info': 'high_quality', 'codec': 'h264',
                             'profile': 'high', 's': res, 'bitrate': '10M'}, gpuID)

    nv12FrameSize = int(nvEnc.Width() * nvEnc.Height() * 3 / 2)
    encFrame = np.ndarray(shape=(0), dtype=np.uint8)

    #Number of frames we've sent to encoder
    framesSent = 0
    #Number of frames we've received from encoder
    framesReceived = 0
    #Number of frames we've got from encoder during flush.
    #This number is included in number of received frames.
    #We use separate counter to check if encoder receives packets one by one
    #during flush.
    framesFlushed = 0

    while (framesSent < total_num_frames):
        rawFrame = np.fromfile(decFile, np.uint8, count=nv12FrameSize)
        if not (rawFrame.size):
            print('No more input frames')
            break

        success = nvEnc.EncodeSingleFrame(rawFrame, encFrame, sync=False)
        framesSent += 1

        if(success):
            encByteArray = bytearray(encFrame)
            encFile.write(encByteArray)
            framesReceived += 1

    #Encoder is asynchronous, so we need to flush it
    while True:
        success = nvEnc.FlushSinglePacket(encFrame)
        if (success) and (framesReceived < total_num_frames):
            encByteArray = bytearray(encFrame)
            encFile.write(encByteArray)
            framesReceived += 1
            framesFlushed += 1
        else:
            break

    print(framesReceived, '/', total_num_frames,
          ' frames encoded and written to output file.')
    print(framesFlushed, ' frame(s) received during encoder flush.')


if __name__ == "__main__":

    print("This sample encodes first ", total_num_frames,
          " frames of input raw NV12 file to H.264 video on given GPU.")
    print("It reconfigures encoder on-the fly to illustrate bitrate change, IDR frame force and encoder reset.")
    print("Usage: SampleEncode.py $gpu_id $input_file $output_file $width $height")

    if(len(sys.argv) < 6):
        print("Provide gpu ID, path to input and output files, width and height")
        exit(1)

    gpuID = int(sys.argv[1])
    decFilePath = sys.argv[2]
    encFilePath = sys.argv[3]
    width = sys.argv[4]
    heihgt = sys.argv[5]

    encode(gpuID, decFilePath, encFilePath, width, heihgt)

    exit(0)
