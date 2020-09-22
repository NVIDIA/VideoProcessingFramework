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
import PyNvCodec as nvc
import numpy as np
import sys

total_num_frames = 444

def encode(gpuID, decFilePath, encFilePath, width, height):
    decFile = open(decFilePath, "rb")
    encFile = open(encFilePath, "wb")
    res = width + 'x' + height

    nvEnc = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': res, 'bitrate' : '10M'}, 
        gpuID)

    nv12FrameSize = int(nvEnc.Width() * nvEnc.Height() * 3 / 2)
    encFrame = np.ndarray(shape=(0), dtype=np.uint8)
    
    #Append dummy unregistered SEI message with some Fibonacci numbers
    seiMessage = np.ndarray(shape=(8), dtype=np.uint8)
    seiMessage[0] = 1
    seiMessage[1] = 1
    seiMessage[2] = 2
    seiMessage[3] = 3
    seiMessage[4] = 5
    seiMessage[5] = 8
    seiMessage[6] = 13
    seiMessage[7] = 21

    frameNum = 0
    while (frameNum < total_num_frames):
        # Will only change bitrate.
        if(frameNum == 111):
            nvEnc.Reconfigure({'bitrate' : '15M'})

        # Will change bitrate and force frame #222 to be IDR I-frame.
        if(frameNum == 222):
            nvEnc.Reconfigure({'bitrate' : '20M'}, force_idr = True,)

        # Will change bitrate, reset encoder and print encoder settings to stdout.
        # Encoder reset also forces next frame to be IDR I-frame.
        if(frameNum == 333):
            nvEnc.Reconfigure({'bitrate' : '25M'}, reset_encoder = True, verbose = True)

        rawFrame = np.fromfile(decFile, np.uint8, count = nv12FrameSize)
        if not (rawFrame.size):
            break
    
        success = nvEnc.EncodeSingleFrame(rawFrame, encFrame, seiMessage, sync = False)
        if(success):
            encByteArray = bytearray(encFrame)
            encFile.write(encByteArray)

        frameNum += 1

    #Encoder is asynchronous, so we need to flush it
    success = nvEnc.Flush(encFrame)
    if(success):
        encByteArray = bytearray(encFrame)
        encFile.write(encByteArray)


if __name__ == "__main__":

    print("This sample encodes first ", total_num_frames, " frames of input raw NV12 file to H.264 video on given GPU.")
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