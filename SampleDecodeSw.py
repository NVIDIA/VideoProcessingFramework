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

def decode(encFilePath, decFilePath):
    decFile = open(decFilePath, "wb")
    nvDec = nvc.PyFfmpegDecoder(encFilePath, {'flags2' : '+export_mvs'})
    rawFrameYUV = np.ndarray(shape=(0), dtype=np.uint8)
    motionVectors = np.ndarray(shape=(0), dtype=np.uint8)

    dec_frame = 0
    while (dec_frame < 128):
        success = nvDec.DecodeSingleFrame(rawFrameYUV)
        if not (success):
            print("Frame not decoded")
        else:
            print("Frame decoded") 
            bits = bytearray(rawFrameYUV)
            decFile.write(bits)

        success = nvDec.GetSideData(motionVectors, nvc.FrameSideData.AV_FRAME_DATA_MOTION_VECTORS)
        if success:
            print("Motion vectors extracted. Size: ", motionVectors.size)

        dec_frame += 1

if __name__ == "__main__":

    print("This sample decodes input video to raw NV12 file on given GPU.")
    print("Usage: SampleDecode.py $gpu_id $input_file $output_file")

    if(len(sys.argv) < 3):
        print("Provide path to input and output files")
        exit(1)

    encFilePath = sys.argv[1]
    decFilePath = sys.argv[2]

    decode(encFilePath, decFilePath)