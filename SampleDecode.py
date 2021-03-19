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

def decode(gpuID, encFilePath, decFilePath):
    decFile = open(decFilePath, "wb")
    nvDec = nvc.PyNvDecoder(encFilePath, gpuID)

    #Amount of memory in RAM we need to store decoded frame
    frameSize = nvDec.Framesize()
    rawFrameNV12 = np.ndarray(shape=(frameSize), dtype=np.uint8)

    nvDec.DecodeSingleFrame(rawFrameNV12)

    while True:
        try:
            success = nvDec.DecodeSingleFrame(rawFrameNV12)
            if not (success):
                print('No more video frames.')
                break
    
            bits = bytearray(rawFrameNV12)
            decFile.write(bits)

        except nvc.HwResetException:
            print('Continue after HW decoder was reset')
            continue

if __name__ == "__main__":

    print("This sample decodes input video to raw NV12 file on given GPU.")
    print("Usage: SampleDecode.py $gpu_id $input_file $output_file.")

    if(len(sys.argv) < 4):
        print("Provide gpu ID, path to input and output files")
        exit(1)

    gpuID = int(sys.argv[1])
    encFilePath = sys.argv[2]
    decFilePath = sys.argv[3]

    decode(gpuID, encFilePath, decFilePath)