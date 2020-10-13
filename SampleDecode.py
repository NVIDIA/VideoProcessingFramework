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

    nvDmx = nvc.PyFFmpegDemuxer(encFilePath)
    nvDec = nvc.PyNvDecoder(nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), gpuID)

    packet = np.ndarray(shape=(0), dtype=np.uint8)
    frameSize = int(nvDmx.Width() * nvDmx.Height() * 3 / 2)
    rawFrame = np.ndarray(shape=(frameSize), dtype=np.uint8)

    while True:
        # Demuxer has sync design, it returns packet every time it's called.
        # If demuxer can't return packet it usually means EOF.
        if not nvDmx.DemuxSinglePacket(packet):
            break

        # Decoder is async by design.
        # As it consumes packets from demuxer one at a time it may not return
        # decoded surface every time the decoding function is called.
        if nvDec.DecodeFrameFromPacket(rawFrame, packet):
            bits = bytearray(rawFrame)
            decFile.write(bits)

    # Now we flush decoder to emtpy decoded frames queue.
    while True:
        if nvDec.FlushSingleFrame(rawFrame):
            bits = bytearray(rawFrame)
            decFile.write(bits)
        else:
            break
    
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