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
import av

def decode(gpuID, encFilePath, decFilePath):
    input_ = av.open(encFilePath)
    in_stream = input_.streams.video[0]
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height

    axbFile = open('AnnexB.h264', "wb")
    decFile = open(decFilePath, "wb")
    nvDec = nvc.PyNvDecoder(width, height, nvc.PixelFormat.NV12, nvc.CudaVideoCodec.H264, gpuID)

    #Amount of memory in RAM we need to store decoded frame
    frameSize = int(width * height * 3 / 2)
    rawFrameNV12 = np.ndarray(shape=(frameSize), dtype=np.uint8)

    for packet in input_.demux(in_stream):
        if packet.dts is None:
            continue

        packet_data = np.frombuffer(packet.to_bytes(), dtype=np.uint8)
        bits = bytearray(packet_data)
        axbFile.write(bits)

        success = nvDec.DecodeFrameFromPacket(rawFrameNV12, packet_data)
        if (success):
            bits = bytearray(rawFrameNV12)
            decFile.write(bits)
    
if __name__ == "__main__":

    #print("This sample decodes input video to raw NV12 file on given GPU.")
    #print("Usage: SampleDecode.py $gpu_id $input_file $output_file.")

    #if(len(sys.argv) < 4):
    #    print("Provide gpu ID, path to input and output files")
    #    exit(1)

    #gpuID = int(sys.argv[1])
    #encFilePath = sys.argv[2]
    #decFilePath = sys.argv[3]

    decode(0, 'big_buck_bunny_1080p_h264.mov', 'bbb.nv12')