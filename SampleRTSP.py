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
import sys
import PyNvCodec as nvc

gpuID = 0
numFrames = 512

print("This sample takes rtsp stream URL as input and transcodes it to local H.264 file")

if(len(sys.argv) < 2):
    print("Provide rtsp stream URL as CLI argument")
    exit(1)

encFile = sys.argv[1]
outFile = open("rtsp_stream.h264", "wb")

try:
    nvDec = nvc.PyNvDecoder(encFile, gpuID)
    nvEnc = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': '640x480'}, gpuID)

    # Get certain amount of frames from source
    frame_num = 0
    while frame_num < numFrames:
        rawSurface = nvDec.DecodeSingleSurface()
        # Decoder will return zero-size frame if input file is over
        if (rawSurface.Empty()):
            break

        # Encoder has some latency to it so it may return empty
        # encoded frame which is OK
        encFrame = nvEnc.EncodeSingleSurface(rawSurface)
        if(encFrame.size):
            encByteArray = bytearray(encFrame)
            outFile.write(encByteArray)

        frame_num += 1

    #Encoder is asynchronous, so we need to flush it
    encFrames = nvEnc.Flush()
    for encFrame in encFrames:
        if(encFrame.size):
            encByteArray = bytearray(encFrame)
            outFile.write(encByteArray)

except Exception as e:
    print(getattr(e, 'message', str(e)))

exit(0)