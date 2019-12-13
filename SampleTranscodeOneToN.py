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

gpuID = 0

encFile = "big_buck_bunny_1080p_h264.mov"
xcodeFile1 = open("big_buck_bunny_1080p.h264", "wb")
xcodeFile2 = open("big_buck_bunny_720p.hevc",  "wb")

nvDec = nvc.PyNvDecoder(encFile, gpuID)
nvEnc1 = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': '1920x1080'}, gpuID)
nvEnc2 = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'hevc', 's': '1280x720'},  gpuID)

while True:
    rawSurface = nvDec.DecodeSingleSurface()
    if (rawSurface.Empty()):
        # Empty surface means we have reached EOF
        break
    
    encFrame = nvEnc1.EncodeSingleSurface(rawSurface)
    if(encFrame.size):
        encByteArray = bytearray(encFrame)
        xcodeFile1.write(encByteArray)

    encFrame = nvEnc2.EncodeSingleSurface(rawSurface)
    if(encFrame.size):
        encByteArray = bytearray(encFrame)
        xcodeFile2.write(encByteArray)

#Encoders are asynchronous, so we need to flush them
encFrames = nvEnc1.Flush()
for encFrame in encFrames:
    if(encFrame.size):
        encByteArray = bytearray(encFrame)
        xcodeFile1.write(encByteArray)

encFrames = nvEnc2.Flush()
for encFrame in encFrames:
    if(encFrame.size):
        encByteArray = bytearray(encFrame)
        xcodeFile2.write(encByteArray)