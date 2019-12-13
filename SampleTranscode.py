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
xcodeFile = open("big_buck_bunny_1080p.h264", "wb")

nvDec = nvc.PyNvDecoder(encFile, gpuID)
nvEnc = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': '1920x1080'}, gpuID)

while True:
    rawSurface = nvDec.DecodeSingleSurface()
    if (rawSurface.Empty()):
        # Empty surface means we have reached EOF
        break
    
    encFrame = nvEnc.EncodeSingleSurface(rawSurface)
    if(encFrame.size):
        frameByteArray = bytearray(encFrame)
        xcodeFile.write(frameByteArray)

#Encoder is asynchronous, so we need to flush it
encFrames = nvEnc.Flush()
for encFrame in encFrames:
    if(encFrame.size):
        encByteArray = bytearray(encFrame)
        xcodeFile.write(encByteArray)