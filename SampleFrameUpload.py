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
decFile = open("big_buck_bunny_1080p_h264.nv12", "rb")
encFile = open("big_buck_bunny_1080p_h264.h264", "wb")

nvEnc = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': '1920x1080'}, gpuID)
nvUpl = nvc.PyFrameUploader(nvEnc.Width(), nvEnc.Height(), nvEnc.PixelFormat(), gpuID)

#Size of raw Full HD NV12 frame is 1920 * (1080 + 540)
nv12FrameSize = 1920 * (1080 + 540)

while True:
    rawFrame = np.fromfile(decFile, dtype = np.uint8, count = nv12FrameSize)
    if not (rawFrame.size):
        break
    
    rawSurface = nvUpl.UploadSingleFrame(rawFrame)
    if (rawSurface.Empty()):
        break

    encFrame = nvEnc.EncodeSingleSurface(rawSurface)
    if(encFrame.size):
        encByteArray = bytearray(encFrame)
        encFile.write(encByteArray)

#Encoder is asyncronous, so we need to flush it
encFrames = nvEnc.Flush()
for encFrame in encFrames:
    if(encFrame.size):
        encByteArray = bytearray(encFrame)
        encFile.write(encByteArray)