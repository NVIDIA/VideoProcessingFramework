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

gpuID = 0
encFile = "big_buck_bunny_1080p_h264.mov"
decFile = open("big_buck_bunny_1080p_h264.nv12", "wb")

nvDec = nvc.PyNvDecoder(encFile, gpuID)

while True:
    rawFrame = nvDec.DecodeSingleFrame()
    # Decoder will return zero-size frame if input file is over;
    if not (rawFrame.size):
        break
    
    frameByteArray = bytearray(rawFrame)
    decFile.write(frameByteArray)