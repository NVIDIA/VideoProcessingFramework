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

# Decoding to YUV420 output file
encFile = "big_buck_bunny_1080p_h264.mov"
decFile = open("big_buck_bunny_1920_1080.yuv420", "wb")

nvDec = nvc.PyNvDecoder(encFile, gpuID)
nvCvt = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, gpuID)
nvDwl = nvc.PySurfaceDownloader(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.YUV420, gpuID)

while True:
    rawSurfaceNV12 = nvDec.DecodeSingleSurface()
    if (rawSurfaceNV12.Empty()):
        break
    
    rawSurfaceYUV420 = nvCvt.Execute(rawSurfaceNV12)
    if (rawSurfaceYUV420.Empty()):
        break

    rawFrameYUV420 = nvDwl.DownloadSingleSurface(rawSurfaceYUV420)
    if not (rawFrameYUV420.size):
        break

    frameByteArray = bytearray(rawFrameYUV420)
    decFile.write(frameByteArray)

decFile.close()

# Encoding from UYV420 input file
decFile = open("big_buck_bunny_1920_1080.yuv420", "rb")
encFile = open("big_buck_bunny_1080p_h264.h264",  "wb")

nvEnc = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': '1920x1080'}, gpuID)
nvUpl = nvc.PyFrameUploader(nvEnc.Width(), nvEnc.Height(), nvc.PixelFormat.YUV420, gpuID)
nvCvt = nvc.PySurfaceConverter(nvEnc.Width(), nvEnc.Height(), nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, gpuID)

#Size of raw Full HD YUV420 frame is 1920 * 1080 + 960 * 540 + 960 * 540
yuv420FrameSize = 1920 * (1080 + 540)

while True:
    rawFrameYUV420 = np.fromfile(decFile, dtype = np.uint8, count = yuv420FrameSize)
    if not (rawFrameYUV420.size):
        break

    rawSurfaceYUV420 = nvUpl.UploadSingleFrame(rawFrameYUV420)
    if (rawSurfaceYUV420.Empty()):
        break
    
    rawSurfaceNV12 = nvCvt.Execute(rawSurfaceYUV420)
    if (rawSurfaceNV12.Empty()):
        break;

    encFrame = nvEnc.EncodeSingleSurface(rawSurfaceNV12)
    if(encFrame.size):
        encByteArray = bytearray(encFrame)
        encFile.write(encByteArray)

#Encoder is asyncronous, so we need to flush it
encFrames = nvEnc.Flush()
for encFrame in encFrames:
    if(encFrame.size):
        encByteArray = bytearray(encFrame)
        encFile.write(encByteArray)