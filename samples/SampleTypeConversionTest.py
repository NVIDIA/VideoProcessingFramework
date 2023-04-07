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

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
import sys
import os

if os.name == 'nt':
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file = sys.stderr)
        print("Can't set CUDA DLLs search path.", file = sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file = sys.stderr)
        exit(1)

import pycuda.driver as cuda
import PyNvCodec as nvc
import numpy as np
import torch
import nvcv

def decode(gpuID, encFilePath, decFilePath):
    cuda.init()
    cuda_ctx = cuda.Device(gpuID).retain_primary_context()
    cuda_ctx.push()
    cuda_str = cuda.Stream()
    cuda_ctx.pop()

    decFile = open(decFilePath, "wb")

    nvDmx = nvc.PyFFmpegDemuxer(encFilePath)
    nvDec = nvc.PyNvDecoder(nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), cuda_ctx.handle, cuda_str.handle)

    nvDwn = nvc.PySurfaceDownloader(nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), cuda_ctx.handle, cuda_str.handle)

    res = str(nvDmx.Width()) + 'x' + str(nvDmx.Height())

    nvEnc = nvc.PyNvEncoder({'preset': 'P5', 'tuning_info': 'high_quality', 'codec': 'h264',
                             'profile': 'high', 's': res, 'bitrate': '10M'}, cuda_ctx.handle, cuda_str.handle)

    packet = np.ndarray(shape=(0), dtype=np.uint8)
    frameSize = int(nvDmx.Width() * nvDmx.Height() * 3 / 2)
    rawFrame = np.ndarray(shape=(frameSize), dtype=np.uint8)
    pdata_in, pdata_out = nvc.PacketData(), nvc.PacketData()

    encFrame = np.ndarray(shape=(0), dtype=np.uint8)

    # Determine colorspace conversion parameters.
    # Some video streams don't specify these parameters so default values
    # are most widespread bt601 and mpeg.
    cspace, crange = nvDmx.ColorSpace(), nvDmx.ColorRange()
    if nvc.ColorSpace.UNSPEC == cspace:
        cspace = nvc.ColorSpace.BT_601
    if nvc.ColorRange.UDEF == crange:
        crange = nvc.ColorRange.MPEG
    cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)
    print('Color space: ', str(cspace))
    print('Color range: ', str(crange))

    while True:
        # Demuxer has sync design, it returns packet every time it's called.
        # If demuxer can't return packet it usually means EOF.
        torch.cuda.nvtx.range_push("DemuxSinglePacket")
        success = nvDmx.DemuxSinglePacket(packet)
        torch.cuda.nvtx.range_pop()
        if not success:
            break


        # Get last packet data to obtain frame timestamp
        torch.cuda.nvtx.range_push("LastPacketData")
        nvDmx.LastPacketData(pdata_in)
        torch.cuda.nvtx.range_pop()

        # Decoder is async by design.
        # As it consumes packets from demuxer one at a time it may not return
        # decoded surface every time the decoding function is called.
        torch.cuda.nvtx.range_push("nvDec.DecodeSurfaceFromPacket")

        surface_nv12 = nvDec.DecodeSurfaceFromPacket(pdata_in, packet, pdata_out ,True)
        torch.cuda.nvtx.range_pop()

        if surface_nv12.width == 0 and surface_nv12.height == 0:
            continue

        torch.cuda.nvtx.range_push("nvEnc.EncodeSingleSurface")
        success = nvEnc.EncodeFromNVCVImage(surface_nv12, encFrame)
        torch.cuda.nvtx.range_pop()

        if (success):
            bits = bytearray(encFrame)
            decFile.write(bits)
            print("Interop test successfull")
            break;



if __name__ == "__main__":

    print("This sample decodes input video to raw YUV420 file on given GPU.")
    print("Usage: SampleDecode.py $gpu_id $input_file $output_file.")

    if(len(sys.argv) < 4):
        print("Provide gpu ID, path to input and output files")
        exit(1)

    gpuID = int(sys.argv[1])
    encFilePath = sys.argv[2]
    decFilePath = sys.argv[3]

    decode(gpuID, encFilePath, decFilePath)
