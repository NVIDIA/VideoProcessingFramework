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

import PyNvCodec as nvc
import numpy as np

def decode(gpuID, encFilePath, decFilePath):
    decFile = open(decFilePath, "wb")

    nvDmx = nvc.PyFFmpegDemuxer(encFilePath)
    nvDec = nvc.PyNvDecoder(nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), gpuID)
    nvCvt = nvc.PySurfaceConverter(nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvc.PixelFormat.YUV420, gpuID)
    nvDwn = nvc.PySurfaceDownloader(nvDmx.Width(), nvDmx.Height(), nvCvt.Format(), gpuID)

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
        surface_nv12 = nvDec.DecodeSurfaceFromPacket(packet)
        if not surface_nv12.Empty():
            surface_yuv420 = nvCvt.Execute(surface_nv12)
            if surface_yuv420.Empty():
                break
            if not nvDwn.DownloadSingleSurface(surface_yuv420, rawFrame):
                break
            bits = bytearray(rawFrame)
            decFile.write(bits)

    # Now we flush decoder to emtpy decoded frames queue.
    while True:
        surface_nv12 = nvDec.FlushSingleSurface()
        if surface_nv12.Empty():
            break
        surface_yuv420 = nvCvt.Execute(surface_nv12)
        if surface_yuv420.Empty():
            break
        if not nvDwn.DownloadSingleSurface(surface_yuv420, rawFrame):
            break
        bits = bytearray(rawFrame)
        decFile.write(bits)
    
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