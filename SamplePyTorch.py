#
# Copyright 2021 Kognia Sports Intelligence
# Copyright 2021 NVIDIA Corporation
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

import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import numpy as np

def main(gpuID, encFilePath, dstFilePath):
    dstFile = open(dstFilePath, "wb")
    nvDec = nvc.PyNvDecoder(encFilePath, gpuID)

    w = nvDec.Width()
    h = nvDec.Height()
    res = str(w) + 'x' + str(h)
    nvEnc = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': res, 'bitrate' : '10M'}, gpuID)

    # Surface converters
    to_rgb = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
    to_yuv = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420, gpuID)
    to_nv12 = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, gpuID)

    # RGB Surface to import PyTorch tensor to
    surface_rgb = nvc.Surface.Make(nvc.PixelFormat.RGB, w, h, gpuID)

    # Encoded video frame
    encFrame = np.ndarray(shape=(0), dtype=np.uint8)

    # PyTorch tensor the VPF Surfaces will be exported to
    surface_tensor = torch.zeros(h, w, 3, dtype=torch.uint8,
                                 device=torch.device(f'cuda:{gpuID}'))

    while True:
        rawSurface = nvDec.DecodeSingleSurface()
        if rawSurface.Empty():
            break

        # Export VPF RGB Surface to PyTorch tensor.
        # Please note that pitch is equal to width * 3.
        # SurfacePlane is raw CUDA 2D memory allocation chunk so for
        # interleaved RGB frame it's width is 3x picture width.
        rgb24 = to_rgb.Execute(rawSurface)
        rgb24.PlanePtr().Export(surface_tensor.data_ptr(), w * 3, gpuID)

        # PROCESS YOUR TENSOR HERE.
        # THIS DUMMY PROCESSING WILL JUST MAKE VIDEO FRAMES DARKER.
        dark_frame = torch.floor_divide(surface_tensor, 2)

        # Import to VPF Surface. Same thing about pitch as before.
        surface_rgb.PlanePtr().Import(dark_frame.data_ptr(), w * 3, gpuID)
        # Convert to NV12
        surface_yuv = to_yuv.Execute(surface_rgb)
        surface_nv12 = to_nv12.Execute(surface_yuv)
        # Encode
        success = nvEnc.EncodeSingleSurface(surface_nv12, encFrame)
        if success:
            encByteArray = bytearray(encFrame)
            dstFile.write(encByteArray)

    # Encoder is asynchronous, so we need to flush it
    while True:
        success = nvEnc.FlushSinglePacket(encFrame)
        if(success):
            encByteArray = bytearray(encFrame)
            dstFile.write(encByteArray)
        else:
            break

if __name__ == "__main__":

    print("This sample transcode and process with pytorch an input video on given GPU.")
    print("Usage: SamplePyTorch.py $gpu_id $input_file $output_file.")

    if(len(sys.argv) < 4):
        print("Provide gpu ID, path to input and output files")
        exit(1)

    gpuID = int(sys.argv[1])
    encFilePath = sys.argv[2]
    decFilePath = sys.argv[3]
    main(gpuID, encFilePath, decFilePath)
