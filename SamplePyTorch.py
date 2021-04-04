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
import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import numpy as np
import sys


def main(gpuID, method, encFilePath, dstFilePath):
    dstFile = open(dstFilePath, "wb")
    nvDec = nvc.PyNvDecoder(encFilePath, gpuID)

    w = nvDec.Width()
    h = nvDec.Height()
    res = str(w) + 'x' + str(h)
    nvEnc = nvc.PyNvEncoder(
        {'preset': 'hq', 'codec': 'h264', 's': res, 'bitrate' : '10M'}, gpuID)

    # define surface converters
    to_rgb = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
    to_planar = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)
    to_yuv = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420, gpuID)
    to_nv12 = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, gpuID)

    # There are 2 ways to convert the surface into a PyTorch tensor. Using the specific VPF pytorch extension
    # or taking advantage of the tensor pointer. In the second case we can avoid converting to RGB_planar.
    methods = ['PYTORCHNVCODEC', 'TOPTR_3HW', 'TOPTR_HW3']

    encFrame = np.ndarray(shape=(0), dtype=np.uint8)
    while True:
        rawSurface = nvDec.DecodeSingleSurface()
        if rawSurface.Empty():
            break

        # Convert to RGB interleaved
        rgb_byte = to_rgb.Execute(rawSurface)

        # Convert the rgb surface to a PyTorch tensor
        surface_tensor: torch.Tensor
        if method == methods[0]:
            # Using the PytorchNvCodec module
            # -------------------------------
            rgb_planar = to_planar.Execute(rgb_byte)
            surfPlane = rgb_planar.PlanePtr()
            surface_tensor = pnvc.makefromDevicePtrUint8(
                surfPlane.GpuMem(), surfPlane.Width(), surfPlane.Height(), surfPlane.Pitch(), surfPlane.ElemSize())
            surface_tensor.resize_(surfPlane.Height()//h, h, w) # to 3xHxW
        elif method == methods[1]:
            # Direct memory mapping to tensor with shape 3HW
            # ----------------------------------------------
            surface_tensor = torch.zeros((3, h, w), dtype=torch.uint8,
                                         device=torch.device(f'cuda:{gpuID}'))
            rgb_planar = to_planar.Execute(rgb_byte)
            rgb_planar.PlanePtr().Export(surface_tensor.data_ptr(), w, gpuID)
        elif method == methods[2]:
            # Direct memory mapping to tensor with shape HW3
            # ----------------------------------------------
            surface_tensor = torch.zeros((h, w, 3), dtype=torch.uint8,
                                         device=torch.device(f'cuda:{gpuID}'))
            rgb_byte.PlanePtr().Export(surface_tensor.data_ptr(), w, gpuID)
            surface_tensor = surface_tensor.permute(2, 0, 1)  # to 3xHxW
        else:
            raise RuntimeError('invalid method')

        # PROCESS YOUR TENSOR HERE

        # Create surface from a PyTorch tensor
        rawFrame = surface_tensor.permute(1, 2, 0).contiguous()  # to HxWx3
        new_surf = nvc.Surface.Make(nvc.PixelFormat.RGB, w, h, gpuID)
        new_surf.PlanePtr().Import(rawFrame.data_ptr(), int(w*3), gpuID)
        new_surf = to_yuv.Execute(new_surf)
        new_surf = to_nv12.Execute(new_surf)
        success = nvEnc.EncodeSingleSurface(new_surf, encFrame)
        if(success):
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
    print("Usage: SamplePyTorch.py $gpu_id $to_tensor_method $input_file $output_file.")
    print("valid to_tensor_methods:")
    print("   PYTORCHNVCODEC: Standard approach using the VPF PyTorch extension")
    print("   TOPTR_3HW: Direct memory mapping to PyTorch tensor with shape 3HW")
    print("   TOPTR_HW3: Direct memory mapping to PyTorch tensor with shape HW3")

    if(len(sys.argv) < 5):
        print("Provide gpu ID, to_tensor_method, path to input and output files")
        exit(1)

    gpuID = int(sys.argv[1])
    method = sys.argv[2]
    encFilePath = sys.argv[3]
    decFilePath = sys.argv[4]
    main(gpuID, method, encFilePath, decFilePath)
