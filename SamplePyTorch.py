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
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

import torch
import torchvision.transforms as T
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import numpy as np


class cconverter:
    """
    Colorspace conversion chain.
    """

    def __init__(self, width: int, height: int, gpu_id: int):
        self.gpu_id = gpu_id
        self.w = width
        self.h = height
        self.chain = []

    def add(self, src_fmt: nvc.PixelFormat, dst_fmt: nvc.PixelFormat) -> None:
        self.chain.append(nvc.PySurfaceConverter(
            self.w, self.h, src_fmt, dst_fmt, self.gpu_id))

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface
        cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601,
                                             nvc.ColorRange.MPEG)

        for cvt in self.chain:
            surf = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError('Failed to perform color conversion')

        return surf.Clone(self.gpu_id)


def surface_to_tensor(surface: nvc.Surface) -> torch.Tensor:
    """
    Converts planar rgb surface to cuda float tensor.
    """
    if surface.Format() != nvc.PixelFormat.RGB_PLANAR:
        raise RuntimeError('Surface shall be of RGB_PLANAR pixel format')

    surf_plane = surface.PlanePtr()
    img_tensor = pnvc.DptrToTensor(surf_plane.GpuMem(),
                                   surf_plane.Width(),
                                   surf_plane.Height(),
                                   surf_plane.Pitch(),
                                   surf_plane.ElemSize())
    if img_tensor is None:
        raise RuntimeError('Can not export to tensor.')

    img_tensor.resize_(3, int(surf_plane.Height()/3), surf_plane.Width())
    img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)
    img_tensor = torch.divide(img_tensor, 255.0)
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    return img_tensor


def tensor_to_surface(img_tensor: torch.tensor, gpu_id: int) -> nvc.Surface:
    """
    Converts cuda float tensor to planar rgb surface.
    """
    if len(img_tensor.shape) != 3 and img_tensor.shape[0] != 3:
        raise RuntimeError('Shape of the tensor must be (3, height, width)')

    tensor_w, tensor_h = img_tensor.shape[2], img_tensor.shape[1]
    img = torch.clamp(img_tensor, 0.0, 1.0)
    img = torch.multiply(img, 255.0)
    img = img.type(dtype=torch.cuda.ByteTensor)

    surface = nvc.Surface.Make(
        nvc.PixelFormat.RGB_PLANAR, tensor_w, tensor_h, gpu_id)
    surf_plane = surface.PlanePtr()
    pnvc.TensorToDptr(img, surf_plane.GpuMem(),
                      surf_plane.Width(),
                      surf_plane.Height(),
                      surf_plane.Pitch(),
                      surf_plane.ElemSize())

    return surface


def main(gpu_id, encFilePath, dstFilePath):
    dstFile = open(dstFilePath, "wb")
    nvDec = nvc.PyNvDecoder(encFilePath, gpu_id)

    w = nvDec.Width()
    h = nvDec.Height()
    res = str(w) + 'x' + str(h)
    nvEnc = nvc.PyNvEncoder(
        {'preset': 'P4', 'codec': 'h264', 's': res, 'bitrate': '10M'}, gpu_id)

    # Surface converters
    to_rgb = cconverter(w, h, gpu_id)
    to_rgb.add(nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420)
    to_rgb.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB)
    to_rgb.add(nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR)

    to_nv12 = cconverter(w, h, gpu_id)
    to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
    to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
    to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)

    # Encoded video frame
    encFrame = np.ndarray(shape=(0), dtype=np.uint8)

    while True:
        # Decode NV12 surface
        src_surface = nvDec.DecodeSingleSurface()
        if src_surface.Empty():
            break

        # Convert to planar RGB
        rgb_pln = to_rgb.run(src_surface)
        if rgb_pln.Empty():
            break

        # PROCESS YOUR TENSOR HERE.
        # THIS DUMMY PROCESSING JUST ADDS RANDOM ROTATION.
        src_tensor = surface_to_tensor(rgb_pln)
        dst_tensor = T.RandomRotation(degrees=(-1, 1))(src_tensor)
        surface_rgb = tensor_to_surface(dst_tensor, gpu_id)

        # Convert back to NV12
        dst_surface = to_nv12.run(surface_rgb)
        if src_surface.Empty():
            break

        # Encode
        success = nvEnc.EncodeSingleSurface(dst_surface, encFrame)
        if success:
            byteArray = bytearray(encFrame)
            dstFile.write(byteArray)

    # Encoder is asynchronous, so we need to flush it
    while True:
        success = nvEnc.FlushSinglePacket(encFrame)
        if(success):
            byteArray = bytearray(encFrame)
            dstFile.write(byteArray)
        else:
            break


if __name__ == "__main__":

    print("This sample transcode and process with pytorch an input video on given GPU.")
    print("Usage: SamplePyTorch.py $gpu_id $input_file $output_file.")

    if(len(sys.argv) < 4):
        print("Provide gpu ID, path to input and output files")
        exit(1)

    gpu_id = int(sys.argv[1])
    encFilePath = sys.argv[2]
    decFilePath = sys.argv[3]
    main(gpu_id, encFilePath, decFilePath)
