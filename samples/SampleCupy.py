#
# Copyright 2023 @royinx

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
from typing import Any
import PyNvCodec as nvc
import numpy as np
import cupy as cp

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
        self.chain.append(
            nvc.PySurfaceConverter(self.w, self.h, src_fmt, dst_fmt, self.gpu_id)
        )

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface
        cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

        for cvt in self.chain:
            surf = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError("Failed to perform color conversion")

        return surf.Clone(self.gpu_id)

class CupyNVC:
    def get_memptr(self, surface: nvc.Surface) -> int:
        return surface.PlanePtr().GpuMem()

    def SurfaceToArray(self, surface: nvc.Surface) -> cp.array:
        """
        Converts surface to cupy unit8 tensor.

        - surface: nvc.Surface
        - return: cp.array (height, width, 3)
        """
        if surface.Format() != nvc.PixelFormat.RGB:
            raise RuntimeError("Surface shall be of RGB PLANAR format , got {}".format(surface.Format()))
        plane = surface.PlanePtr()
        # cuPy array zero copy non ownned
        height, width, pitch = (plane.Height(), plane.Width(), plane.Pitch())
        cupy_mem = cp.cuda.UnownedMemory(self.get_memptr(surface), height * width * 1, surface)
        cupy_memptr = cp.cuda.MemoryPointer(cupy_mem, 0)
        cupy_frame = cp.ndarray((height, width // 3, 3), cp.uint8, cupy_memptr, strides=(pitch, 3, 1)) # RGB

        return cupy_frame

    def _memcpy(self, surface: nvc.Surface, img_array: cp.array) -> None:
        cp.cuda.runtime.memcpy2DAsync(self.get_memptr(surface),
                                        surface.Pitch(),
                                        img_array.data.ptr,
                                        surface.Width(),
                                        surface.Width(),
                                        surface.Height()*3,
                                        cp.cuda.runtime.memcpyDeviceToDevice,
                                        0) # null_stream.ptr: 0
        return

    def ArrayToSurface(self, img_array: cp.array, gpu_id: int) -> nvc.Surface:
        """
        Converts cupy ndarray to rgb surface.
        - surface: cp.array
        - return: nvc.Surface
        """
        img_array = img_array.astype(cp.uint8)
        img_array = cp.transpose(img_array, (2,0,1)) # HWC to CHW
        img_array = cp.ascontiguousarray(img_array)
        _ ,tensor_h , tensor_w= img_array.shape
        surface = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, tensor_w, tensor_h, gpu_id)
        self._memcpy(surface, img_array)
        return surface

def grayscale(img_array: cp.array) -> cp.array:
    img_array = cp.matmul(img_array, cp.array([0.299, 0.587, 0.114]).T)
    img_array = cp.expand_dims(img_array, axis=-1)
    img_array = cp.tile(img_array, (1,1,3)) # view as 3 channel image (packed RGB: HWC)
    return img_array

def contrast_boost(img_array: cp.array) -> cp.array:
    """
    histogram equalization
    """
    channel_min = cp.quantile(img_array, 0.05, axis=(0,1))
    channel_max = cp.quantile(img_array, 0.95, axis=(0,1))
    img_array = img_array.astype(cp.float32)
    for c, (cmin, cmax) in enumerate(zip(channel_min, channel_max)):
        img_array[c] = cp.clip(img_array[c], cmin, cmax)
    img_array = img_array- channel_min.reshape(1,1,-1)
    img_array /= (channel_max - channel_min).reshape(1,1,-1)
    img_array = cp.multiply(img_array, 255.0)
    return img_array

def main(gpu_id: int, encFilePath: str, dstFilePath: str):
    dstFile = open(dstFilePath, "wb")
    nvDec = nvc.PyNvDecoder(encFilePath, gpu_id)
    cpnvc = CupyNVC()

    w = nvDec.Width()
    h = nvDec.Height()
    res = str(w) + "x" + str(h)
    nvEnc = nvc.PyNvEncoder(
        {"preset": "P4", "codec": "h264", "s": res, "bitrate": "10M"}, gpu_id
    )

    # Surface converters
    to_rgb = cconverter(w, h, gpu_id)
    to_rgb.add(nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420)
    to_rgb.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB)

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

        # Convert to packed RGB: HWC , planar CHW
        rgb_sur = to_rgb.run(src_surface)
        if rgb_sur.Empty():
            break

        # PROCESS YOUR TENSOR HERE.
        # THIS DUMMY PROCESSING JUST ADDS RANDOM ROTATION.
        src_array = cpnvc.SurfaceToArray(rgb_sur)
        dst_array = contrast_boost(src_array)
        dst_array = grayscale(dst_array)
        surface_rgb = cpnvc.ArrayToSurface(dst_array, gpu_id)

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
        if success:
            byteArray = bytearray(encFrame)
            dstFile.write(byteArray)
        else:
            break


if __name__ == "__main__":


    if len(sys.argv) < 4:
        print("This sample transcode and process with pytorch an input video on given GPU.")
        print("Provide gpu ID, path to input and output files")
        print("Usage: SamplePyTorch.py $gpu_id $input_file $output_file.")
        print("Example: \npython3 samples/SampleCupy.py 0 tests/test.mp4 tests/dec_test.mp4")
        exit(1)

    gpu_id = int(sys.argv[1])
    encFilePath = sys.argv[2]
    decFilePath = sys.argv[3]
    main(gpu_id, encFilePath, decFilePath)
