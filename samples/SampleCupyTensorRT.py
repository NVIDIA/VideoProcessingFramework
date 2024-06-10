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
sys.path.append(".")
import os
from typing import Any
import PyNvCodec as nvc
import tensorrt as trt
import numpy as np
import cupy as cp
from samples.SampleTensorRTResnet import resnet_categories

class TensorRT:
    def __init__(self,engine_file):
        super().__init__()
        self.TRT_LOGGER = trt.Logger()
        self.engine = self.get_engine(engine_file)
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()

    def get_engine(self, engine_file_path):
        if not os.path.exists(engine_file_path):
            raise "run ./samples/SampleTensorRTResnet.py to generate engine file"
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, \
            trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        """
        In this Application, we use cupy for in and out

        trt use gpu array to run inference.
        while bindings store the gpu array ptr , via the method :
            cupy.ndarray.data.ptr
            cupu.cuda.alloc_pinned_memory
            cupy.cuda.runtime.malloc.mem_alloc
        """
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cp.cuda.Stream(non_blocking=False)

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            device_array = cp.empty(shape, dtype)
            self.bindings.append(device_array.data.ptr) # cupy array ptr
            # Append to the appropriate list.
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append(device_array)
            elif self.engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                self.outputs.append(device_array)

    def inference(self,inputs:cp.ndarray) -> list:
        inputs = cp.ascontiguousarray(inputs)
        cp.cuda.runtime.memcpyAsync(dst = self.inputs[0].data.ptr,
                                    src = inputs.data.ptr,
                                    size= inputs.nbytes,
                                    kind = cp.cuda.runtime.memcpyDeviceToDevice,
                                    stream = self.stream.ptr)
        self.context.execute_async_v2(bindings=self.bindings,
                                    stream_handle=self.stream.ptr)
        self.stream.synchronize()
        return [out for out in self.outputs]


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
    def resize(self, width: int, height: int, src_fmt: nvc.PixelFormat) -> None:
        self.chain.append(
            nvc.PySurfaceResizer(width, height, src_fmt, self.gpu_id)
        )
        self.h = height
        self.w = width

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface
        cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

        for cvt in self.chain:
            if isinstance(cvt, nvc.PySurfaceResizer):
                surf = cvt.Execute(surf)
            else:
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

def normalize(tensor: cp.array, mean:list , std:list) -> cp.array:
    """
    normalize along the last axis
    """
    tensor -= cp.array(mean).reshape(1,1,-1)
    tensor /= cp.array(std).reshape(1,1,-1)
    return tensor

def main(gpu_id: int, encFilePath: str):
    engine = TensorRT("resnet50.trt")
    nvDec = nvc.PyNvDecoder(encFilePath, gpu_id)
    cpnvc = CupyNVC()

    w = nvDec.Width()
    h = nvDec.Height()

    # Surface converters
    to_rgb = cconverter(w, h, gpu_id)
    to_rgb.add(nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420)
    to_rgb.resize(224,224, nvc.PixelFormat.YUV420)
    to_rgb.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB)

    # Encoded video frame
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
        src_array = cpnvc.SurfaceToArray(rgb_sur)
        src_array = src_array.astype(cp.float32)

        # preprocess
        src_array /= 255.0
        src_array = normalize(src_array,
                              mean= [0.485, 0.456, 0.406],
                              std = [0.229, 0.224, 0.225])
        src_array = cp.transpose(src_array, (2,0,1))
        src_array = cp.expand_dims(src_array, axis=0) # NCHW

        pred = engine.inference(src_array)
        pred = pred[0] # extract first output layer

        idx = cp.argmax(pred)
        print("Image type: ", resnet_categories[cp.asnumpy(idx)])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("This sample decode and inference an input video with cupy on given GPU.")
        print("[Usage]: python3 samples/SampleCupyTensorRT.py <gpu_id> <video_path>")
        exit(1)

    gpu_id = int(sys.argv[1])
    encFilePath = sys.argv[2]
    main(gpu_id, encFilePath)
