#
# Copyright 2020 NVIDIA Corporation
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

if os.name == "nt":
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
        paths = sys_path.split(";")
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

import PyNvCodec as nvc
import numpy as np
import cv2


def load_remap(remap_file):
    remap_x, remap_y = np.load(remap_file, allow_pickle=True).values()
    if remap_x.shape != remap_y.shape:
        raise ValueError(
            "remap_x.shape != remap_y.shape: ", remap_x.shape, " != ", remap_y.shape
        )

    if not remap_x.flags["C_CONTIGUOUS"]:
        remap_x = np.ascontiguousarray(remap_x, dtype=remap_x.dtype)
    if not remap_y.flags["C_CONTIGUOUS"]:
        remap_y = np.ascontiguousarray(remap_y, dtype=remap_y.dtype)

    print("----> load remap_x: ", remap_x.shape, remap_x.dtype, remap_x.strides)
    print("----> load remap_y: ", remap_y.shape, remap_y.dtype, remap_y.strides)
    return remap_x, remap_y


total_num_frames = 4


def decode(gpuID, encFilePath, remapFilePath):

    nvDec = nvc.PyNvDecoder(encFilePath, gpuID)
    w = nvDec.Width()
    h = nvDec.Height()

    to_rgb = nvc.PySurfaceConverter(
        w, h, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID
    )
    cc1 = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_709, nvc.ColorRange.JPEG)

    # init remaper
    remap_x, remap_y = load_remap(remapFilePath)
    remap_h, remap_w = remap_x.shape
    nv_remap = nvc.PySurfaceRemaper(remap_x, remap_y, nvc.PixelFormat.RGB, gpuID)

    nv_dwn = nvc.PySurfaceDownloader(remap_w, remap_h, nvc.PixelFormat.RGB, gpuID)

    dec_frame = 0
    while dec_frame < total_num_frames:
        rawSurface = nvDec.DecodeSingleSurface()
        if rawSurface.Empty():
            print("DecodeSingleSurface Failed.")
            break
        rgb24_origin = to_rgb.Execute(rawSurface, cc1)
        if rgb24_origin.Empty():
            print("Convert to rgb Failed.")
            break
        rgb24_remap = nv_remap.Execute(rgb24_origin)
        if rgb24_remap.Empty():
            print("Remap Failed.")
            break
        rawFrameRGB = np.ndarray(shape=(remap_h, remap_w, 3), dtype=np.uint8)
        if not nv_dwn.DownloadSingleSurface(rgb24_remap, rawFrameRGB):
            print("DownloadSingleSurface Failed.")
            break
        undistort_img = cv2.cvtColor(rawFrameRGB, cv2.COLOR_RGB2BGR)
        print("dump image shape: ", undistort_img.shape)
        cv2.imwrite("%s.jpg" % dec_frame, undistort_img)
        dec_frame += 1


if __name__ == "__main__":

    print(
        "This sample decodes first ",
        total_num_frames,
        " frames from input video and undistort them.",
    )
    print("Usage: SampleRemap.py $gpu_id $input_file $remap_npz_file")

    if len(sys.argv) < 4:
        print("Provide gpu_id, path to input, path to remap file")
        exit(1)

    gpu_id = int(sys.argv[1])
    encFilePath = sys.argv[2]
    remapFilePath = sys.argv[3]

    decode(gpu_id, encFilePath, remapFilePath)
