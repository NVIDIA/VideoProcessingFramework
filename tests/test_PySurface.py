#
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
from os.path import join, dirname

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
import unittest

try:
    import pycuda.driver as cuda
    import torch
except ImportError as e:
    raise unittest.SkipTest(f"Skipping because of insufficient dependencies: {e}")


# Ground truth information about input video
gt_file = join(dirname(__file__), "test.mp4")
gt_width = 848
gt_height = 464
gt_is_vfr = False
gt_pix_fmt = nvc.PixelFormat.NV12
gt_framerate = 30
gt_num_frames = 96
gt_color_space = nvc.ColorSpace.BT_709
gt_color_range = nvc.ColorRange.MPEG


class TestSurfacePycuda(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        self.gpu_id = 0
        enc_file = gt_file
        cuda.init()
        self.cuda_ctx = cuda.Device(self.gpu_id).retain_primary_context()
        self.cuda_ctx.push()
        self.cuda_str = cuda.Stream()
        self.cuda_ctx.pop()
        self.nvDec = nvc.PyNvDecoder(
            enc_file, self.cuda_ctx.handle, self.cuda_str.handle
        )
        self.nvDwn = nvc.PySurfaceDownloader(
            self.nvDec.Width(),
            self.nvDec.Height(),
            self.nvDec.Format(),
            self.cuda_ctx.handle,
            self.cuda_str.handle,
        )

    def test_pycuda_memcpy_Surface_Surface(self):

        while True:
            surf_src = self.nvDec.DecodeSingleSurface()
            if surf_src.Empty():
                break
            src_plane = surf_src.PlanePtr()

            surf_dst = nvc.Surface.Make(
                self.nvDec.Format(),
                self.nvDec.Width(),
                self.nvDec.Height(),
                self.gpu_id,
            )
            self.assertFalse(surf_dst.Empty())
            dst_plane = surf_dst.PlanePtr()

            memcpy_2d = cuda.Memcpy2D()
            memcpy_2d.width_in_bytes = src_plane.Width() * src_plane.ElemSize()
            memcpy_2d.src_pitch = src_plane.Pitch()
            memcpy_2d.dst_pitch = dst_plane.Pitch()
            memcpy_2d.width = src_plane.Width()
            memcpy_2d.height = src_plane.Height()
            memcpy_2d.set_src_device(src_plane.GpuMem())
            memcpy_2d.set_dst_device(dst_plane.GpuMem())
            memcpy_2d(self.cuda_str)

            frame_src = np.ndarray(shape=(0), dtype=np.uint8)
            if not self.nvDwn.DownloadSingleSurface(surf_src, frame_src):
                self.fail("Failed to download decoded surface")

            frame_dst = np.ndarray(shape=(0), dtype=np.uint8)
            if not self.nvDwn.DownloadSingleSurface(surf_dst, frame_dst):
                self.fail("Failed to download decoded surface")

            if not np.array_equal(frame_src, frame_dst):
                self.fail("Video frames are not equal")

    def test_pycuda_memcpy_Surface_Tensor(self):

        while True:
            surf_src = self.nvDec.DecodeSingleSurface()
            if surf_src.Empty():
                break
            src_plane = surf_src.PlanePtr()

            surface_tensor = torch.zeros(
                src_plane.Height(),
                src_plane.Width(),
                1,
                dtype=torch.uint8,
                device=torch.device(f"cuda:{self.gpu_id}"),
            )
            dst_plane = surface_tensor.data_ptr()

            memcpy_2d = cuda.Memcpy2D()
            memcpy_2d.width_in_bytes = src_plane.Width() * src_plane.ElemSize()
            memcpy_2d.src_pitch = src_plane.Pitch()
            memcpy_2d.dst_pitch = self.nvDec.Width()
            memcpy_2d.width = src_plane.Width()
            memcpy_2d.height = src_plane.Height()
            memcpy_2d.set_src_device(src_plane.GpuMem())
            memcpy_2d.set_dst_device(dst_plane)
            memcpy_2d(self.cuda_str)

            frame_src = np.ndarray(shape=(0), dtype=np.uint8)
            if not self.nvDwn.DownloadSingleSurface(surf_src, frame_src):
                self.fail("Failed to download decoded surface")

            frame_dst = surface_tensor.to("cpu").numpy()
            frame_dst = frame_dst.reshape((src_plane.Height() * src_plane.Width()))

            if not np.array_equal(frame_src, frame_dst):
                self.fail("Video frames are not equal")

    def test_list_append(self):
        dec_frames = []
        nvDec = nvc.PyNvDecoder(gt_file, 0)

        # Decode all the surfaces and store them in the list.
        while True:
            surf = nvDec.DecodeSingleSurface()
            if not surf or surf.Empty():
                break
            else:
                # Please note that we need to clone surfaces because those
                # surfaces returned by decoder belongs to it's internal
                # memory pool.
                dec_frames.append(surf.Clone(self.gpu_id))

        # Make sure all the surfaces are kept.
        self.assertEqual(len(dec_frames), gt_num_frames)

        # Now compare saved surfaces with data from decoder to make sure
        # no crruption happened.
        nvDec = nvc.PyNvDecoder(gt_file, 0)
        nvDwn = nvc.PySurfaceDownloader(
            nvDec.Width(), nvDec.Height(), nvDec.Format(), self.gpu_id
        )

        for surf in dec_frames:
            dec_frame = np.ndarray(shape=(0), dtype=np.uint8)
            svd_frame = np.ndarray(shape=(0), dtype=np.uint8)

            nvDwn.DownloadSingleSurface(surf, svd_frame)
            nvDec.DecodeSingleFrame(dec_frame)

            self.assertTrue(np.array_equal(dec_frame, svd_frame))


if __name__ == "__main__":
    unittest.main()
