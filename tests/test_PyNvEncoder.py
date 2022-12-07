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

# Ground truth information about input video
gt_file = join(dirname(__file__), "test.mp4")
gt_file_res_change = join(dirname(__file__), "test_res_change.h264")
gt_width = 848
gt_height = 464
gt_res_change = 47
gt_is_vfr = False
gt_pix_fmt = nvc.PixelFormat.NV12
gt_framerate = 30
gt_num_frames = 96
gt_timebase = 8.1380e-5
gt_color_space = nvc.ColorSpace.BT_709
gt_color_range = nvc.ColorRange.MPEG


class TestEncoderBasic(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_encode_all_surfaces(self):
        gpu_id = 0
        res = str(gt_width) + "x" + str(gt_height)
        encFrame = np.ndarray(shape=(0), dtype=np.uint8)

        nvDec = nvc.PyNvDecoder(gt_file, gpu_id)
        nvEnc = nvc.PyNvEncoder(
            {
                "preset": "P4",
                "tuning_info": "high_quality",
                "codec": "h264",
                "profile": "high",
                "s": res,
                "bitrate": "1M",
            },
            gpu_id,
        )

        frames_sent = 0
        frames_recv = 0

        while True:
            dec_surf = nvDec.DecodeSingleSurface()
            if not dec_surf or dec_surf.Empty():
                break
            frames_sent += 1

            nvEnc.EncodeSingleSurface(dec_surf, encFrame)
            if encFrame.size:
                frames_recv += 1

        while True:
            success = nvEnc.FlushSinglePacket(encFrame)
            if success and encFrame.size:
                frames_recv += 1
            else:
                break

        self.assertEqual(frames_sent, frames_recv)

    def test_reconfigure(self):
        gpu_id = 0
        res = str(gt_width) + "x" + str(gt_height)
        encFrame = np.ndarray(shape=(0), dtype=np.uint8)

        nvDec = nvc.PyNvDecoder(gt_file_res_change, gpu_id)
        nvRcn = nvc.PyNvDecoder(
            gt_width, gt_height, nvc.PixelFormat.NV12, nvc.CudaVideoCodec.H264, gpu_id
        )
        nvEnc = nvc.PyNvEncoder(
            {
                "preset": "P4",
                "tuning_info": "high_quality",
                "codec": "h264",
                "profile": "high",
                "s": res,
                "bitrate": "1M",
            },
            gpu_id,
        )

        frames_recn = 0
        while True:
            dec_surf = nvDec.DecodeSingleSurface()
            if not dec_surf or dec_surf.Empty():
                break

            sw = dec_surf.Width()
            sh = dec_surf.Height()
            if sw != gt_width or sh != gt_height:
                # Flush encoder before reconfigure.
                # Some encoded frames will be lost but that doesn't matter.
                # Decoder will be reconfigured upon resolution change anyway.
                while nvEnc.FlushSinglePacket(encFrame):
                    frames_recn += 1

                # Now reconfigure.
                res = str(sw) + "x" + str(sh)
                self.assertTrue(
                    nvEnc.Reconfigure({"s": res}, force_idr=True, reset_encoder=True)
                )
                self.assertEqual(nvEnc.Width(), sw)
                self.assertEqual(nvEnc.Height(), sh)

            nvEnc.EncodeSingleSurface(dec_surf, encFrame)

            if encFrame.size:
                dec_surf = nvRcn.DecodeSurfaceFromPacket(encFrame)
                if dec_surf and not dec_surf.Empty():
                    frames_recn += 1
                    if frames_recn < gt_res_change:
                        self.assertEqual(dec_surf.Width(), gt_width)
                        self.assertEqual(dec_surf.Height(), gt_height)
                    else:
                        self.assertEqual(dec_surf.Width(), sw)
                        self.assertEqual(dec_surf.Height(), sh)


if __name__ == "__main__":
    unittest.main()
