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

import PyNvCodec as nvc
import numpy as np
import unittest


class TestDecoderBasic(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        gpu_id = 0
        enc_file = 'test.mkv'
        self.nvDec = nvc.PyNvDecoder(enc_file, gpu_id)

    def test_width(self):
        self.assertEqual(1280, self.nvDec.Width())

    def test_height(self):
        self.assertEqual(720, self.nvDec.Height())

    def test_color_space(self):
        self.assertEqual(nvc.ColorSpace.BT_709, self.nvDec.ColorSpace())

    def test_color_range(self):
        self.assertEqual(nvc.ColorRange.JPEG, self.nvDec.ColorRange())

    def test_format(self):
        self.assertEqual(nvc.PixelFormat.NV12, self.nvDec.Format())

    def test_framerate(self):
        self.assertEqual(30, self.nvDec.Framerate())

    def test_avgframerate(self):
        self.assertEqual(30, self.nvDec.AvgFramerate())

    def test_isvfr(self):
        self.assertEqual(False, self.nvDec.IsVFR())

    def test_framesize(self):
        frame_size = int(self.nvDec.Width() * self.nvDec.Height() * 3 / 2)
        self.assertEqual(frame_size, self.nvDec.Framesize())

    def test_timebase(self):
        epsilon = 1e-4
        gt_timebase = 1e-3
        self.assertLessEqual(
            np.abs(gt_timebase - self.nvDec.Timebase()), epsilon)

    def test_lastpacketdata(self):
        try:
            pdata = nvc.PacketData()
            self.nvDec.LastPacketData(pdata)
        except:
            self.fail("Test case raised exception unexpectedly!")


class TestDecoderBuiltin(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        gpu_id = 0
        enc_file = 'test.avi'
        self.nvDec = nvc.PyNvDecoder(enc_file, gpu_id)

    def test_decodesinglesurface_noargs(self):
        try:
            surf = self.nvDec.DecodeSingleSurface()
            self.assertIsNotNone(surf)
            self.assertFalse(surf.Empty())
        except:
            self.fail("Test case raised exception unexpectedly!")

    def test_decodesinglesurface_outpktdata(self):
        last_pts = nvc.NO_PTS
        dec_frame = 0
        while True:
            pdata = nvc.PacketData()
            surf = self.nvDec.DecodeSingleSurface(pdata)
            if surf.Empty():
                break
            self.assertNotEqual(pdata.pts, nvc.NO_PTS)
            if(0 != dec_frame):
                self.assertGreaterEqual(pdata.pts, last_pts)
            last_pts = pdata.pts
            dec_frame += 1

    def test_decode_all_surfaces(self):
        dec_frames = 0
        while True:
            surf = self.nvDec.DecodeSingleSurface()
            if not surf or surf.Empty():
                break
            else:
                dec_frames += 1
        self.assertEqual(30, dec_frames)


if __name__ == '__main__':
    unittest.main()
