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
import random

# Ground truth information about input video
gt_file = 'test.mp4'
gt_width = 848
gt_height = 464
gt_is_vfr = False
gt_pix_fmt = nvc.PixelFormat.NV12
gt_framerate = 30
gt_num_frames = 96
gt_color_space = nvc.ColorSpace.BT_709
gt_color_range = nvc.ColorRange.MPEG


class TestDecoderBasic(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        gpu_id = 0
        enc_file = gt_file
        self.nvDec = nvc.PyNvDecoder(enc_file, gpu_id)

    def test_width(self):
        self.assertEqual(gt_width, self.nvDec.Width())

    def test_height(self):
        self.assertEqual(gt_height, self.nvDec.Height())

    def test_color_space(self):
        self.assertEqual(gt_color_space, self.nvDec.ColorSpace())

    def test_color_range(self):
        self.assertEqual(gt_color_range, self.nvDec.ColorRange())

    def test_format(self):
        self.assertEqual(gt_pix_fmt, self.nvDec.Format())

    def test_framerate(self):
        self.assertEqual(gt_framerate, self.nvDec.Framerate())

    def test_avgframerate(self):
        self.assertEqual(gt_framerate, self.nvDec.AvgFramerate())

    def test_isvfr(self):
        self.assertEqual(gt_is_vfr, self.nvDec.IsVFR())

    def test_framesize(self):
        frame_size = int(self.nvDec.Width() * self.nvDec.Height() * 3 / 2)
        self.assertEqual(frame_size, self.nvDec.Framesize())

    def test_timebase(self):
        epsilon = 1e-4
        gt_timebase = 8.1380e-5
        self.assertLessEqual(
            np.abs(gt_timebase - self.nvDec.Timebase()), epsilon)

    def test_lastpacketdata(self):
        try:
            pdata = nvc.PacketData()
            self.nvDec.LastPacketData(pdata)
        except:
            self.fail("Test case raised exception unexpectedly!")


class TestDecoderStandalone(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        gpu_id = 0
        enc_file = gt_file
        self.nvDmx = nvc.PyFFmpegDemuxer(enc_file, {})
        self.nvDec = nvc.PyNvDecoder(
            self.nvDmx.Width(), self.nvDmx.Height(), self.nvDmx.Format(),
            self.nvDmx.Codec(), gpu_id)

    def test_decodesurfacefrompacket(self):
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        while self.nvDmx.DemuxSinglePacket(packet):
            surf = self.nvDec.DecodeSurfaceFromPacket(packet)
            self.assertIsNotNone(surf)
            if not surf.Empty():
                self.assertNotEqual(0, surf.PlanePtr().GpuMem())
                self.assertEqual(self.nvDmx.Width(), surf.Width())
                self.assertEqual(self.nvDmx.Height(), surf.Height())
                self.assertEqual(self.nvDmx.Format(), surf.Format())
                return

    def test_decodesurfacefrompacket_outpktdata(self):
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        in_pdata = nvc.PacketData()
        last_pts = nvc.NO_PTS
        dec_frame = 0
        while self.nvDmx.DemuxSinglePacket(packet):
            self.nvDmx.LastPacketData(in_pdata)
            out_pdata = nvc.PacketData()
            surf = self.nvDec.DecodeSurfaceFromPacket(
                in_pdata, packet, out_pdata)
            self.assertIsNotNone(surf)
            if not surf.Empty():
                dec_frame += 1
            else:
                break
            if 0 != dec_frame:
                self.assertGreaterEqual(out_pdata.pts, last_pts)
                last_pts = out_pdata.pts

    def test_decode_all_surfaces(self):
        dec_frames = 0
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        while self.nvDmx.DemuxSinglePacket(packet):
            surf = self.nvDec.DecodeSurfaceFromPacket(packet)
            self.assertIsNotNone(surf)
            if not surf.Empty():
                dec_frames += 1
        while True:
            surf = self.nvDec.FlushSingleSurface()
            self.assertIsNotNone(surf)
            if not surf.Empty():
                dec_frames += 1
            else:
                break
        self.assertEqual(gt_num_frames, dec_frames)


class TestDecoderBuiltin(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        gpu_id = 0
        enc_file = gt_file
        self.nvDec = nvc.PyNvDecoder(enc_file, gpu_id)

    def test_decodesinglesurface(self):
        try:
            surf = self.nvDec.DecodeSingleSurface()
            self.assertIsNotNone(surf)
            self.assertFalse(surf.Empty())
        except:
            self.fail("Test case raised exception unexpectedly!")

    def test_decodesinglesurface_outpktdata(self):
        dec_frame = 0
        last_pts = nvc.NO_PTS
        while True:
            pdata = nvc.PacketData()
            surf = self.nvDec.DecodeSingleSurface(pdata)
            if surf.Empty():
                break
            self.assertNotEqual(pdata.pts, nvc.NO_PTS)
            if 0 != dec_frame:
                self.assertGreaterEqual(pdata.pts, last_pts)
            dec_frame += 1
            last_pts = pdata.pts

    def test_decodesinglesurface_sei(self):
        total_sei_size = 0
        while True:
            sei = np.ndarray(shape=(0), dtype=np.uint8)
            surf = self.nvDec.DecodeSingleSurface(sei)
            if surf.Empty():
                break
            total_sei_size += sei.size
        self.assertNotEqual(0, total_sei_size)

    def test_decodesinglesurface_seek(self):
        start_frame = random.randint(0, gt_num_frames-1)
        dec_frames = 1
        seek_ctx = nvc.SeekContext(
            seek_frame=start_frame, seek_criteria=nvc.SeekCriteria.BY_NUMBER)
        surf = self.nvDec.DecodeSingleSurface(seek_ctx)
        self.assertNotEqual(True, surf.Empty())
        while True:
            surf = self.nvDec.DecodeSingleSurface()
            if surf.Empty():
                break
            dec_frames += 1
        self.assertEqual(gt_num_frames-start_frame, dec_frames)

    def test_decode_all_surfaces(self):
        dec_frames = 0
        while True:
            surf = self.nvDec.DecodeSingleSurface()
            if not surf or surf.Empty():
                break
            dec_frames += 1
        self.assertEqual(gt_num_frames, dec_frames)


if __name__ == '__main__':
    unittest.main()
