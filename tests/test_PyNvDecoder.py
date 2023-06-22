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
import random

# Ground truth information about input video
gt_file = join(dirname(__file__), "test.mp4")
gt_file_res_change = join(dirname(__file__), "test_res_change.h264")
gt_width = 848
gt_height = 464
gt_res_change = 47
gt_res_change_factor = 0.5
gt_is_vfr = False
gt_pix_fmt = nvc.PixelFormat.NV12
gt_framerate = 30
gt_num_frames = 96
gt_timebase = 8.1380e-5
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
        self.assertLessEqual(np.abs(gt_timebase - self.nvDec.Timebase()), epsilon)

    def test_lastpacketdata(self):
        try:
            pdata = nvc.PacketData()
            self.nvDec.LastPacketData(pdata)
        except:
            self.fail("Test case raised exception unexpectedly!")


class TestDecoderStandalone(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_decodesurfacefrompacket(self):
        nvDmx = nvc.PyFFmpegDemuxer(gt_file, {})
        nvDec = nvc.PyNvDecoder(
            nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), 0
        )

        packet = np.ndarray(shape=(0), dtype=np.uint8)
        while nvDmx.DemuxSinglePacket(packet):
            surf = nvDec.DecodeSurfaceFromPacket(packet)
            self.assertIsNotNone(surf)
            if not surf.Empty():
                self.assertNotEqual(0, surf.PlanePtr().GpuMem())
                self.assertEqual(nvDmx.Width(), surf.Width())
                self.assertEqual(nvDmx.Height(), surf.Height())
                self.assertEqual(nvDmx.Format(), surf.Format())
                return

    def test_decodesurfacefrompacket_outpktdata(self):
        nvDmx = nvc.PyFFmpegDemuxer(gt_file, {})
        nvDec = nvc.PyNvDecoder(
            nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), 0
        )

        dec_frames = 0
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        out_bst_size = 0
        while nvDmx.DemuxSinglePacket(packet):
            in_pdata = nvc.PacketData()
            nvDmx.LastPacketData(in_pdata)
            out_pdata = nvc.PacketData()
            surf = nvDec.DecodeSurfaceFromPacket(in_pdata, packet, out_pdata)
            self.assertIsNotNone(surf)
            if not surf.Empty():
                dec_frames += 1
                out_bst_size += out_pdata.bsl

        while True:
            out_pdata = nvc.PacketData()
            surf = nvDec.FlushSingleSurface(out_pdata)
            if not surf.Empty():
                out_bst_size += out_pdata.bsl
            else:
                break

        self.assertNotEqual(0, out_bst_size)

    def test_decode_all_surfaces(self):
        nvDmx = nvc.PyFFmpegDemuxer(gt_file, {})
        nvDec = nvc.PyNvDecoder(
            nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), 0
        )

        dec_frames = 0
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        while nvDmx.DemuxSinglePacket(packet):
            surf = nvDec.DecodeSurfaceFromPacket(packet)
            self.assertIsNotNone(surf)
            if not surf.Empty():
                dec_frames += 1
        while True:
            surf = nvDec.FlushSingleSurface()
            self.assertIsNotNone(surf)
            if not surf.Empty():
                dec_frames += 1
            else:
                break
        self.assertEqual(gt_num_frames, dec_frames)


class TestDecoderBuiltin(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_decodesinglesurface(self):
        gpu_id = 0
        enc_file = gt_file
        nvDec = nvc.PyNvDecoder(enc_file, gpu_id)
        try:
            surf = nvDec.DecodeSingleSurface()
            self.assertIsNotNone(surf)
            self.assertFalse(surf.Empty())
        except:
            self.fail("Test case raised exception unexpectedly!")

    def test_decodesinglesurface_outpktdata(self):
        gpu_id = 0
        enc_file = gt_file
        nvDec = nvc.PyNvDecoder(enc_file, gpu_id)

        dec_frame = 0
        last_pts = nvc.NO_PTS
        while True:
            pdata = nvc.PacketData()
            surf = nvDec.DecodeSingleSurface(pdata)
            if surf.Empty():
                break
            self.assertNotEqual(pdata.pts, nvc.NO_PTS)
            if 0 != dec_frame:
                self.assertGreaterEqual(pdata.pts, last_pts)
            dec_frame += 1
            last_pts = pdata.pts

    def test_decodesinglesurface_sei(self):
        gpu_id = 0
        enc_file = gt_file
        nvDec = nvc.PyNvDecoder(enc_file, gpu_id)

        total_sei_size = 0
        while True:
            sei = np.ndarray(shape=(0), dtype=np.uint8)
            surf = nvDec.DecodeSingleSurface(sei)
            if surf.Empty():
                break
            total_sei_size += sei.size
        self.assertNotEqual(0, total_sei_size)

    def test_decodesinglesurface_seek(self):
        gpu_id = 0
        enc_file = gt_file
        nvDec = nvc.PyNvDecoder(enc_file, gpu_id)

        start_frame = random.randint(0, gt_num_frames - 1)
        dec_frames = 1
        seek_ctx = nvc.SeekContext(seek_frame=start_frame)
        surf = nvDec.DecodeSingleSurface(seek_ctx)
        self.assertNotEqual(True, surf.Empty())
        while True:
            surf = nvDec.DecodeSingleSurface()
            if surf.Empty():
                break
            dec_frames += 1
        self.assertEqual(gt_num_frames - start_frame, dec_frames)

    def test_decodesinglesurface_cmp_vs_continuous(self):
        gpu_id = 0
        enc_file = gt_file
        nvDec = nvc.PyNvDecoder(enc_file, gpu_id)

        # First get reconstructed frame with seek
        for idx in range(0, gt_num_frames):
            seek_ctx = nvc.SeekContext(seek_frame=idx)
            frame_seek = np.ndarray(shape=(0), dtype=np.uint8)
            pdata_seek = nvc.PacketData()
            self.assertTrue(nvDec.DecodeSingleFrame(frame_seek, seek_ctx, pdata_seek))

            # Then get it with continuous decoding
            nvDec = nvc.PyNvDecoder(gt_file, 0)
            frame_cont = np.ndarray(shape=(0), dtype=np.uint8)
            pdata_cont = nvc.PacketData()
            for i in range(0, idx + 1):
                self.assertTrue(nvDec.DecodeSingleFrame(frame_cont, pdata_cont))

            # Compare frames
            if not np.array_equal(frame_seek, frame_cont):
                fail_msg = ""
                fail_msg += "Seek frame number: " + str(idx) + ".\n"
                fail_msg += "Seek frame pts:    " + str(pdata_seek.pts) + ".\n"
                fail_msg += "Cont frame pts:    " + str(pdata_cont.pts) + ".\n"
                fail_msg += "Video frames are not same\n"
                self.fail(fail_msg)

    def test_decode_all_surfaces(self):
        nvDec = nvc.PyNvDecoder(gt_file, 0)

        dec_frames = 0
        while True:
            surf = nvDec.DecodeSingleSurface()
            if not surf or surf.Empty():
                break
            dec_frames += 1
        self.assertEqual(gt_num_frames, dec_frames)

    def test_decode_resolution_change(self):
        nvDec = nvc.PyNvDecoder(gt_file_res_change, 0)
        rw = int(gt_width * gt_res_change_factor)
        rh = int(gt_height * gt_res_change_factor)

        dec_frames = 0
        while True:
            surf = nvDec.DecodeSingleSurface()
            if not surf or surf.Empty():
                break
            else:
                dec_frames += 1

            if dec_frames < gt_res_change:
                self.assertEqual(surf.Width(), gt_width)
                self.assertEqual(surf.Height(), gt_height)
            else:
                self.assertEqual(surf.Width(), rw)
                self.assertEqual(surf.Height(), rh)


if __name__ == "__main__":
    unittest.main()
