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
gt_width = 848
gt_height = 464
gt_is_vfr = False
gt_pix_fmt = nvc.PixelFormat.NV12
gt_framerate = 30
gt_num_frames = 96
gt_len_seconds = 3.23
gt_color_space = nvc.ColorSpace.BT_709
gt_color_range = nvc.ColorRange.MPEG


class TestDemuxer(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        enc_file = gt_file
        self.nvDmx = nvc.PyFFmpegDemuxer(enc_file, {})

    def test_width(self):
        self.assertEqual(gt_width, self.nvDmx.Width())

    def test_height(self):
        self.assertEqual(gt_height, self.nvDmx.Height())

    def test_color_space(self):
        self.assertEqual(gt_color_space, self.nvDmx.ColorSpace())

    def test_color_range(self):
        self.assertEqual(gt_color_range, self.nvDmx.ColorRange())

    def test_format(self):
        self.assertEqual(gt_pix_fmt, self.nvDmx.Format())

    def test_framerate(self):
        self.assertEqual(gt_framerate, self.nvDmx.Framerate())

    def test_avgframerate(self):
        self.assertEqual(gt_framerate, self.nvDmx.AvgFramerate())

    def test_isvfr(self):
        self.assertEqual(gt_is_vfr, self.nvDmx.IsVFR())

    def test_timebase(self):
        epsilon = 1e-4
        gt_timebase = 8.1380e-5
        self.assertLessEqual(np.abs(gt_timebase - self.nvDmx.Timebase()), epsilon)

    def test_demux_all_packets(self):
        num_packets = 0
        last_dts = 0
        while True:
            pdata = nvc.PacketData()
            packet = np.ndarray(shape=(0), dtype=np.uint8)
            if not self.nvDmx.DemuxSinglePacket(
                packet,
            ):
                break
            self.nvDmx.LastPacketData(pdata)
            if 0 != num_packets:
                self.assertGreaterEqual(pdata.dts, last_dts)
            last_dts = pdata.dts
            num_packets += 1
        self.assertEqual(gt_num_frames, num_packets)

    def test_seek_framenum(self):
        seek_frame = random.randint(0, gt_num_frames - 1)
        if self.nvDmx.IsVFR():
            print("Seek on VFR sequence, skipping this test")
            pass
        for mode in (nvc.SeekMode.EXACT_FRAME, nvc.SeekMode.PREV_KEY_FRAME):
            packet = np.ndarray(shape=(0), dtype=np.uint8)
            sk = nvc.SeekContext(
                seek_frame=seek_frame,
                mode=mode,
            )
            self.assertTrue(self.nvDmx.Seek(sk, packet))
            pdata = nvc.PacketData()
            self.nvDmx.LastPacketData(pdata)
            if nvc.SeekMode.EXACT_FRAME == mode:
                self.assertEqual(pdata.dts, pdata.duration * seek_frame)
            elif nvc.SeekMode.PREV_KEY_FRAME == mode:
                self.assertLessEqual(pdata.dts, pdata.duration * seek_frame)

    def test_seek_timestamp(self):
        timestamp = random.random() * gt_len_seconds
        if self.nvDmx.IsVFR():
            print("Seek on VFR sequence, skipping this test")
            return

        packet = np.ndarray(shape=(0), dtype=np.uint8)
        sk = nvc.SeekContext(
            seek_ts=timestamp,
            mode=nvc.SeekMode.PREV_KEY_FRAME,
        )
        self.assertTrue(self.nvDmx.Seek(sk, packet))
        pdata = nvc.PacketData()
        self.nvDmx.LastPacketData(pdata)
        self.assertLessEqual(pdata.dts * self.nvDmx.Timebase(), timestamp)

    def test_demux_single_packet(self):
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        while self.nvDmx.DemuxSinglePacket(packet):
            self.assertNotEqual(0, packet.size)

    def test_sei(self):
        total_sei_size = 0
        while True:
            packet = np.ndarray(shape=(0), dtype=np.uint8)
            sei = np.ndarray(shape=(0), dtype=np.uint8)
            if not self.nvDmx.DemuxSinglePacket(packet, sei):
                break
            total_sei_size += sei.size
        self.assertNotEqual(0, total_sei_size)

    def test_lastpacketdata(self):
        try:
            pdata = nvc.PacketData()
            self.nvDmx.LastPacketData(pdata)
        except:
            self.fail("Test case raised exception unexpectedly!")


if __name__ == "__main__":
    unittest.main()
