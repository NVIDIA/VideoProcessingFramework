#
# Copyright 2023 Vision Labs LLC
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
import json
from pydantic import BaseModel

class GroundTruth(BaseModel):
    uri: str
    width: int
    height: int
    is_vfr: bool
    pix_fmt: str
    framerate: float
    num_frames: int
    timebase: float
    color_space: str
    color_range: str


class TestDecoderBasic(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
                
        f = open("gt_files.json")
        data = json.load(f)["basic"]
        self.gtInfo = GroundTruth(**data)
        self.ffDec = nvc.PyFfmpegDecoder(self.gtInfo.uri, {})

    def test_width(self):
        self.assertEqual(self.gtInfo.width, self.ffDec.Width())

    def test_height(self):
        self.assertEqual(self.gtInfo.height, self.ffDec.Height())

    def test_color_space(self):
        self.assertEqual(self.gtInfo.color_space, str(self.ffDec.ColorSpace()))

    def test_color_range(self):
        self.assertEqual(self.gtInfo.color_range, str(self.ffDec.ColorRange()))

    def test_format(self):
        self.assertEqual(self.gtInfo.pix_fmt, str(self.ffDec.Format()))

    def test_framerate(self):
        self.assertEqual(self.gtInfo.framerate, self.ffDec.Framerate())

    def test_avgframerate(self):
        self.assertEqual(self.gtInfo.framerate, self.ffDec.AvgFramerate())

    def test_timebase(self):
        epsilon = 1e-4
        self.assertLessEqual(np.abs(self.gtInfo.timebase - self.ffDec.Timebase()), epsilon)

    def test_decode_all_frames(self):
        dec_frames = 0
        frame = np.ndarray(dtype=np.uint8, shape=())
        while self.ffDec.DecodeSingleFrame(frame):
            dec_frames += 1
        self.assertEqual(self.gtInfo.num_frames, dec_frames)

if __name__ == "__main__":
    unittest.main()
