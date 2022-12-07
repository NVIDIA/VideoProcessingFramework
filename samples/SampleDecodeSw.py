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
import argparse
from pathlib import Path


def decode(encFilePath, decFilePath):
    decFile = open(decFilePath, "wb")

    nvDec = nvc.PyFfmpegDecoder(encFilePath, {})
    rawFrameYUV = np.ndarray(shape=(0), dtype=np.uint8)

    while True:
        success = nvDec.DecodeSingleFrame(rawFrameYUV)
        if success:
            bits = bytearray(rawFrameYUV)
            decFile.write(bits)
        else:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "This sample decodes input video to raw YUV file using libavcodec SW decoder."
    )
    parser.add_argument(
        "-e",
        "--encoded-file-path",
        type=Path,
        required=True,
        help="Encoded video file (read from)",
    )
    parser.add_argument(
        "-r",
        "--raw-file-path",
        type=Path,
        required=True,
        help="Raw YUV video file (write to)",
    )
    parser.add_argument(
        "-v", "--verbose", default=False, action="store_true", help="Verbose"
    )

    args = parser.parse_args()

    decode(args.encoded_file_path.as_posix(), args.raw_file_path.as_posix())
