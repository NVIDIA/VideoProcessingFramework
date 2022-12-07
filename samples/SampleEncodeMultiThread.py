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

import pycuda.driver as cuda
import PyNvCodec as nvc
import numpy as np

from threading import Thread


class Worker(Thread):
    def __init__(self, gpuID: int, width: int, height: int, rawFilePath: str):
        Thread.__init__(self)

        res = str(width) + "x" + str(height)

        # Retain primary CUDA device context and create separate stream per thread.
        self.ctx = cuda.Device(gpuID).retain_primary_context()
        self.ctx.push()
        self.str = cuda.Stream()
        self.ctx.pop()

        # Initialize color conversion context.
        # Accurate color rendition doesn't matter in this sample so just use
        # most common bt601 and mpeg.
        self.cc_ctx = nvc.ColorspaceConversionContext(
            color_space=nvc.ColorSpace.BT_601, color_range=nvc.ColorRange.MPEG
        )

        self.nvUpl = nvc.PyFrameUploader(
            width, height, nvc.PixelFormat.YUV420, self.ctx.handle, self.str.handle
        )

        self.nvCvt = nvc.PySurfaceConverter(
            width,
            height,
            nvc.PixelFormat.YUV420,
            nvc.PixelFormat.NV12,
            self.ctx.handle,
            self.str.handle,
        )

        self.nvEnc = nvc.PyNvEncoder(
            {"preset": "P1", "codec": "h264", "s": res},
            self.ctx.handle,
            self.str.handle,
        )

        self.rawFile = open(rawFilePath, "rb")

        self.encFrame = np.ndarray(shape=(0), dtype=np.uint8)

    def run(self):
        try:
            while True:
                frameSize = self.nvEnc.Width() * self.nvEnc.Height() * 3 / 2
                rawFrame = np.fromfile(self.rawFile, np.uint8, count=int(frameSize))
                if not (rawFrame.size):
                    print("No more video frames.")
                    break

                rawSurface = self.nvUpl.UploadSingleFrame(rawFrame)
                if rawSurface.Empty():
                    print("Failed to upload video frame to GPU.")
                    break

                cvtSurface = self.nvCvt.Execute(rawSurface, self.cc_ctx)
                if cvtSurface.Empty():
                    print("Failed to do color conversion.")
                    break

                self.nvEnc.EncodeSingleSurface(cvtSurface, self.encFrame)

            # Encoder is asynchronous, so we need to flush it
            success = self.nvEnc.Flush(self.encFrame)

        except Exception as e:
            print(getattr(e, "message", str(e)))


def create_threads(gpu_id: int, width: int, height: int, input: str, num_threads: int):
    cuda.init()

    thread_pool = []
    for i in range(0, num_threads):
        thread = Worker(gpu_id, width, height, input)
        thread.start()
        thread_pool.append(thread)

    for thread in thread_pool:
        thread.join()


if __name__ == "__main__":
    print("This sample encodes multiple videos simultaneously from same YUV file.")
    print("Usage: SampleDecode.py $gpu_id $width $height $input_file $num_threads")

    if len(sys.argv) < 6:
        print("Provide input CLI arguments as shown above")
        exit(1)

    gpu_id = int(sys.argv[1])
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    input = sys.argv[4]
    num_threads = int(sys.argv[5])

    create_threads(gpu_id, width, height, input, num_threads)
