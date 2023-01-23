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
    def __init__(self, gpuID, encFile):
        Thread.__init__(self)

        # Retain primary CUDA device context and create separate stream per thread.
        self.ctx = cuda.Device(gpuID).retain_primary_context()
        self.ctx.push()
        self.str = cuda.Stream()
        self.ctx.pop()

        # Create Decoder with given CUDA context & stream.
        self.nvDec = nvc.PyNvDecoder(encFile, self.ctx.handle, self.str.handle)

        width, height = self.nvDec.Width(), self.nvDec.Height()
        hwidth, hheight = int(width / 2), int(height / 2)

        # Determine colorspace conversion parameters.
        # Some video streams don't specify these parameters so default values
        # are most widespread bt601 and mpeg.
        cspace, crange = self.nvDec.ColorSpace(), self.nvDec.ColorRange()
        if nvc.ColorSpace.UNSPEC == cspace:
            cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == crange:
            crange = nvc.ColorRange.MPEG
        self.cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)
        print("Color space: ", str(cspace))
        print("Color range: ", str(crange))

        # Initialize colorspace conversion chain
        if self.nvDec.ColorSpace() != nvc.ColorSpace.BT_709:
            self.nvYuv = nvc.PySurfaceConverter(
                width,
                height,
                self.nvDec.Format(),
                nvc.PixelFormat.YUV420,
                self.ctx.handle,
                self.str.handle,
            )
        else:
            self.nvYuv = None

        if self.nvYuv:
            self.nvCvt = nvc.PySurfaceConverter(
                width,
                height,
                self.nvYuv.Format(),
                nvc.PixelFormat.RGB,
                self.ctx.handle,
                self.str.handle,
            )
        else:
            self.nvCvt = nvc.PySurfaceConverter(
                width,
                height,
                self.nvDec.Format(),
                nvc.PixelFormat.RGB,
                self.ctx.handle,
                self.str.handle,
            )

        self.nvRes = nvc.PySurfaceResizer(
            hwidth, hheight, self.nvCvt.Format(), self.ctx.handle, self.str.handle
        )
        self.nvDwn = nvc.PySurfaceDownloader(
            hwidth, hheight, self.nvRes.Format(), self.ctx.handle, self.str.handle
        )
        self.num_frame = 0

    def run(self):
        try:
            while True:
                try:
                    self.rawSurface = self.nvDec.DecodeSingleSurface()
                    if self.rawSurface.Empty():
                        print("No more video frames")
                        break
                except nvc.HwResetException:
                    print("Continue after HW decoder was reset")
                    continue

                if self.nvYuv:
                    self.yuvSurface = self.nvYuv.Execute(self.rawSurface, self.cc_ctx)
                    self.cvtSurface = self.nvCvt.Execute(self.yuvSurface, self.cc_ctx)
                else:
                    self.cvtSurface = self.nvCvt.Execute(self.rawSurface, self.cc_ctx)
                if self.cvtSurface.Empty():
                    print("Failed to do color conversion")
                    break

                self.resSurface = self.nvRes.Execute(self.cvtSurface)
                if self.resSurface.Empty():
                    print("Failed to resize surface")
                    break

                self.rawFrame = np.ndarray(
                    shape=(self.resSurface.HostSize()), dtype=np.uint8
                )
                success = self.nvDwn.DownloadSingleSurface(
                    self.resSurface, self.rawFrame
                )
                if not (success):
                    print("Failed to download surface")
                    break

                self.num_frame += 1
                if 0 == self.num_frame % self.nvDec.Framerate():
                    print(f"Thread {self.ident} at frame {self.num_frame}")

        except Exception as e:
            print(getattr(e, "message", str(e)))
            fout.close()


def create_threads(gpu_id, input_file, num_threads):
    cuda.init()

    thread_pool = []
    for i in range(0, num_threads):
        thread = Worker(gpu_id, input_file)
        thread.start()
        thread_pool.append(thread)

    for thread in thread_pool:
        thread.join()


if __name__ == "__main__":

    print(
        "This sample decodes video streams in parallel threads. It does not save output."
    )
    print("GPU-accelerated color conversion and resize are also applied.")
    print("This sample may serve as a stability test.")
    print("Usage: python SampleDecodeMultiThread.py $gpu_id $input $num_threads")

    if len(sys.argv) < 4:
        print("Provide input CLI arguments as shown above")
        exit(1)

    gpu_id = int(sys.argv[1])
    input_file = sys.argv[2]
    num_threads = int(sys.argv[3])

    create_threads(gpu_id, input_file, num_threads)
