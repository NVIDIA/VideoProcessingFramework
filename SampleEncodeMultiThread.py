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

if os.name == 'nt':
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file = sys.stderr)
        print("Can't set CUDA DLLs search path.", file = sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file = sys.stderr)
        exit(1)

import PyNvCodec as nvc
import numpy as np

from threading import Thread
 
class Worker(Thread):
    def __init__(self, gpuID, width, height, rawFilePath, encFilePath):
        Thread.__init__(self)

        res = width + 'x' + height
        
        self.nvUpl = nvc.PyFrameUploader(int(width), int(height), nvc.PixelFormat.YUV420, gpuID)
        self.nvCvt = nvc.PySurfaceConverter(int(width), int(height), nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, gpuID)
        self.nvEnc = nvc.PyNvEncoder({'preset': 'hq', 'codec': 'h264', 's': res}, gpuID)

        self.encFile = open(encFilePath, "wb")
        self.rawFile = open(rawFilePath, "rb")
 
    def run(self):
        try:
            while True:
                frameSize = self.nvEnc.Width() * self.nvEnc.Height() * 3 / 2
                rawFrame = np.fromfile(self.rawFile, np.uint8, count = int(frameSize))
                if not (rawFrame.size):
                    print('No more video frames')
                    break

                rawSurface = self.nvUpl.UploadSingleFrame(rawFrame)
                if (rawSurface.Empty()):
                    print('Failed to upload video frame to GPU')
                    break
 
                cvtSurface = self.nvCvt.Execute(rawSurface)
                if (cvtSurface.Empty()):
                    print('Failed to do color conversion')
                    break

                encFrame = np.ndarray(shape=(0), dtype=np.uint8)
                success = self.nvEnc.EncodeSingleSurface(cvtSurface, encFrame)
                if(success):
                    bits = bytearray(encFrame)
                    self.encFile.write(bits)

            #Encoder is asynchronous, so we need to flush it
            encFrame = np.ndarray(shape=(0), dtype=np.uint8)
            success = self.nvEnc.Flush(encFrame)
            if(success):
                bits = bytearray(encFrame)
                self.encFile.write(bits)
 
        except Exception as e:
            print(getattr(e, 'message', str(e)))
            decFile.close()
 
def create_threads(gpu_id1, width_1, height_1, input_file1, output_file1,
                   gpu_id2, width_2, height_2, input_file2, output_file2):
 
    th1 = Worker(gpu_id1, width_1, height_1, input_file1, output_file1)
    th2 = Worker(gpu_id2, width_2, height_2, input_file2, output_file2)
 
    th1.start()
    th2.start()
 
    th1.join()
    th2.join()
 
if __name__ == "__main__":
    print("This sample encodes 2 videos simultaneously from YUV files into 1/4 of initial size.")
    print("Usage: SampleDecode.py $gpu_id1 $width_1 $height_1 $input_file1 $output_file_1 $gpu_id2 $width_2 $height_2 $input_file2 $output_file2")
 
    if(len(sys.argv) < 11):
        print("Provide input CLI arguments as shown above")
        exit(1)
 
    gpu_1 = int(sys.argv[1])
    width_1 = sys.argv[2]
    height_1 = sys.argv[3]
    input_1 = sys.argv[4]
    output_1 = sys.argv[5]
 
    gpu_2 = int(sys.argv[6])
    width_2 = sys.argv[7]
    height_2 = sys.argv[8]
    input_2 = sys.argv[9]
    output_2 = sys.argv[10]
 
    create_threads(gpu_1, width_1, height_1, input_1, output_1, gpu_2, width_2, height_2, input_2, output_2)
