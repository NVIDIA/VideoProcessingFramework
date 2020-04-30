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
import PyNvCodec as nvc
import numpy as np
import sys

from threading import Thread
 
class Worker(Thread):
    def __init__(self, gpuID, encFile, outFile):
        Thread.__init__(self)

        self.decFile = open(outFile, "wb")
        
        self.nvDec = nvc.PyNvDecoder(encFile, gpuID)
        
        width, height = self.nvDec.Width(), self.nvDec.Height()
        hwidth, hheight = int(width / 2), int(height / 2)

        self.nvCvt = nvc.PySurfaceConverter(width, height, self.nvDec.Format(), nvc.PixelFormat.YUV420, gpuID)
        self.nvRes = nvc.PySurfaceResizer(hwidth, hheight, self.nvCvt.Format(), gpuID)
        self.nvDwn = nvc.PySurfaceDownloader(hwidth, hheight, self.nvRes.Format(), gpuID)
 
    def run(self):
        try:
            while True:
                rawSurface = self.nvDec.DecodeSingleSurface()
                if (rawSurface.Empty()):
                    print('No more video frames')
                    break
 
                cvtSurface = self.nvCvt.Execute(rawSurface)
                if (cvtSurface.Empty()):
                    print('Failed to do color conversion')
                    break

                resSurface = self.nvRes.Execute(cvtSurface)
                if (resSurface.Empty()):
                    print('Failed to resize surface')
                    break
 
                rawFrame = np.ndarray(shape=(resSurface.HostSize()), dtype=np.uint8)
                success = self.nvDwn.DownloadSingleSurface(resSurface, rawFrame)
                if not (success):
                    print('Failed to download surface')
                    break
 
                bits = bytearray(rawFrame)
                self.decFile.write(bits)
 
        except Exception as e:
            print(getattr(e, 'message', str(e)))
            decFile.close()
 
def create_threads(gpu_id1, input_file1, output_file1,
                 gpu_id2, input_file2, output_file2):
 
    th1 = Worker(gpu_id1, input_file1, output_file1)
    th2 = Worker(gpu_id2, input_file2, output_file2)
 
    th1.start()
    th2.start()
 
    th1.join()
    th2.join()
 
if __name__ == "__main__":
    print("This sample decodes 2 videos simultaneously, resize them to 1/4 of initial size and save to raw YUV files.")
    print("Usage: SampleDecode.py $gpu_id1 $input_file1 $output_file_1 $gpu_id2 $input_file2 $output_file2")
 
    if(len(sys.argv) < 7):
        print("Provide input CLI arguments as shown above")
        exit(1)
 
    gpu_1 = int(sys.argv[1])
    input_1 = sys.argv[2]
    output_1 = sys.argv[3]
 
    gpu_2 = int(sys.argv[4])
    input_2 = sys.argv[5]
    output_2 = sys.argv[6]
 
    create_threads(gpu_1, input_1, output_1, gpu_2, input_2, output_2)
