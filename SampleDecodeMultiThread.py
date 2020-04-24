#
# Copyright 2019 NVIDIA Corporation
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
from threading import Thread
import PyNvCodec as nvc
import numpy as np
import sys

class DecoderThread(Thread):   
    def __init__(self, gpuID, encFile, outFile):
        try:
            Thread.__init__(self)
            self.decFile = open(outFile, "wb")
        
            self.nvDec = nvc.PyNvDecoder(encFile, gpuID)
            half_w = int(self.nvDec.Width() / 2)
            half_h = int(self.nvDec.Height() / 2)
        
            self.nvRes = nvc.PySurfaceResizer(half_w, half_h, nvc.PixelFormat.NV12, gpuID)
            self.nvCvt = nvc.PySurfaceConverter(half_w, half_h, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, gpuID)
            self.nvDwl = nvc.PySurfaceDownloader(half_w, half_h, nvc.PixelFormat.YUV420, gpuID)

        except Exception as e:
            print(getattr(e, 'message', str(e)))
    
    def run(self):
        try:
            while True:
                rawSurfaceNV12 = self.nvDec.DecodeSingleSurface()
                if (rawSurfaceNV12.Empty()):
                    print('failed to decode video frame')
                    break

                smallerNV12 = self.nvRes.Execute(rawSurfaceNV12)
                if (smallerNV12.Empty()):
                    print('failed to resize video frame')
                    break
    
                rawSurfaceYUV420 = self.nvCvt.Execute(smallerNV12)
                if (rawSurfaceYUV420.Empty()):
                    print('failed to do color conversion')
                    break
                
                #Amount of memory in RAM we need to store YUV420 surface
                frameSize = rawSurfaceYUV420.HostSize()
                rawFrameYUV420 = np.ndarray(shape=(frameSize), dtype=np.uint8)
                
                success = self.nvDwl.DownloadSingleSurface(rawSurfaceYUV420, rawFrameYUV420)
                if not (success):
                    print('failed to download surface')
                    break

                frameByteArray = bytearray(rawFrameYUV420)
                self.decFile.write(frameByteArray)
        except Exception as e:
            print(getattr(e, 'message', str(e)))
            self.decFile.close()

def create_threads(gpu_id1, input_file1, output_file1,
                   gpu_id2, input_file2, output_file2):

    thread1 = DecoderThread(gpu_id1, input_file1, output_file1)
    thread2 = DecoderThread(gpu_id2, input_file2, output_file2)

    thread1.start()
    thread2.start()
 
if __name__ == "__main__":
    print("This sample decodes 2 videos simultaneously, resize to half resolution and save to raw YUV files.")
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