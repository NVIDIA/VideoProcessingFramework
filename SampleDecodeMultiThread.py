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
from multiprocessing import Process, current_process
import PyNvCodec as nvc
import numpy as np
import sys
 
def run(gpuID, encFile, outFile):
    try:
        decFile = open(outFile, "wb")
 
        nvDec = nvc.PyNvDecoder(encFile, gpuID)
        w, h = nvDec.Width, nvDec.Height()
        nvCvt = nvc.PySurfaceConverter(w, h, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, gpuID)
        nvDwl = nvc.PySurfaceDownloader(w, h, nvc.PixelFormat.YUV420, gpuID)
 
        while True:
            rawSurface = nvDec.DecodeSingleSurface()
            if (rawSurface.Empty()):
                print('failed to decode video frame')
                break
 
            cvtSurface = nvCvt.Execute(rawSurface)
            if (cvtSurface.Empty()):
                print('failed to do color conversion')
                break
 
            #Amount of memory in RAM we need to store YUV420 surface
            frameSize = cvtSurface.HostSize()
            rawFrame = np.ndarray(shape=(frameSize), dtype=np.uint8)
 
            success = nvDwl.DownloadSingleSurface(cvtSurface, rawFrame)
            if not (success):
                print('failed to download surface')
                break
 
            frameByteArray = bytearray(rawFrame)
            decFile.write(frameByteArray)
 
    except Exception as e:
        print(getattr(e, 'message', str(e)))
        decFile.close()
 
def create_procs(gpu_id1, input_file1, output_file1,
                 gpu_id2, input_file2, output_file2):
 
    proc1 = Process(target=run, name='DecProcess1', args=(gpu_id1, input_file1, output_file1))
    proc2 = Process(target=run, name='DecProcess2', args=(gpu_id2, input_file2, output_file2))
 
    proc1.start()
    proc2.start()
 
    proc1.join()
    proc2.join()
 
if __name__ == "__main__":
    print("This sample decodes 2 videos simultaneously, and save to raw YUV files.")
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
 
    create_procs(gpu_1, input_1, output_1, gpu_2, input_2, output_2)
