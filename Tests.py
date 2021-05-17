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

import SampleDecode as dec
import SampleEncode as enc
import PyNvCodec as nvc
import time
 
if __name__ == "__main__":

    gpu_id = int(sys.argv[1])
    input = sys.argv[2]
    output = sys.argv[3]

    dec_file = "temp.nv12"

    decoder = dec.NvDecoder(gpu_id, input, dec_file, dec.InitMode.STANDALONE)
    decoder.seek(7, nvc.SeekMode.PREV_KEY_FRAME)
    decoder.decode(frames_to_decode = 256, verbose=True)
    num_frames = decoder.dec_frames()
    print (str(num_frames), ' frames decoded.')

    time.sleep(1)

    decoder = dec.NvDecoder(gpu_id, input, dec_file, dec.InitMode.BUILTIN)
    decoder.seek(11, nvc.SeekMode.PREV_KEY_FRAME)
    decoder.decode(frames_to_decode = 256, verbose=True)
    num_frames = decoder.dec_frames()
    print (str(num_frames), ' frames decoded.')

    enc.encode(gpu_id, dec_file, output, 1920, 1080)

    exit(0)