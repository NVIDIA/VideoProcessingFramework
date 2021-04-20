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
import SampleDecode as dec
import SampleEncode as enc
import sys
 
if __name__ == "__main__":

    gpu_id = int(sys.argv[1])
    input = sys.argv[2]
    output = sys.argv[3]

    dec_file = "temp.nv12"

    decoder = dec.NvDecoder(gpu_id, input, dec_file)
    decoder.decode(verbose=True)
    num_frames = decoder.num_frames()
    print (str(num_frames + 1), ' frames decoded.')

    enc.encode(gpu_id, dec_file, output, 1920, 1080)