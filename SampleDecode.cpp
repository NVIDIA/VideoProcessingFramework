/*
 * Copyright 2020 NVIDIA Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PyNvCodec.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

int main(int argc, const char **argv) {
  if (argc < 4) {
    cerr << "Provide gpu ID, path to input and output files" << endl;
    return 1;
  } else {
    cout << "This sample decodes input video to raw NV12 file on given GPU."
         << endl;
    cout << "Usage: SampleDecode $gpu_id $input_file $output_file." << endl;
  }

  // GPU ID;
  stringstream ss;
  ss << argv[1];
  int gpu_id = -1;
  ss >> gpu_id;

  // Input file;
  string input(argv[2]);

  // Output file;
  string output(argv[3]);
  ofstream fout(output, std::ios::binary);

  PyNvDecoder nvDec(input, gpu_id);
  do {
    std::vector<uint8_t> frame;
    auto success = nvDec.DecodeSingleFrame(frame);
    if (!success) {
      break;
    } else {
      fout.write((const char *)frame.data(), frame.size());
    }
  } while (true);

  fout.close();
  return 0;
}