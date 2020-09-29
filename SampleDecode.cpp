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

#include "VPF_Classes.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace VPF;
using namespace std;

int main(int argc, char **argv) {
  map<string, string> cli_options;
  for (int i = 1; i < argc - 1; i+=2) {
    cli_options[argv[i]] = argv[i + 1];
  }

  // Parse input file path;
  string pathToFile;
  auto it = cli_options.find("-i");
  if (cli_options.end() != it) {
    pathToFile = it->second;
    cli_options.erase(it);
  } else {
    cerr << "Provide path to input file as -i %filepath%" << endl;
    return 1;
  }

  // Parse output file path;
  string outputPath;
  it = cli_options.find("-o");
  if (cli_options.end() != it) {
    outputPath = it->second;
    cli_options.erase(it);
  } else {
    cerr << "Provide path to output file as -o %filepath%" << endl;
    return 1;
  }

  // Parse GPU ID;
  uint32_t gpuID;
  it = cli_options.find("-gpu");
  if (cli_options.end() != it) {
    stringstream ss;
    ss << it->second;
    ss >> gpuID;
    cli_options.erase(it);
  } else {
    cerr << "Provide GPU ID as -gpu %gpu_id%" << endl;
    return 1;
  }

  // Treat the rest as FFmpeg options;
  VpfNvDecoderContext ctx(pathToFile, gpuID, cli_options);
  VpfNvDecoder dec(ctx);

  ofstream fout(outputPath, ios::trunc | ios::binary);
  if (!fout.is_open()) {
    cerr << "Can't open output file " << outputPath << endl;
    return 1;
  }

  while (true) {
    VpfNvDecoderArgs args;
    if (!dec.DecodeSingleFrame(args)) {
      cerr << args.errorMessage;
      return 1;
    }

    fout.write((const char *)args.decodedFrame.get(), args.decodedFrameSize);
  }

  return 0;
}