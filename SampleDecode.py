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
import PyNvCodec as nvc
import numpy as np
import sys

from enum import Enum

class DecodeStatus(Enum):
    ERROR = 0,
    FRAME_SUBMITTED = 1,
    FRAME_READY = 2

class Decoder:
    def __init__(self, gpu_id, enc_file, dec_file):
        # Initialize standalone demuxer to be able to seek through file
        self.nv_dmx = nvc.PyFFmpegDemuxer(enc_file)
        # Initialize decoder
        self.nv_dec = nvc.PyNvDecoder(self.nv_dmx.Width(), self.nv_dmx.Height(), 
                                      self.nv_dmx.Format(), self.nv_dmx.Codec(), 
                                      gpu_id)
        # Current frame being decoded
        self.curr_frame = -1
        # Total amount of decoded frames
        self.num_frames_decoded = 0
        # Numpy array to store decoded frames pixels
        self.frame_nv12 = np.ndarray(shape=(0), dtype=np.uint8)
        # Output file
        self.out_file = open(dec_file, "wb")
        # Encoded video packet
        self.packet = np.ndarray(shape=(0), dtype=np.uint8)
        # Encoded packet data
        self.packet_data = nvc.PacketData()

    # Returns video width in pixels
    def width(self):
        return self.nv_dmx.Width()

    # Returns video height in pixels
    def height(self):
        return self.nv_dmx.Height()

    # Returns number of decoded frames
    def num_frames(self):
        return self.num_frames_decoded

    # Returns current frame number
    def curr_frame(self):
        return self.curr_frame

    # Returns number of frames in video
    def stream_num_frames(self):
        return self.nv_dms.Numframes()

    # Seek for particular frame number
    def seek(self, seek_frame):
        seek_ctx = nvc.SeekContext(int(seek_frame))
        try:
            self.nv_dmx.Seek(seek_ctx)
            self.curr_frame = seek_frame
        except Exception as e:
            print(getattr(e, 'message', str(e)))

    # Decode single video frame
    # Returns 0 in case of error
    #         1 in case frame was submitted to HW but not decoded yet
    #         2 in case frame was submitted to HW and there's a decoded frame
    def decode_frame(self, verbose=False):
        status = DecodeStatus.ERROR
        try:
            # Demux packet from incoming file
            self.curr_frame += 1
            if not self.nv_dmx.DemuxSinglePacket(self.packet):
                return status

            # Decode it.
            # Decoder is async so it may not return decoded frame
            # the same moment the function is called.
            frame_ready = self.nv_dec.DecodeFrameFromPacket(self.frame_nv12, self.packet)
            if frame_ready:
                self.num_frames_decoded += 1
                status = DecodeStatus.FRAME_READY
            else:
                status = DecodeStatus.FRAME_SUBMITTED

            # Get last demuxed packet data.
            # It stores decoded frame data such as pts, duration etc.
            self.nv_dmx.LastPacketData(self.packet_data)

            if verbose:
                print("curr_frame:     ", self.curr_frame)
                print("frame pts:      ", self.packet_data.pts)
                print("frame dts:      ", self.packet_data.dts)
                print("frame pos:      ", self.packet_data.pos)
                print("frame duration: ", self.packet_data.duration)
                print("")

        except nvc.HwResetException:
            print('Continuing after HW decoder was reset')

        except Exception as e:
            print(getattr(e, 'message', str(e)))

        return status

    # Send empty packet to decoder to flush decoded frames queue
    def flush_frame(self, verbose=False):
        return self.nv_dec.FlushSingleFrame(self.packet)

    # Write current video frame to output file
    def dump_frame(self):
        bits = bytearray(self.frame_nv12)
        self.out_file.write(bits)

    # Decode all available video frames and write them to output file
    def decode(self, verbose=False):
        # Main decoding cycle
        while True:
            status = self.decode_frame(verbose)
            if status == DecodeStatus.ERROR:
                break
            elif status == DecodeStatus.FRAME_READY:
                self.dump_frame()

        # Now flush decoded frames queue
        while True:
            if not self.flush_frame(verbose):
                break
            else:
                self.dump_frame()

if __name__ == "__main__":

    print("This sample decodes input video to raw NV12 file on given GPU.")
    print("Usage: SampleDecode.py $gpu_id $input_file $output_file.")

    if(len(sys.argv) < 4):
        print("Provide gpu ID, path to input and output files")
        exit(1)

    gpu_id = int(sys.argv[1])
    enc_filePath = sys.argv[2]
    decFilePath = sys.argv[3]

    dec = Decoder(gpu_id, enc_filePath, decFilePath)
    dec.decode()