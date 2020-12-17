import av
import io
from av.bitstream import BitStreamFilter, BitStreamFilterContext

import PyNvCodec as nvc
import numpy as np
import sys

#This class implements demuxer which uses PyAV to extract Annex.B
#H.264 and H.265 NAL Units.


class Pyav_Demuxer:
    def __init__(self, input_file):
        #Create input container and input video stream
        self.input_container = av.open(input_file)
        self.in_stream = self.input_container.streams.video[0]

        #Create bitstream filter context
        self.bsf_name = 'h264_mp4toannexb' if self.in_stream.codec_context.name == 'h264' else 'hevc_mp4toannexb'
        self.bsfc = BitStreamFilterContext(self.bsf_name)

        #Create raw byte IO instead of file IO
        #Fake the extension to satisfy FFmpeg muxer
        self.byte_io = io.BytesIO()
        self.byte_io.name = 'muxed.h264' if self.bsf_name == 'h264_mp4toannexb' else 'muxed.h265'

        #Create output container which will create Annex.B NAL Units
        self.out_container = av.open(self.byte_io, 'wb')
        self.out_stream = self.out_container.add_stream(template=self.in_stream)

    #Will return single encoded H.264 / H.265 packet
    def get_packet(self):
        for packet in self.input_container.demux(self.in_stream):
            if packet.dts is None:
                continue

            for out_packet in self.bsfc(packet):
                self.out_container.mux_one(out_packet)
                self.byte_io.flush()

                enc_packet = np.frombuffer(buffer=self.byte_io.getvalue(), dtype=np.uint8)

                #Truncate byte IO so that it stores just single packet
                self.byte_io.seek(0)
                self.byte_io.truncate()

                return enc_packet

    def width(self):
        return self.in_stream.codec_context.width

    def height(self):
        return self.in_stream.codec_context.height

    #Only H.264 and H.265 are supported for now
    def cuda_video_codec(self):
        return nvc.CudaVideoCodec.H264 if self.bsf_name == 'h264_mp4toannexb' else nvc.CudaVideoCodec.HEVC

    #Only NV12 and YUV444 are supported for now
    def pixel_format(self):
        if self.in_stream.codec_context.pix_fmt == 'yuv420p':
            return nvc.PixelFormat.NV12
        elif self.n_stream.codec_context.pix_fmt == 'yuv444p':
            return nvc.PixelFormat.YUV444
        else:
            return nvc.PixelFormat.UNDEFINED


def decode(input_file, output_file):
    demuxer = Pyav_Demuxer(input_file)
    dec_file = open(output_file, 'wb')

    nvDec = nvc.PyNvDecoder(demuxer.width(), demuxer.height(), demuxer.pixel_format(), nvc.CudaVideoCodec.H264, 0)
    rawFrame = np.ndarray(shape=(0), dtype=np.uint8)

    #Main decoding cycle
    while True:
        enc_packet = demuxer.get_packet()

        #This means EOF, no more video packets
        if not enc_packet.size:
            break

        if nvDec.DecodeFrameFromPacket(rawFrame, enc_packet):
            bits = bytearray(rawFrame)
            dec_file.write(bits)

    # Now we flush decoder to emtpy decoded frames queue.
    while True:
        if nvDec.FlushSingleFrame(rawFrame):
            bits = bytearray(rawFrame)
            dec_file.write(bits)
        else:
            break

    dec_file.close()


def main():
    decode("/home/roman/Videos/bbb_sunflower_1080p_30fps_normal.mp4", "/home/roman/Videos/bbb_sunflower_1080p_30fps_normal.yuv")
    return 0


if __name__ == "__main__":
    main()
