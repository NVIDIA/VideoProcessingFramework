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

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
import PyNvCodec as nvc
import os
import tqdm
import argparse
from pathlib import Path
from enum import Enum

import numpy as np
import logging

logger = logging.getLogger(__file__)

if os.name == "nt":
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        logger.error("CUDA_PATH environment variable is not set.")
        logger.error("Can't set CUDA DLLs search path.")
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(";")
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        logger.error("PATH environment variable is not set.")
        exit(1)


class InitMode(Enum):
    # Decoder will be created with built-in demuxer.
    BUILTIN = (0,)
    # Decoder will be created with standalone FFmpeg VPF demuxer.
    STANDALONE = 1


class DecodeStatus(Enum):
    # Decoding error.
    DEC_ERR = (0,)
    # Frame was submitted to decoder.
    # No frames are ready for display yet.
    DEC_SUBM = (1,)
    # Frame was submitted to decoder.
    # There's a frame ready for display.
    DEC_READY = 2


class NvDecoder:
    def __init__(
        self,
        gpu_id: int,
        enc_file: str,
        dec_file: str,
        dmx_mode=InitMode.STANDALONE,
    ):
        # Save mode, we will need this later
        self.init_mode = dmx_mode

        if self.init_mode == InitMode.STANDALONE:
            # Initialize standalone demuxer.
            self.nv_dmx = nvc.PyFFmpegDemuxer(enc_file)
            # Initialize decoder.
            self.nv_dec = nvc.PyNvDecoder(
                self.nv_dmx.Width(),
                self.nv_dmx.Height(),
                self.nv_dmx.Format(),
                self.nv_dmx.Codec(),
                gpu_id,
            )
        else:
            # Initialize decoder with built-in demuxer.
            self.nv_dmx = None
            self.nv_dec = nvc.PyNvDecoder(enc_file, gpu_id)

        # Frame to seek to next time decoding function is called.
        # Negative values means 'don't use seek'.  Non-negative values mean
        # seek frame number.
        self.sk_frm = int(-1)
        # Total amount of decoded frames
        self.num_frames_decoded = int(0)
        # Numpy array to store decoded frames pixels
        self.frame_nv12 = np.ndarray(shape=(0), dtype=np.uint8)
        # Output file
        self.out_file = open(dec_file, "wb")
        # Encoded video packet
        self.packet = np.ndarray(shape=(0), dtype=np.uint8)
        # Encoded packet data
        self.packet_data = nvc.PacketData()
        # Seek mode
        self.seek_mode = nvc.SeekMode.PREV_KEY_FRAME

    # Returns decoder creation mode
    def mode(self) -> InitMode:
        return self.init_mode

    # Returns video width in pixels
    def width(self) -> int:
        if self.mode() == InitMode.STANDALONE:
            return self.nv_dmx.Width()
        else:
            return self.nv_dec.Width()

    # Returns video height in pixels
    def height(self) -> int:
        if self.mode() == InitMode.STANDALONE:
            return self.nv_dmx.Height()
        else:
            return self.nv_dec.Height()

    # Returns number of decoded frames.
    def dec_frames(self) -> int:
        return self.num_frames_decoded

    # Returns frame rate
    def framerate(self) -> float:
        if self.mode() == InitMode.STANDALONE:
            return self.nv_dmx.Framerate()
        else:
            return self.nv_dec.Framerate()

    # Returns average frame rate
    def avg_framerate(self) -> float:
        if self.mode() == InitMode.STANDALONE:
            return self.nv_dmx.AvgFramerate()
        else:
            return self.nv_dec.AvgFramerate()

    # Returns True if video has various frame rate, False otherwise
    def is_vfr(self) -> bool:
        if self.mode() == InitMode.STANDALONE:
            return self.nv_dmx.IsVFR()
        else:
            return self.nv_dec.IsVFR()

    # Returns number of frames in video.
    def stream_num_frames(self) -> int:
        if self.mode() == InitMode.STANDALONE:
            return self.nv_dmx.Numframes()
        else:
            return self.nv_dec.Numframes()

    # Seek for particular frame number.
    def seek(
        self,
        seek_frame: int,
        seek_mode: nvc.SeekMode,
        seek_criteria: str = "ts",
    ) -> None:
        # Next time we decode frame decoder will seek for this frame first.
        self.sk_frm = seek_frame
        self.seek_mode = seek_mode
        self.seek_criteria = seek_criteria
        self.num_frames_decoded = 0

    def decode_frame_standalone(self, verbose=False) -> DecodeStatus:
        status = DecodeStatus.DEC_ERR

        try:
            # Check if we need to seek first.
            if self.sk_frm >= 0:
                if self.sk_frm.is_integer():
                    self.sk_frm = int(self.sk_frm)
                logger.info(f"Seeking for the {self.seek_criteria} {self.sk_frm}")
                seek_ctx = nvc.SeekContext(
                    **{"seek_" + self.seek_criteria: self.sk_frm},
                    mode=self.seek_mode
                )
                self.sk_frm = -1

                if not self.nv_dmx.Seek(seek_ctx, self.packet):
                    return status

                logger.info("We are at frame with pts {str(seek_ctx.out_frame_pts)}")
            # Otherwise we just demux next packet.
            elif not self.nv_dmx.DemuxSinglePacket(self.packet):
                return status

            # Send encoded packet to Nvdec.
            # Nvdec is async so it may not return decoded frame immediately.
            frame_ready = self.nv_dec.DecodeFrameFromPacket(
                self.frame_nv12, self.packet
            )
            if frame_ready:
                self.num_frames_decoded += 1
                status = DecodeStatus.DEC_READY
            else:
                status = DecodeStatus.DEC_SUBM

            # Get last demuxed packet data.
            # It stores info such as pts, duration etc.
            self.nv_dmx.LastPacketData(self.packet_data)

            if verbose:
                logger.info(f"frame pts (decode order)      :{self.packet_data.pts}")
                logger.info(f"frame dts (decode order)      :{self.packet_data.dts}")
                logger.info(f"frame pos (decode order)      :{self.packet_data.pos}")
                logger.info(
                    f"frame duration (decode order) :{self.packet_data.duration}"
                )
        except Exception as e:
            logger.info(f"{getattr(e, 'message', str(e))}")

        return status

    def decode_frame_builtin(self, verbose=False) -> DecodeStatus:
        status = DecodeStatus.DEC_ERR

        try:
            frame_ready = False
            frame_cnt_inc = 0

            if self.sk_frm >= 0:
                logger.info("Seeking for the frame ", str(self.sk_frm))
                seek_ctx = nvc.SeekContext(
                    int(self.sk_frm), self.seek_mode, self.seek_criteria
                )
                self.sk_frm = -1

                frame_ready = self.nv_dec.DecodeSingleFrame(
                    self.frame_nv12, seek_ctx, self.packet_data
                )
                frame_cnt_inc = seek_ctx.num_frames_decoded
            else:
                frame_ready = self.nv_dec.DecodeSingleFrame(
                    self.frame_nv12, self.packet_data
                )
                frame_cnt_inc = 1

            # Nvdec is sync in this mode so if frame isn't returned it means
            # EOF or error.
            if frame_ready:
                self.num_frames_decoded += 1
                status = DecodeStatus.DEC_READY

                if verbose:
                    logger.info(f"Decoded {frame_cnt_inc} frames internally")
            else:
                return status

            if verbose:
                logger.info(f"frame pts (display order)      :{self.packet_data.pts}")
                logger.info(f"frame dts (display order)      :{self.packet_data.dts}")
                logger.info(f"frame pos (display order)      :{self.packet_data.pos}")
                logger.info(
                    f"frame duration (display order) :{self.packet_data.duration}"
                )

        except Exception as e:
            logger.info(f"{getattr(e, 'message', str(e))}")

        return status

    # Decode single video frame
    def decode_frame(self, verbose=False) -> DecodeStatus:
        if self.mode() == InitMode.STANDALONE:
            return self.decode_frame_standalone(verbose)
        else:
            return self.decode_frame_builtin(verbose)

    # Send empty packet to decoder to flush decoded frames queue.
    def flush_frame(self, verbose=False) -> None:
        ret = self.nv_dec.FlushSingleFrame(self.frame_nv12)
        if ret:
            self.num_frames_decoded += 1

        return ret

    # Write current video frame to output file.
    def dump_frame(self) -> None:
        bits = bytearray(self.frame_nv12)
        self.out_file.write(bits)

    # Decode all available video frames and write them to output file.
    def decode(self, frames_to_decode=-1, verbose=False, dump_frames=True) -> None:
        # Main decoding cycle
        pbar = tqdm.tqdm(total=frames_to_decode, ascii=False, unit=" frames")
        pbar.set_description("Decoding ")
        while (
            (self.dec_frames() < frames_to_decode) if (frames_to_decode > 0) else True
        ):
            status = self.decode_frame(verbose)
            if status == DecodeStatus.DEC_ERR:
                break
            elif dump_frames and status == DecodeStatus.DEC_READY:
                self.dump_frame()
            pbar.update()

        # Check if we need flush the decoder
        need_flush = (
            (self.dec_frames() < frames_to_decode) if (frames_to_decode > 0) else True
        )

        # Flush decoded frames queue.
        # This is needed only if decoder is initialized without built-in
        # demuxer and we're not limited in amount of frames to decode.
        while need_flush and (self.mode() == InitMode.STANDALONE):
            if not self.flush_frame(verbose):
                break
            elif dump_frames:
                self.dump_frame()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "This sample decodes input video to raw NV12 file on given GPU."
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        required=True,
        help="GPU id, check nvidia-smi",
    )
    parser.add_argument(
        "-e",
        "--encoded-file-path",
        type=Path,
        required=True,
        help="Encoded video file (read from)",
    )
    parser.add_argument(
        "-r",
        "--raw-file-path",
        type=Path,
        required=True,
        help="Raw NV12 video file (write to)",
    )
    parser.add_argument(
        "-v", "--verbose", default=False, action="store_true", help="Verbose"
    )

    args = parser.parse_args()

    dec = NvDecoder(
        args.gpu_id,
        args.encoded_file_path.as_posix(),
        args.raw_file_path.as_posix(),
    )
    dec.decode(verbose=args.verbose)

    exit(0)
