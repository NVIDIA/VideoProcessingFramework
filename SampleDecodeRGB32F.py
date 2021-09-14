#
# Copyright 2019 NVIDIA Corporation
# Copyright 2021 Videonetics Technology Private Limited
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
import pathlib
import shutil
import glob
from cv2 import cv2

if os.name == 'nt':
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    ffmpeg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "ffmpeg/bin")
    if cuda_path:
        os.add_dll_directory(cuda_path)
        os.add_dll_directory(ffmpeg_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 "build/PyNvCodec/Debug"))
import PyNvCodec as nvc
from enum import Enum
import numpy as np


class InitMode(Enum):
    # Decoder will be created with built-in demuxer.
    BUILTIN = 0,
    # Decoder will be created with standalone FFmpeg VPF demuxer.
    STANDALONE = 1


class DecodeStatus(Enum):
    # Decoding error.
    DEC_ERR = 0,
    # Frame was submitted to decoder.
    # No frames are ready for display yet.
    DEC_SUBM = 1,
    # Frame was submitted to decoder.
    # There's a frame ready for display.
    DEC_READY = 2


def remove_file_or_dir(inp_path, recursive=False):
    """param <path> could either be relative or absolute."""
    path = str(pathlib.Path(inp_path).absolute())
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        glob_list = glob.glob(path, recursive=recursive)
        if len(glob_list) > 0:
            for name in glob_list:
                if os.path.isfile(name) or os.path.islink(name):
                    os.remove(name)  # remove the file
                elif os.path.isdir(name):
                    shutil.rmtree(name)  # remove dir and all contains


def remove_dump_folder():
    remove_file_or_dir("session")


def get_folder(sub_folder) -> str:
    session_folder = os.path.join(os.getcwd(), sub_folder)
    if not os.path.exists(session_folder):
        try:
            os.makedirs(session_folder)
            print("{} folder created in {}".format(sub_folder, session_folder))
        except OSError as e:
            print(e)
            raise
    return session_folder + os.path.sep


def get_dump_folder() -> str:
    return get_folder("session")


def write_planar_rgb_32f(file_name, rgb_32f_planar):
    c, w, h = rgb_32f_planar.shape
    assert c == 3
    # logging.info(rgb_32f.shape)
    rgb_32f_planar *= 255.0
    img = np.ndarray(shape=(w, h, c), dtype=np.uint8, order="C")
    img[..., 0] = rgb_32f_planar[0]
    img[..., 1] = rgb_32f_planar[1]
    img[..., 2] = rgb_32f_planar[2]

    img = img.astype(np.uint8)
    cv2.imwrite(file_name, img)


def write_rgb_32f(file_name, rgb_32f):
    w, h, c = rgb_32f.shape
    assert c == 3
    # logging.info(rgb_32f.shape)
    #rgb_32f *= 255.0
    img = rgb_32f.astype(np.uint8)
    cv2.imwrite(file_name, img)


class NvDecoder:
    def __init__(self,
                 gpu_id: int,
                 enc_file: str,
                 dec_file: str,
                 dmx_mode=InitMode.BUILTIN):
        # Save mode, we will need this later
        self.init_mode = dmx_mode

        if self.init_mode == InitMode.STANDALONE:
            # Initialize standalone demuxer.
            self.nv_dmx = nvc.PyFFmpegDemuxer(enc_file)
            # Initialize decoder.
            self.nv_dec = nvc.PyNvDecoder(self.nv_dmx.Width(),
                                          self.nv_dmx.Height(),
                                          self.nv_dmx.Format(),
                                          self.nv_dmx.Codec(), gpu_id)
        else:
            # Initialize decoder with built-in demuxer.
            self.nv_dmx = None
            self.nv_dec = nvc.PyNvDecoder(enc_file, gpu_id)

        cspace, crange = self.nv_dec.ColorSpace(), self.nv_dec.ColorRange()
        print(cspace, crange)
        if nvc.ColorSpace.UNSPEC == cspace:
            cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == crange:
            crange = nvc.ColorRange.MPEG
        print('Color space: ', str(cspace))
        print('Color range: ', str(crange))

        self.cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)

        self.from_nv12_to_rgb8 = nvc.PySurfaceConverter(
            self.width(), self.height(), self.nv_dec.Format(),
            nvc.PixelFormat.RGB, gpu_id)
        self.from_rgb8_to_resized_rgb8 = nvc.PySurfaceResizer(
            416, 416, self.from_nv12_to_rgb8.Format(), gpu_id)
        self.from_resized_rgb8_to_rgb32F = nvc.PySurfaceConverter(
            416, 416, self.from_rgb8_to_resized_rgb8.Format(),
            nvc.PixelFormat.RGB_32F, gpu_id)
        self.from_rgb32F_to_rgb32F_planar = nvc.PySurfaceConverter(
            416, 416, self.from_resized_rgb8_to_rgb32F.Format(),
            nvc.PixelFormat.RGB_32F_PLANAR, gpu_id)
        self.rgb32F_downloader = nvc.PySurfaceDownloader(
            416, 416, nvc.PixelFormat.RGB_32F, gpu_id)
        self.rgb32F_planar_downloader = nvc.PySurfaceDownloader(
            416, 416, nvc.PixelFormat.RGB_32F_PLANAR, gpu_id)
        self.rgb32F_planar_surface_contiguous = nvc.Surface.Make(
            nvc.PixelFormat.RGB_32F_PLANAR_CONTIGUOUS, 416, 416, gpu_id)
        self.gpu_id = gpu_id

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
    def seek(self, seek_frame: int, seek_mode: nvc.SeekMode,
             seek_criteria: nvc.SeekCriteria) -> None:
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
                print('Seeking for the frame ', str(self.sk_frm))
                seek_ctx = nvc.SeekContext(int(self.sk_frm), self.seek_mode,
                                           self.seek_criteria)
                self.sk_frm = -1

                if not self.nv_dmx.Seek(seek_ctx, self.packet):
                    return status

                print('We are at frame with pts', str(seek_ctx.out_frame_pts))
            # Otherwise we just demux next packet.
            elif not self.nv_dmx.DemuxSinglePacket(self.packet):
                return status

            # Send encoded packet to Nvdec.
            # Nvdec is async so it may not return decoded frame immediately.
            frame_ready = self.nv_dec.DecodeFrameFromPacket(
                self.frame_nv12, self.packet)
            if frame_ready:
                self.num_frames_decoded += 1
                status = DecodeStatus.DEC_READY
            else:
                status = DecodeStatus.DEC_SUBM

            # Get last demuxed packet data.
            # It stores info such as pts, duration etc.
            self.nv_dmx.LastPacketData(self.packet_data)

            if verbose:
                print("frame pts (decode order)      :", self.packet_data.pts)
                print("frame dts (decode order)      :", self.packet_data.dts)
                print("frame pos (decode order)      :", self.packet_data.pos)
                print("frame duration (decode order) :",
                      self.packet_data.duration)
                print("")
        except Exception as e:
            print(getattr(e, 'message', str(e)))

        return status

    def decode_frame_builtin(self, verbose=False) -> DecodeStatus:
        status = DecodeStatus.DEC_ERR

        try:
            frame_ready = False
            frame_cnt_inc = 0

            if self.sk_frm >= 0:
                print('Seeking for the frame ', str(self.sk_frm))
                seek_ctx = nvc.SeekContext(int(self.sk_frm), self.seek_mode,
                                           self.seek_criteria)
                self.sk_frm = -1

                # frame_ready = self.nv_dec.DecodeSingleFrame(self.frame_nv12,
                #                                             seek_ctx, self.packet_data)
                nv12_surface = self.nv_dec.DecodeSingleSurface()
                frame_cnt_inc = seek_ctx.num_frames_decoded
                if not nv12_surface.Empty():
                    rgb8_surface = self.from_nv12_to_rgb8.Execute(
                        nv12_surface, self.cc_ctx)
                    if not rgb8_surface.Empty():
                        frame_ready = True
                        print("Here........")
            else:
                # frame_ready = self.nv_dec.DecodeSingleFrame(self.frame_nv12, self.packet_data)
                nv12_surface = self.nv_dec.DecodeSingleSurface()
                if not nv12_surface.Empty():
                    rgb8_surface = self.from_nv12_to_rgb8.Execute(
                        nv12_surface, self.cc_ctx)
                    if not rgb8_surface.Empty():
                        rgb8_resized_surface = self.from_rgb8_to_resized_rgb8.Execute(
                            rgb8_surface)
                        if not rgb8_resized_surface.Empty():
                            rgb32F_surface = self.from_resized_rgb8_to_rgb32F.Execute(
                                rgb8_resized_surface, None)
                            if not rgb32F_surface.Empty():
                                rgb32F_planar_surface = self.from_rgb32F_to_rgb32F_planar.Execute(
                                    rgb32F_surface, None)
                                if not rgb32F_planar_surface.Empty():
                                    print(
                                        f"rgb32F_planar_surface            | Width: {rgb32F_planar_surface.Width()} Height: {rgb32F_planar_surface.Height()} Pitch: {rgb32F_planar_surface.Pitch()} NumPlanes: {rgb32F_planar_surface.NumPlanes()} HostSize: {rgb32F_planar_surface.HostSize()}"
                                    )

                                    rgb32F_planar_surface.PlanePtr().Export(
                                        self.rgb32F_planar_surface_contiguous.
                                        PlanePtr().GpuMem(),
                                        self.rgb32F_planar_surface_contiguous.
                                        Pitch(), self.gpu_id)
                                    print(
                                        f"rgb32F_planar_surface_contiguous | Width: {self.rgb32F_planar_surface_contiguous.Width()} Height: {self.rgb32F_planar_surface_contiguous.Height()} Pitch: {self.rgb32F_planar_surface_contiguous.Pitch()} NumPlanes: {self.rgb32F_planar_surface_contiguous.NumPlanes()} HostSize: {self.rgb32F_planar_surface_contiguous.HostSize()}"
                                    )
                                    rgb_32f_frame = np.ndarray(
                                        shape=(3,
                                               rgb32F_planar_surface.Width(),
                                               rgb32F_planar_surface.Height()),
                                        dtype=np.float32)
                                    # print(rgb_32f_frame)
                                    success = self.rgb32F_planar_downloader.DownloadSingleSurface(
                                        rgb32F_planar_surface, rgb_32f_frame)
                                    if success:
                                        frame_ready = True
                                        frame_cnt_inc = 1
                                        rgb32F_surface
                                        rgb_32f_frame = np.reshape(
                                            rgb_32f_frame,
                                            (3, rgb32F_surface.Width(),
                                             rgb32F_surface.Height()))

                                        dump_folder = get_dump_folder()
                                        file_name = f"{dump_folder}/{self.num_frames_decoded:05d}.jpg"
                                        write_planar_rgb_32f(
                                            file_name, rgb_32f_frame)

            # Nvdec is sync in this mode so if frame isn't returned it means
            # EOF or error.
            if frame_ready:
                self.num_frames_decoded += 1
                status = DecodeStatus.DEC_READY

                if verbose:
                    print('Decoded ', frame_cnt_inc, ' frames internally')
            else:
                return status

            if verbose:
                print("frame pts (display order)      :", self.packet_data.pts)
                print("frame dts (display order)      :", self.packet_data.dts)
                print("frame pos (display order)      :", self.packet_data.pos)
                print("frame duration (display order) :",
                      self.packet_data.duration)
                print("")

        except Exception as e:
            print(getattr(e, 'message', str(e)))

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
    def decode(self,
               frames_to_decode=-1,
               verbose=False,
               dump_frames=True) -> None:
        # Main decoding cycle
        while (self.dec_frames() < frames_to_decode) if (
                frames_to_decode > 0) else True:
            status = self.decode_frame(verbose)
            if status == DecodeStatus.DEC_ERR:
                break
            elif dump_frames and status == DecodeStatus.DEC_READY:
                self.dump_frame()

        # Check if we need flush the decoder
        need_flush = (self.dec_frames() < frames_to_decode) if (
            frames_to_decode > 0) else True

        # Flush decoded frames queue.
        # This is needed only if decoder is initialized without built-in
        # demuxer and we're not limited in amount of frames to decode.
        while need_flush and (self.mode() == InitMode.STANDALONE):
            if not self.flush_frame(verbose):
                break
            elif dump_frames:
                self.dump_frame()


if __name__ == "__main__":

    print("This sample decodes input video to raw NV12 file on given GPU.")
    print("Usage: SampleDecode.py $gpu_id $input_file $output_file.")

    # if(len(sys.argv) < 4):
    #     print("Provide gpu ID, path to input and output files")
    #     exit(1)

    # gpu_id = int(sys.argv[1])
    # enc_filePath = sys.argv[2]
    # decFilePath = sys.argv[3]
    gpu_id = 0
    enc_filePath = "d:/WorkFiles/VideoProcessingFramework/videos/2min_1080p.mp4"
    decFilePath = "d:/WorkFiles/VideoProcessingFramework/videos/1.264"

    # s_src = nvc.Surface.Make(nvc.PixelFormat.RGB_32F_PLANAR, 416, 416, 0)
    # print(
    #     f"| Width: {s_src.Width()} Height: {s_src.Height()} Pitch: {s_src.Pitch()} NumPlanes: {s_src.NumPlanes()} HostSize: {s_src.HostSize()}"
    # )

    # s = nvc.Surface.Make(nvc.PixelFormat.RGB_32F_PLANAR_CONTIGUOUS, 416, 416,
    #                      0)
    # print(
    #     f"| Width: {s.Width()} Height: {s.Height()} Pitch: {s.Pitch()} NumPlanes: {s.NumPlanes()} HostSize: {s.HostSize()}"
    # )
    # s_src.PlanePtr().Export(s.PlanePtr().GpuMem(), s.Pitch(), gpu_id)
    # print(
    #     f"| Width: {s.Width()} Height: {s.Height()} Pitch: {s.Pitch()} NumPlanes: {s.NumPlanes()} HostSize: {s.HostSize()}"
    # )

    dec = NvDecoder(gpu_id, enc_filePath, decFilePath)
    dec.decode()

    exit(0)
