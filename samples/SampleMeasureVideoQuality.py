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
import queue
import sys
import os
import argparse
from pathlib import Path

if os.name == "nt":
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(";")
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

from math import log10, sqrt
import PyNvCodec as nvc
import numpy as np
from inspect import signature


def measure_psnr(gt: np.ndarray, dist: np.ndarray) -> float:
    """
    Measures the distance between frames using PSNR metric.

    Parameters
    ----------
    gt:     Ground Truth picture
    dist:   Distorted picture
    """
    mse = np.mean((gt - dist) ** 2)
    if mse == 0:
        return 100.0

    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def single_frame_encode_measure(
    raw_frame: np.ndarray,
    nvEnc: nvc.PyNvEncoder,
    nvDec: nvc.PyNvDecoder,
    vq_assess_func,
    frame_queue,
    fout,
) -> float:
    """
    Encodes single input frame and does visual quality estimation with given
    function.

    Parameters
    ----------
    raw_frame:      input raw frame in NV12 format
    nvEnc:          PyNvEncoder class to be used for encoding
    nvDec:          PyNvDecoder class to be used for getting recon frame
    vq_assess_func: Function to access visual quality
    frame_queue:    Queue which is used to store input raw frames temporarily
    fout            Handle to file used to store recon frames
    """

    # Video quality assessment function shall has certain signature.
    # In this sample PSNR is used for the sake of simplicity.
    sig = signature(vq_assess_func)
    assert str(sig) == "(gt: numpy.ndarray, dist: numpy.ndarray) -> float"

    recon_frame = np.ndarray(shape=(0), dtype=np.uint8)
    enc_packet = np.ndarray(shape=(0), dtype=np.uint8)
    enc_done = False

    if raw_frame.size:
        # Put frame into queue. This is required because Nvenc has some latency
        # to it and it doesn't return encoded frame immediately.
        frame_queue.put(raw_frame)
        # Encode it. Nvenc doesn't return recon frame (that's a HW limitation).
        # To over come it, we encode and then decode by hand.
        nvEnc.EncodeSingleFrame(raw_frame, enc_packet)
        if not enc_packet.size:
            # This isn't error. In the begining raw frames will be sent to HW
            # but not ready yet.
            return None
    else:
        # No more input frames. However, due to mentioned Nvenc latecy, there
        # are some frames left in the Nvenc queue. Hence we flush it.
        nvEnc.FlushSinglePacket(enc_packet)
        if not enc_packet.size:
            # All frames were sent to Nvenc and received.
            enc_done = True

    if not enc_done:
        # Encoder isn't done yet. Continue sending frames to HW.
        success = nvDec.DecodeFrameFromPacket(recon_frame, enc_packet)
    else:
        # All the frames are received from Nvenc. However, Nvdec is async by
        # design as well. Hence now we need to flush it as well to receive all
        # the frames we sent earlier.
        success = nvDec.FlushSingleFrame(recon_frame)

    if success:
        # Nvenc accept frames in display order and Nvdec returns frames in same
        # order as well. Hence no reordering here, usual in-order frame queue
        # is used to compare recon frame we got from Nvdec against ground truth
        # frame stored in queue.
        gt_frame = frame_queue.get()
        if gt_frame.size:
            # Store recon frames to disk if necessary.
            if fout:
                byte_array = bytearray(recon_frame)
                fout.write(byte_array)
            # Measure the distance between ground truth and recon frames.
            return vq_assess_func(gt_frame, recon_frame)
        else:
            # Something goes wrong if we're here. We've got a frame from Nvdec
            # but raw frame queue is empty which shall not happen.
            raise RuntimeError("unexpected empty queue.")
    else:
        return None


def main(gpu_id: int, input: str, output: str, width: int, height: int, verbose: bool):

    res = str(width) + "x" + str(height)
    decFile = open(input, "rb")
    frameSize = int(width * height * 3 / 2)
    frameQueue = queue.Queue()
    fout = open(output, "wb") if output else None

    nvDec = nvc.PyNvDecoder(
        width, height, nvc.PixelFormat.NV12, nvc.CudaVideoCodec.H264, gpu_id
    )

    nvEnc = nvc.PyNvEncoder(
        {
            "preset": "P4",
            "tuning_info": "high_quality",
            "codec": "h264",
            "profile": "high",
            "s": res,
            "bitrate": "10M",
        },
        gpu_id,
    )

    while True:
        rawFrame = np.fromfile(decFile, np.uint8, count=frameSize)
        score = single_frame_encode_measure(
            rawFrame, nvEnc, nvDec, measure_psnr, frameQueue, fout
        )
        if score:
            print("VQ score: ", "%.2f" % score)
            if verbose:
                print("Frame queue size: ", frameQueue.qsize())
        if not frameQueue.qsize():
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """This samples assesses Nvenc compression quality using PSNR metric.
           Input file must be NV12 raw.""",
        add_help=False,
    )
    parser.add_argument(
        "-g",
        type=int,
        required=True,
        help="GPU id, check nvidia-smi",
    )
    parser.add_argument(
        "-i",
        type=Path,
        required=True,
        help="Path to input raw file",
    )
    parser.add_argument(
        "-o",
        type=Path,
        required=False,
        help="Path to reconstructed raw file",
    )
    parser.add_argument(
        "-w",
        type=int,
        required=True,
        help="Raw file width",
    )
    parser.add_argument(
        "-h",
        type=int,
        required=True,
        help="Raw file height",
    )
    parser.add_argument("-v", default=False, action="store_true", help="Verbose mode")

    args = parser.parse_args()
    main(
        args.g,
        args.i.as_posix(),
        args.o.as_posix() if args.o else None,
        args.w,
        args.h,
        args.v,
    )
