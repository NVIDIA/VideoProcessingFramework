import PyNvCodec as nvc
import numpy as np
import unittest
from os.path import join, dirname


def test_issue_455():
    gpuID = 0

    nvEnc = nvc.PyNvEncoder({'bitrate': '30K', 'fps': '10', 'codec': 'hevc', 's': '256x256'}, gpuID)
    nvDec = nvc.PyNvDecoder(256, 256, nvc.PixelFormat.NV12, nvc.CudaVideoCodec.HEVC, gpuID)

    rawFrame = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)

    print('Raw frame size is ' + str(rawFrame.size) + ' bytes.')

    encodedFrame = np.ndarray(shape=(0), dtype=np.uint8)

    count, success = 0, False

    while success is not True and count < 10:
        success = nvEnc.EncodeSingleFrame(rawFrame, encodedFrame, sync=False)
        count += 1

    print('Encoded frame size is ' + str(encodedFrame.size) + ' bytes.')

    exception_raised = False
    try:
        success = nvDec.DecodeSingleFrame(encodedFrame)
    except Exception as ex:
        exception_raised = True
        assert ("Tried to call DecodeSurface/DecodeFrame on a Decoder that has been initialized without a built-in "
                "demuxer. Please use DecodeSurfaceFromPacket/DecodeFrameFromPacket instead or intialize the decoder"
                " with a demuxer when decoding from a file" == str(ex))
    assert exception_raised

    decodedFrame = np.ndarray(shape=(0), dtype=np.uint8)
    success = nvDec.DecodeFrameFromPacket(decodedFrame, encodedFrame)


@unittest.skip('Skipping because still causing segfault due to built-in demuxer being NULL')
def test_issue_457():
    encFilePath = join(dirname(__file__), "test_res_change.h264")
    nvDec = nvc.PyFfmpegDecoder(encFilePath, {}, 1)
    nvDec.GetMotionVectors()
