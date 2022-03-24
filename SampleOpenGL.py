#
# Copyright 2022 @Yves33, @sandhawalia
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
import os
import sys
import time
import argparse
import numpy as np
import pycuda
from pathlib import Path
import ctypes

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import PyNvCodec as nvc
from utils import get_logger


logger = get_logger(__file__)


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


class FPSLogger:
    def __init__(self, interval):
        self.interval = interval
        self.framecount = 0
        self.seconds = time.time_ns() / 1e9

    def log(self, titlebar=True, fmt="fps : {0}"):
        self.framecount += 1
        if self.seconds + self.interval < time.time_ns() / 1e9:
            self.fps = self.framecount / self.interval
            self.framecount = 0
            self.seconds = time.time_ns() / 1e9
            if titlebar:
                glutSetWindowTitle(fmt.format(self.fps))
            else:
                logger.info(fmt.format(self.fps))


class OpenGLApplication:
    def __init__(
        self,
        encoded_video_file: str,
        gpu_id: int = 0,
        width: int = 500,
        height: int = 500,
    ):

        self.cpu = False

        # Setup up display window
        self.setup_display(width, height)
        # Loading drivers + compiling shaders, Done once?
        self.setup_opengl()
        # Setup decoder and downsampler
        self.setup_vpf(encoded_video_file, gpu_id)
        #
        self.create_textures()
        #
        self.cuda_gl_handshake()

    def setup_display(self, width, height):
        logger.info(f"Setting up display {width}x{height}")
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        glutCreateWindow(b"Simple PyOpenGL example")
        logger.info(f"Done setting up display {width}x{height}")

    def setup_opengl(self):
        self.program = self.compile_shaders()
        import pycuda.autoinit
        import pycuda.gl.autoinit

        self.vao = GLuint()
        glCreateVertexArrays(1, self.vao)

    def setup_vpf(self, encoded_video_file, gpu_id):

        self.nv_dec = nvc.PyNvDecoder(encoded_video_file, gpu_id)
        self.width, self.height = self.nv_dec.Width(), self.nv_dec.Height()
        self.cc_ctx = nvc.ColorspaceConversionContext(
            self.nvDec.ColorSpace(), self.nv_dec.ColorRange()
        )
        if self.nv_dec.ColorSpace() != nvc.ColorSpace.BT_709:
            self.nv_yuv = nvc.PySurfaceConverter(
                self.width, self.height, self.nv_dec.Format(), nvc.PixelFormat.YUV420, 0
            )
            self.nv_cvt = nvc.PySurfaceConverter(
                self.width, self.height, self.nvYuv.Format(), nvc.PixelFormat.RGB, 0
            )
        else:
            self.nv_yuv = None
            self.nv_cvt = nvc.PySurfaceConverter(
                self.width, self.height, self.nv_dec.Format(), nvc.PixelFormat.RGB, 0
            )
        self.nv_down = nvc.PySurfaceDownloader(
            self.width, self.height, self.nv_cvt.Format(), 0
        )
        self.data = np.zeros((self.width * self.height, 3), np.uint8)

    def create_textures(self):

        ## create texture for GL display
        self.texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            self.width,
            self.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.cuda_img = pycuda.gl.RegisteredImage(
            int(self.texture), GL_TEXTURE_2D, pycuda.gl.graphics_map_flags.NONE
        )  # WRITE_DISCARD)

    def cuda_gl_handshake(self):
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)
        glBufferData(GL_ARRAY_BUFFER, self.data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        import pycuda.autoinit
        import pycuda.gl.autoinit

        self.cuda_pbo = pycuda.gl.RegisteredBuffer(int(self.pbo))

        self.vao = 0
        glGenVertexArrays(1, self.vao)
        glBindVertexArray(self.vao)

    def compile_shaders(self):
        vertex_shader_source = """
        #version 450 core
        out vec2 uv;
        void main( void)
        {
            // Declare a hard-coded array of positions
            const vec2 vertices[4] = vec2[4](vec2(-0.5,  0.5),
                                             vec2( 0.5,  0.5),
                                             vec2( 0.5, -0.5),
                                             vec2(-0.5, -0.5));
            // Index into our array using gl_VertexID
            uv=vertices[gl_VertexID]+vec2(0.5,0.5);
            gl_Position = vec4(2*vertices[gl_VertexID],1.0,1.0);
            }
        """

        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)

        fragment_shader_source = """
        #version 450 core
        uniform sampler2D s;
        in vec2 uv;
        out vec4 color;
        void main(void)
        {
            color = vec4(texture(s, uv));
        }
        """

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)

        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        # --- Clean up now that we don't need these shaders anymore.
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return program

    def render(self, method: str = "GPUTEX"):
        glClearBufferfv(GL_COLOR, 0, (0, 0, 0))
        ## bind program
        glUseProgram(self.program)
        ## get one frame
        rawSurface = self.nv_dec.DecodeSingleSurface()
        if self.nv_yuv != None:
            yuvSurface = self.nv_yuv.Execute(rawSurface, self.cc_ctx)
            cvtSurface = self.nv_cvt.Execute(yuvSurface, self.cc_ctx)
        else:
            cvtSurface = self.nv_cvt.Execute(rawSurface, self.cc_ctx)
        ## texture update through cpu and system memory
        if self.cpu:
            ## Download surface data to CPU, then update GL texture with these data
            success = self.nv_down.DownloadSingleSurface(cvtSurface, self.data)
            if not success:
                logger.warn("Could not download Cuda Surface to CPU")
                return
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                self.width,
                self.height,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                self.data,
            )
        else:
            ## cuda copy from surface.Plane_Ptr() to pbo, then update texture from PBO
            src_plane = cvtSurface.PlanePtr()
            buffer_mapping = self.cuda_pbo.map()
            buffptr, buffsize = buffer_mapping.device_ptr_and_size()
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(src_plane.GpuMem())
            cpy.set_dst_device(buffptr)
            cpy.width_in_bytes = src_plane.Width()
            cpy.src_pitch = src_plane.Pitch()
            cpy.dst_pitch = self.width * 3
            cpy.height = src_plane.Height()
            cpy(aligned=True)
            # pycuda.driver.Context.synchronize() ## not required?
            buffer_mapping.unmap()
            ## opengl update texture from pbo
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(self.pbo))
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                self.width,
                self.height,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                ctypes.c_void_p(0),
            )

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        ## send uniforms to program and draw quad
        glUniform(glGetUniformLocation(self.program, b"s"), 0)
        glDrawArrays(GL_QUADS, 0, 4)
        ## Display
        glutSwapBuffers()

    def keypressed(self, key, x, y):
        if key.decode("utf-8") == "q":
            glutLeaveMainLoop()
        elif key.decode("utf-8") == "c":
            self.cpu = True
        elif key.decode("utf-8") == "g":
            self.cpu = False

    def animate(self):
        glutPostRedisplay()

    def run(self, verbose: bool = False):

        glutIdleFunc(self.animate)
        glutDisplayFunc(self.render)
        glutMainLoop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "This sample decodes input video to OpenGL Texture.\n"
        + "Requires the GL Utility Toolkit (GLUT) and pyCUDA compiled with GL support\n"
        + "Controls: c -> toggles cpu path (CUDA->cpu->OpenGL)\n"
        + "          g -> toggles gpu path (CUDA->OpenGL)\n"
        + "          q -> exit demo."
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
        "-v", "--verbose", default=False, action="store_true", help="Verbose"
    )

    args = parser.parse_args()

    app = OpenGLApplication(
        args.gpu_id,
        args.encoded_file_path.as_posix(),
    )
    app.run(verbose=args.verbose)

    exit(0)
