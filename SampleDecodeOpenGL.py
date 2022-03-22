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

if os.name == 'nt':
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file = sys.stderr)
        print("Can't set CUDA DLLs search path.", file = sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file = sys.stderr)
        exit(1)

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import PyNvCodec as nvc
import pycuda
import numpy as np
import time
import logging
logging.getLogger().setLevel(logging.INFO)

class FPSLogger:
    def __init__(self,interval):
        self.interval=interval
        self.framecount=0
        self.seconds=time.time_ns()/1e9
    
    def log(self,titlebar=True,fmt="fps : {0}"):
        self.framecount+=1
        if self.seconds+self.interval<time.time_ns()/1e9:
            self.fps=self.framecount/self.interval
            self.framecount=0
            self.seconds=time.time_ns()/1e9
            if titlebar:
                glutSetWindowTitle(fmt.format(self.fps))
            else:
                logging.info(fmt.format(self.fps))

class GLPlayer:
    '''
    adapted from https://stackoverflow.com/questions/34348669/mapping-a-texture-onto-a-quad-with-opengl-4-python-and-vertex-shaders
    '''
    def __init__(self,gpu_id, enc_filePath):
        self.gpu_id=gpu_id
        self.enc_filePath=enc_filePath
        self.cpu=False
        self.vao = 0
        self.framecount=0

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
        uniform sampler2D tex;
        in vec2 uv;
        out vec4 color;
        void main(void)
        {
            color = vec4(texture(tex, uv));
        }
        """
        
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)

        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return program
        
    def initdecoder(self):
        self.nvDec = nvc.PyNvDecoder(self.enc_filePath, self.gpu_id)
        self.width, self.height = self.nvDec.Width(), self.nvDec.Height()
        self.cc_ctx = nvc.ColorspaceConversionContext(self.nvDec.ColorSpace(), self.nvDec.ColorRange())
        if self.nvDec.ColorSpace() != nvc.ColorSpace.BT_709:
            self.nvYuv = nvc.PySurfaceConverter(self.width, self.height, self.nvDec.Format(), nvc.PixelFormat.YUV420, 0)
            self.nvCvt = nvc.PySurfaceConverter(self.width, self.height, self.nvYuv.Format(), nvc.PixelFormat.RGB, 0)
        else:
            self.nvYuv=None
            self.nvCvt = nvc.PySurfaceConverter(self.width, self.height, self.nvDec.Format(), nvc.PixelFormat.RGB, 0)
        
        # create downloader and buffer dor CUDA ->CPU ->GL pipeline
        self.nvDwn = nvc.PySurfaceDownloader(self.width, self.height, self.nvCvt.Format(), 0)
        self.data = np.zeros((self.width*self.height,3),np.uint8)
        
    def initgl(self):
        self.rendering_program = self.compile_shaders()
        
        # create 2d texture with same size as video
        self.texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture  ( GL_TEXTURE_2D, self.texture )
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGB, self.width, self.height, 0,  GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glTexParameterf   ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
        glTexParameterf   ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
        glTexParameteri   ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR )
        glTexParameteri   ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR )
        glBindTexture     ( GL_TEXTURE_2D, 0 )
        
        # create GL buffer for CUDA -> GL pipeline
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)
        glBufferData(GL_ARRAY_BUFFER, np.zeros(self.width*self.height*3,np.uint8), GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        import pycuda.autoinit
        import pycuda.gl.autoinit
        self.cuda_pbo = pycuda.gl.RegisteredBuffer(int(self.pbo))
        
        self.vao = 0
        glGenVertexArrays(1, self.vao)
        glBindVertexArray(self.vao)

    def render(self):
        glClearBufferfv(GL_COLOR, 0, (0,0,0))
        # bind program
        glUseProgram(self.rendering_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # get one frame and ronvert to rgb
        rawSurface = self.nvDec.DecodeSingleSurface()
        if self.nvYuv!=None:
            yuvSurface = self.nvYuv.Execute(rawSurface, self.cc_ctx)
            cvtSurface = self.nvCvt.Execute(yuvSurface, self.cc_ctx)
        else:
            cvtSurface = self.nvCvt.Execute(rawSurface, self.cc_ctx)
        
        if self.cpu:
            self.fpslogger.log(True,"CPU FPS : {0}")
            ## Download surface data to CPU, then update GL texture with these data
            success = self.nvDwn.DownloadSingleSurface(cvtSurface, self.data)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,self.width, self.height,GL_RGB, GL_UNSIGNED_BYTE, self.data)
        else:
            self.fpslogger.log(True,"GPU FPS : {0}")
            ## cuda copy from surface.Plane_Ptr() to GL Buffer, then update texture from GL Buffer
            src_plane=cvtSurface.PlanePtr()
            buffer_mapping = self.cuda_pbo.map()
            buffptr,buffsize=buffer_mapping.device_ptr_and_size()
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(src_plane.GpuMem())
            cpy.set_dst_device(buffptr)
            cpy.width_in_bytes = src_plane.Width()
            cpy.src_pitch = src_plane.Pitch()
            cpy.dst_pitch = self.width*3
            cpy.height = src_plane.Height()
            cpy(aligned=False)
            #pycuda.driver.Context.synchronize() ## not required?
            buffer_mapping.unmap()
            ## opengl update texture from pbo
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(self.pbo))
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,self.width, self.height,GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # send uniforms to program and draw quad
        glUniform1i(glGetUniformLocation(self.rendering_program, b'tex'),0)
        glDrawArrays(GL_QUADS, 0, 4)
        # Display
        glutSwapBuffers()
    
    def keypressed(self,key,x,y):
        if key.decode("utf-8") == 'q':
            glutLeaveMainLoop()
        elif key.decode("utf-8")  =='c':
            self.cpu=True
        elif key.decode("utf-8")  =='g':
            self.cpu=False
            
    def animate(self):
        glutPostRedisplay()

    def run(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        self.initdecoder()
        glutInitWindowSize(self.width//4, self.height//4)
        glutInitWindowPosition(0, 0)
        glutCreateWindow(b"nvDec to PyOpenGL example")
        self.initgl()
        self.fpslogger=FPSLogger(1)
        glutIdleFunc(self.animate)
        glutDisplayFunc(self.render)
        glutKeyboardFunc(self.keypressed)
        glutMainLoop()

if __name__ == "__main__":

    print("This sample decodes input video to OpenGL Texture.")
    print("Requires the GL Utility Toolkit (GLUT) and pyCUDA compiled with GL support")
    print("Usage: SampleDecodeOpenGL.py $gpu_id $input_file.")
    print("Controls: c -> toggles cpu path (CUDA->cpu->OpenGL).")
    print("          g -> toggles gpu path (CUDA->OpenGL).")
    print("          q -> exit demo.")

    
    if(len(sys.argv) < 3):
        print("Provide gpu ID and path to input file")
        exit(1)

    gpu_id = int(sys.argv[1])
    enc_filePath = sys.argv[2]
    
    player = GLPlayer(gpu_id, enc_filePath)
    player.run()
    
    exit(0)
