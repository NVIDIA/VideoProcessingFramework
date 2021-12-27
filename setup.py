import os
import re
import sys
import shutil
import logging
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

def collect_cmake_args():
    '''
    Collect additional cmake args from environment
    '''
    cmake_args = os.environ.get('CMAKE_ARGS', None)
    if not cmake_args:
        PATH_TO_FFMPEG = os.environ.get('PATH_TO_FFMPEG', shutil.which('ffmpeg'))
        FFMPEG_LIB_DIR = os.environ.get('FFMPEG_LIB_DIR')
        FFMPEG_INCLUDE_DIR = os.environ.get('FFMPEG_INCLUDE_DIR')
        VIDEO_CODEC_SDK_DIR = os.environ.get('VIDEO_CODEC_SDK_DIR', None)
        assert VIDEO_CODEC_SDK_DIR is not None, f'Provided VIDEO_CODEC_SDK_DIR: {VIDEO_CODEC_SDK_DIR} is not valid'
        CUDA_LIB_DIR = os.environ.get('CUDA_LIB_DIR', '/usr/local/cuda/lib64')
        CUDA_INCLUDE_DIR = os.environ.get('CUDA_INCLUDE_DIR', '/usr/local/cuda/include')
        GENERATE_PYTORCH_EXTENSION = os.environ.get('GENERATE_PYTORCH_EXTENSION', '0')
        GENERATE_PYTHON_BINDINGS = os.environ.get('GENERATE_PYTHON_BINDINGS', '1')
        cmake_args = f'-DFFMPEG_DIR:PATH={PATH_TO_FFMPEG} \
                    -DVIDEO_CODEC_SDK_DIR:PATH={VIDEO_CODEC_SDK_DIR} \
                    -DGENERATE_PYTHON_BINDINGS:BOOL={GENERATE_PYTHON_BINDINGS} \
                    -DGENERATE_PYTORCH_EXTENSION:BOOL={GENERATE_PYTORCH_EXTENSION} \
                    -DCUDA_INCLUDE_DIR:PATH={CUDA_INCLUDE_DIR} \
                    -DCUDA_LIB_DIR:PATH={CUDA_LIB_DIR} \
                    -DFFMPEG_LIB_DIR:PATH={FFMPEG_LIB_DIR} \
                    -DFFMPEG_INCLUDE_DIR:PATH={FFMPEG_INCLUDE_DIR}'

                    
    return cmake_args

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_INSTALL_PREFIX:PATH={extdir}", # ensure other extensions (pytorch, trt) is output to same extdir
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}"  # not used on MSVC, but no harm
        ]

        build_args = []
        env_cmake_args = collect_cmake_args()
        cmake_args += [item for item in env_cmake_args.split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401
                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass
        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

def get_ext_modules():
    ext_modules = [
        CMakeExtension("PyNvCodec")
    ]
    if os.environ.get('GENERATE_PYTORCH_EXTENSION') == '1':
        ext_modules.append(CMakeExtension("PytorchNvCodec"))
    # TODO: add tensorrt ext module if enabled
    return ext_modules

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
# TODO: Test on windows 
setup(
    name="VPF",
    version="1.1.0",
    license="Apache-2.0 License",
    description="Video Processing Framework",
    url="https://github.com/NVIDIA/VideoProcessingFramework",
    long_description=readme(),
    ext_modules=get_ext_modules(),
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.7",
)