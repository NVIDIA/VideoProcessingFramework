"""

"""

import sys
import os

from pkg_resources import VersionConflict, require

try:
    require("setuptools>=42")
except VersionConflict:
    print("Error: version of setuptools is too old (<42)!")
    sys.exit(1)


if __name__ == "__main__":
    import skbuild

    PytorchNvCodec = "PytorchNvCodec @ git+https://github.com/NVIDIA/VideoProcessingFramework.git#subdirectory=src/PytorchNvCodec/"
    skbuild.setup(
        name="PyNvCodec",
        version="2.0",
        description="Video Processing Library with full NVENC/NVDEC hardware acceleration",
        author="NVIDIA",
        license="Apache 2.0",
        install_requires=["numpy"],
        extras_require={
            # , "PyOpenGL-accelerate" # does not compile on 3.10
            "dev": ["pycuda", "pyopengl", "torch", "torchvision", "opencv-python", "onnx", "tensorrt", f"PytorchNvCodec @ file://{os.getcwd()}/src/PytorchNvCodec/"],
            "samples": ["pycuda", "pyopengl", "torch", "torchvision", "opencv-python", "onnx", "tensorrt", "tqdm", PytorchNvCodec],
            "tests": ["pycuda", "pyopengl", "torch", "torchvision", "opencv-python", PytorchNvCodec],
            "torch": ["torch", "torchvision", PytorchNvCodec],
            "tensorrt": ["torch", "torchvision", PytorchNvCodec],
        },
        dependency_links=[
            "https://pypi.ngc.nvidia.com"
        ],
        packages=["PyNvCodec"],
        package_data={"PyNvCodec": ["__init__.pyi"]},
        package_dir={"": "src"},
        cmake_install_dir="src",
    )
