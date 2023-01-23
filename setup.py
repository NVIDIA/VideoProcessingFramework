"""

"""

import sys

from pkg_resources import VersionConflict, require

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    import skbuild

    skbuild.setup(
        name="PyNvCodec",
        version="0.1.0",
        description="Video Processing Library with full NVENC/NVDEC hardware acceleration",
        author="NVIDIA",
        license="Apache 2.0",
        install_requires=["numpy"],
        extras_require={
            # , "PyOpenGL-accelerate" # does not compile on 3.10
            "tests": ["pycuda", "pyopengl", "torch", "torchvision", "opencv-python"],
            "torch": ["torch", "torchvision", "opencv-python"],
            "tensorrt": ["torch", "torchvision"],
        },
        packages=["PyNvCodec"],
        package_data={"PyNvCodec": ["__init__.pyi"]},
        package_dir={"": "src"},
        cmake_install_dir="src",
    )
