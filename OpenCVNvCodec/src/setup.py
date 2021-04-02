from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.1"


ext_modules = [
    Pybind11Extension(
        "OpenCVNvCodec",
        ["OpenCVNvCodec.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
    ),
]


setup(
    name="OpenCVNvCodec",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
