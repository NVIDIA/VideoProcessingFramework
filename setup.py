import sys

from pkg_resources import VersionConflict, require
from setuptools import setup

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    import skbuild
    skbuild.setup(
        name="PyNvCodec",
        version="0.1.0",
        description="TODO",
        author='NVIDIA',
        license="MIT",
        packages=['PyNvCodec'],
        package_dir={'': 'src'},
        cmake_install_dir='src/PyNvCodec')
