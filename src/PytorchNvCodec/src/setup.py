from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='PytorchNvCodec',
    ext_modules=[CUDAExtension('PytorchNvCodec', ['PytorchNvCodec.cpp'])],
    cmdclass={
        'build_ext': BuildExtension
    })