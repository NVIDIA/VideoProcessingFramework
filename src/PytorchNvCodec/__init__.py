"""

"""

__author__ = "NVIDIA"
__copyright__ = "Copyright 2022, NVIDIA"
__credits__ = []
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "NVIDIA"
__email__ = "TODO"
__status__ = "Production"


try:
    import torch # has to be imported to have libc10 available
    # Import native module
    from _PytorchNvCodec import * 
except ImportError:
    import distutils.sysconfig
    from os.path import join, dirname
    raise RuntimeError("Failed to import native module _PytorchNvCodec! "
                           f"Please check whether \"{join(dirname(__file__), '_PytorchNvCodec' + distutils.sysconfig.get_config_var('EXT_SUFFIX'))}\""  # noqa
                           " exists and can find all library dependencies (CUDA, ffmpeg).\n"
                           "On Unix systems, you can use `ldd` on the file to see whether it can find all dependencies.\n"
                           "On Windows, you can use \"dumpbin /dependents\" in a Visual Studio command prompt or\n"
                           "https://github.com/lucasg/Dependencies/releases.")
