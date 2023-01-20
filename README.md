# VideoProcessingFramework

VPF stands for Video Processing Framework. It’s set of C++ libraries and Python bindings which provides full HW acceleration for video processing tasks such as decoding, encoding, transcoding and GPU-accelerated color space and pixel format conversions.

VPF also supports exporting GPU memory objects such as decoded video frames to PyTorch tensors without Host to Device copies. Check the [Wiki page](https://github.com/NVIDIA/VideoProcessingFramework/wiki/Building-from-source) on how to build from source.

## Installation

VPF works on Windows and Linux. The requirements are as follows

- CUDA Toolkit (npp)
- [FFMPEG](https://github.com/FFmpeg/FFmpeg/) (with libavfilter>=7.110.100)
- `cmake` (>=3.21)
- C++ compiler

### Linux

We recommend Ubuntu 22.04 as it comes with a recent enough ffmpeg system packages.
```bash
# Install dependencies (replace XXX in libnvidia-encode-XXX, libnvidia-decode-XXX with the your driver version)
apt install -y \
          libavfilter-dev \
          libavformat-dev \
          libavcodec-dev \
          libswresample-dev \
          libavutil-dev\
          wget \
          cmake \
          build-essential \
          libnvidia-encode-XXX \
          libnvidia-decode-XXX \
          git
# Install CUDA Toolkit (if not already present)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
# Ensure nvcc to your $PATH (most commonly already done by the CUDA installation)
export PATH=/usr/local/cuda/bin:$PATH

# Install VPF
pip3 install git+https://github.com/NVIDIA/VideoProcessingFramework
# or if you cloned this repository
pip3 install .
```

To check whether VPF is correctly installed run the following Python script
```python
import PyNvCodec
```
Please note that some examples have additional dependencies https://github.com/NVIDIA/VideoProcessingFramework/blob/73a14683a17c8f1c7fa6dd73952f8813bd34a11f/setup.py#L26-L31
that need to be installed via pip.
After resolving those you should be able to run `make run_samples_without_docker` using your local pip installation.

### Windows

- Install a C++ toolchain either via Visual Studio or Tools for Visual Studio (https://visualstudio.microsoft.com/downloads/)
- Install CMake (https://cmake.org/) or `pip install cmake`
- Install the CUDA Toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64
- Download or compile [FFMPEG](https://github.com/FFmpeg/FFmpeg/). Binary packages are available at (https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip)
- Install from the root directory of this repository indicating the location of the compiled FFMPEG in a Powershell console
```
# Indicate path to your FFMPEG installation (with subfolders `bin` with DLLs, `include`, `lib`)
$env:SKBUILD_CONFIGURE_OPTIONS="-DTC_FFMPEG_ROOT=C:/path/to/your/ffmpeg/installation/ffmpeg/" 
# Add CUDA DLLs temporarly to PATH enviroment (we recommend to make this change permanent if not already set by CUDA installation)
$env:PATH +=";$env:CUDA_PATH\bin"
pip install .
```
To check whether VPF is correctly installed run the following Python script
```python
import PyNvCodec
```
Please note that some examples have additional dependencies https://github.com/NVIDIA/VideoProcessingFramework/blob/73a14683a17c8f1c7fa6dd73952f8813bd34a11f/setup.py#L26-L31
that need to be installed via pip.

## Docker

For convenience, we provide a Docker images located at `docker` that you can use to easily install all dependencies for
the samples ([docker](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
are required)


```bash
DOCKER_BUILDKIT=1 docker build \
                --tag vpf-gpu-all \
                -f docker/Dockerfile.gpu \
                --build-arg GEN_PYTORCH_EXT=1 \
                --build-arg GEN_OPENGL_EXT=1 \
                .
docker run -it --rm --gpus=all vpf-gpu-all
```

## Documentation

A documentation for Video Processing Framework can be generated from this repository:
```bash
pip install . # install Video Processing Framework
pip install src/PytorchNvCodec/  # install Torch extension if needed (optional), requires "torch" to be installed before
pip install sphinx  # install documentation tool sphinx
cd docs
make html
```
You can then open `_build/html/index.html` with your browser.

## Community Support
If you did not find the information you need or if you have further questions or problems, you are very welcome to join the developer community at [NVIDIA](https://forums.developer.nvidia.com/categories). We have dedicated categories covering diverse topics related to [video processing and codecs](https://forums.developer.nvidia.com/c/gaming-and-visualization-technologies/visualization/video-processing-optical-flow/189).

The forums are also a place where we would be happy to hear about how you made use of VPF in your project.
