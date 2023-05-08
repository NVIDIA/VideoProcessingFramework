# VideoProcessingFramework

VPF stands for Video Processing Framework. It’s set of C++ libraries and Python bindings which provides full HW acceleration for video processing tasks such as decoding, encoding, transcoding and GPU-accelerated color space and pixel format conversions.

VPF also supports exporting GPU memory objects such as decoded video frames to PyTorch tensors without Host to Device copies. 

## Prerequisites
VPF works on Linux(Ubuntu 20.04 and Ubuntu 22.04 only) and Windows

- NVIDIA display driver: 525.xx.xx or above
- CUDA Toolkit 11.2 or above 
  - CUDA toolkit has driver bundled with it e.g. CUDA Toolkit 12.0 has driver `530.xx.xx`. During installation of CUDA toolkit you could choose to install or skip installation of the bundled driver. Please choose the appropriate option.
- FFMPEG
  - [Compile FFMPEG with shared libraries](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html) 
  - or download pre-compiled binaries from a source you trust.
    - During VPF’s “pip install”(mentioned in sections below) you need to provide a path to the directory where FFMPEG got installed.
  - or you could install system FFMPEG packages (e.g. ```apt install  libavfilter-dev libavformat-dev libavcodec-dev libswresample-dev libavutil-dev``` on Ubuntu)

- Python 3 and above
- Install a C++ toolchain either via Visual Studio or Tools for Visual Studio.
  - Recommended version is Visual Studio 2017 and above
(Windows only)

### Linux

We recommend Ubuntu 20.04 as it comes with a recent enough FFmpeg system packages.
If you want to build FFmpeg from source, you can follow
https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html
```bash
# Install dependencies
apt install -y \
          libavfilter-dev \
          libavformat-dev \
          libavcodec-dev \
          libswresample-dev \
          libavutil-dev\
          wget \
          build-essential \
          git

# Install CUDA Toolkit (if not already present)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda
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
If using Docker via [Nvidia Container Runtime](https://developer.nvidia.com/nvidia-container-runtime),
please make sure to enable the `video` driver capability: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#driver-capabilities via
the `NVIDIA_DRIVER_CAPABILITIES` environment variable in the container or the `--gpus` command line parameter (e.g.
`docker run -it --rm --gpus 'all,"capabilities=compute,utility,video"' nvidia/cuda:12.1.0-base-ubuntu22.04`).

Please note that some examples have additional dependencies that need to be installed via pip (`pip install .[samples]`). 
Samples using PyTorch will require an optional extension which can be installed via
```bash
pip install src/PytorchNvCodec  # install Torch extension if needed (optional), requires "torch" to be installed before
```

After resolving those you should be able to run `make run_samples_without_docker` using your local pip installation.

### Windows

- Install a C++ toolchain either via Visual Studio or Tools for Visual Studio (https://visualstudio.microsoft.com/downloads/)
- Install the CUDA Toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64
- Compile [FFMPEG](https://github.com/FFmpeg/FFmpeg/) with shared libraries or download pre-compiled binaries from a source you trust
- Install from the root directory of this repository indicating the location of the compiled FFMPEG in a Powershell console
```pwsh
# Indicate path to your FFMPEG installation (with subfolders `bin` with DLLs, `include`, `lib`)
$env:SKBUILD_CONFIGURE_OPTIONS="-DTC_FFMPEG_ROOT=C:/path/to/your/ffmpeg/installation/ffmpeg/" 
pip install .
```
To check whether VPF is correctly installed run the following Python script
```python
import PyNvCodec
```
Please note that some examples have additional dependencies (`pip install .[sampels]`) that need to be installed via pip. 
Samples using PyTorch will require an optional extension which can be installed via

```bash
pip install src/PytorchNvCodec  # install Torch extension if needed (optional), requires "torch" to be installed before
```

## Docker

For convenience, we provide a Docker images located at `docker` that you can use to easily install all dependencies for
the samples ([docker](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
are required)


```bash
DOCKER_BUILDKIT=1 docker build \
                --tag vpf-gpu \
                -f docker/Dockerfile \
                --build-arg PIP_INSTALL_EXTRAS=torch \
                .
docker run -it --rm --gpus=all vpf-gpu
```

`PIP_INSTALL_EXTRAS` can be any subset listed under `project.optional-dependencies` in [pyproject.toml](pyproject.toml).

## Documentation

A documentation for Video Processing Framework can be generated from this repository:
```bash
pip install . # install Video Processing Framework
pip install src/PytorchNvCodec  # install Torch extension if needed (optional), requires "torch" to be installed before
pip install sphinx  # install documentation tool sphinx
cd docs
make html
```
You can then open `_build/html/index.html` with your browser.

## Community Support
If you did not find the information you need or if you have further questions or problems, you are very welcome to join the developer community at [NVIDIA](https://forums.developer.nvidia.com/categories). We have dedicated categories covering diverse topics related to [video processing and codecs](https://forums.developer.nvidia.com/c/gaming-and-visualization-technologies/visualization/video-processing-optical-flow/189).

The forums are also a place where we would be happy to hear about how you made use of VPF in your project.
