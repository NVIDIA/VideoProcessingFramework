FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=compute,video

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get -y install \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev\
    wget \
    build-essential \
    ninja-build \
    cmake \
    git \
    python3 \
    python3-pip \
    python-is-python3

ARG PIP_INSTALL_EXTRAS=""
# Build vpf
RUN mkdir /src
WORKDIR /src
COPY src src
COPY setup.py setup.py
COPY pyproject.toml pyproject.toml
COPY CMakeLists.txt CMakeLists.txt
RUN python3 -m pip install --no-cache-dir setuptools wheel
RUN python3 -m pip install --no-cache-dir .[$PIP_INSTALL_EXTRAS] 

RUN mkdir /cwd
WORKDIR /cwd

ENTRYPOINT ["/bin/bash"]
