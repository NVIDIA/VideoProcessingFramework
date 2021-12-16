SHELL:=/bin/sh
SHELLOPTS:=pipefail:errexit

UNAME=$(shell uname -s)

all: vpf_built

vpf_built:
	mkdir -p build && cd build && \
  cmake -D VIDEO_CODEC_SDK_INCLUDE_DIR=$(VIDEO_CODEC_SDK)/Interface \
	  -D NVCUVID_LIBRARY=$(VIDEO_CODEC_SDK)/Lib/linux/stubs/x86_64/libnvcuvid.so \
	  -D NVENCODE_LIBRARY=$(VIDEO_CODEC_SDK)/Lib/linux/stubs/x86_64/libnvidia-encode.so \
	  -D CMAKE_INSTALL_PREFIX=../dist \
	  -D GENERATE_PYTORCH_EXTENSION:BOOL="$(GEN_PYTORCH_EXT)" \
	  -D PYTHON_BINARY=$(PYTHON_BINARY) \
	  -D GENERATE_PYTHON_BINDINGS:BOOL="1" .. && \
  make -j$(nproc) && make install

linter_python: $(shell find . -type f -name "*.py" -not -path "*venv*")
	black $?

build_env:
	poetry install --extras $(EXTRAS)

activate_env:
	. `poetry env info --path`/bin/activate

image:
	cp -a $(VIDEO_CODEC_SDK) Video_Codec_SDK
	docker-compose -f docker/docker-compose.yml build vpf

run:
	docker-compose -f docker/docker-compose.yml run vpf

clean:
	rm -rf *_built
	rm -rf *.egg-info
