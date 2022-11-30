vpf-gpu:
	DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.gpu --tag vpf-gpu .

vpf-gpu-all:
	DOCKER_BUILDKIT=1 docker build \
					--tag vpf-gpu-all \
					-f docker/Dockerfile.gpu \
					--build-arg GEN_PYTORCH_EXT=1 \
					--build-arg GEN_OPENGL_EXT=1 \
					.

run_tests: vpf-gpu
	docker run --rm -v $(shell pwd):/repo --entrypoint "python3" vpf-gpu  -m unittest discover /repo/tests /repo/tests/test.mp4

run_samples: vpf-gpu-all
	docker run --rm -v $(shell pwd):/repo --entrypoint "python3" vpf-gpu-all -m unittest discover /repo/tests /repo/tests/test.mp4
	docker run --rm -v $(shell pwd):/repo --workdir /repo --entrypoint "make" vpf-gpu-all run_samples_without_docker

run_samples_without_docker:
	wget http://www.scikit-video.org/stable/_static/bikes.mp4 -P /tmp
	python3 ./samples/SampleOpenGL.py --gpu-id 0 --encoded-file-path /tmp/bikes.mp4

	
.PHONY: run_tests vpf-gpu

# vim:ft=make
#
