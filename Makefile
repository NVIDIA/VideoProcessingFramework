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
	wget http://www.scikit-video.org/stable/_static/bikes.mp4

	python ./samples/SampleDecode.py -g 0 -e ./bikes.mp4 -r ./tests/test.raw       
	python ./samples/SampleDecodeSw.py -e ./bikes.mp4 -r ./tests/test.raw
	python ./samples/SampleEncodeMultiThread.py 0 848 464 ./tests/test.raw 10
	python ./samples/SampleMeasureVideoQuality.py -g 0 -i ./tests/test.raw -o ./tests/test.raw -w 848 -h 464   
	python ./samples/SamplePyTorch.py 0 ./bikes.mp4 ./tests/out.mp4
	python ./samples/SampleTensorRTResnet.py 0 ./bikes.mp4
	python ./samples/SampleTorchResnet.py  0 ./bikes.mp4     
	python ./samples/SampleDecodeMultiThread.py  0 ./bikes.mp4 10 
	python ./samples/SampleDemuxDecode.py 0 ./tests/test_res_change.h264 ./tests/test.raw
	python ./samples/SampleEncode.py 0 ./tests/test.raw ./tests/output.mp4 848 464              
ifndef DISPLAY 
		echo "skipping rendering samples"
else
		python ./samples/SampleTorchSegmentation.py 0 ./bikes.mp4
		python ./samples/SampleOpenGL.py -g 0 -e ./bikes.mp4        
endif
	# python ./samples/SampleRemap.py 0 ./bikes.mp4 ./tests/remap.npz
	# python ./samples/SampleDecodeRTSP.py 0 rtsp://localhost:8554/mystream rtsp://localhost:8554/mystream # no rtsp stream available for testing

generate-stubs:
	pip3 install mypy
	stubgen -mPyNvCodec._PyNvCodec
	cp out/PyNvCodec/_PyNvCodec.pyi src/PyNvCodec/__init__.pyi
	
.PHONY: run_tests vpf-gpu generate-stubs

# vim:ft=make
#
