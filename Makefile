run_tests:
	python -m unittest discover tests

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
	
.PHONY: run_tests generate-stubs

# vim:ft=make
#
