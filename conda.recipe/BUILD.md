# Steps to build conda package 

1. Build and start docker container to build the package 
```sh
cd docker
make build_conda 
make run_conda VIDEO_CODEC_PATH=<path to video codec>
```

2. Once the docker container is started in iteractive mode 
```sh
cd VideoProcessingFramework
conda mambabuild --user <anaconda username> conda.recipe/ 
```