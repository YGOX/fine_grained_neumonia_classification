#!/bin/bash
cd "$(dirname "$0")"
cd ..

cur_path="$(pwd)"
echo $cur_path

mkdir -p /tmp/docker

docker_name=lung_env
docker stop $docker_name
docker rm $docker_name

docker build -t $docker_name .
docker run  -dit \
               --gpus all \
               --ipc=host \
               -v $(pwd)/input:/app/input \
               -v $(pwd)/output:/app/output \
               -v $(pwd)/data_info:/app/data_info \
               -v /home/felix/.jupyter:/root/.jupyter \
               -v /home/felix/.cache:/root/.cache \
	            -v $(pwd):/app \
               --name $docker_name  -p 9998:8888 $docker_name
# -v /home/felix/pj/detectron2:/home/appuser/detectron2_repo \