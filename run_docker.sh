#!/bin/bash
docker run --gpus all -it\
    -e CUDA_VISIBLE_DEVICES=$3\
    -v $PWD:/face_recognition \
    --memory=64g --memory-swap=64g --cpuset-cpus=0-39 --shm-size=32g \
    --name $1 $2 bash
