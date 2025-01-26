
IMAGE_NAME := "grox"

build:
    docker build . -t {{IMAGE_NAME}}

run:
    docker run -v $PWD:/workspace -it {{IMAGE_NAME}} python main.py

test:
    docker run -v $PWD/workspace -it {{IMAGE_NAME}} /bin/bash

get-data:
    wget http://mattmahoney.net/dc/enwik9.zip
    unzip enwiki.zip
