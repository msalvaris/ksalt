define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  build docker image
    make push					 push container
    make run					 run benchmarking container
endef
export PROJECT_HELP_MSG
PWD:=$(shell pwd)

image_name:=masalvar/ksalt
DATA_DIR:=/mnt/ksalt
setup_volumes:=-v $(PWD):/workspace  \
	-v $(DATA_DIR):/mnt/data 

setup_environment:=--env SCRIPTS='/workspae' \
	--env DATA='/mnt/input' 

help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(image_name) Docker

run:
	nvidia-docker run $(setup_volumes) $(setup_environment) -it $(image_name)
	
notebook:
	nvidia-docker run $(setup_volumes) $(setup_environment) -p 9999:9999 -it $(image_name) bash -c "jupyter notebook --ip=* --port=9999 --no-browser --allow-root"

push:
	docker push $(image_name)



.PHONY: help build push