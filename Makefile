define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  build docker image
    make push					 push container
    make run					 run benchmarking container
endef
export PROJECT_HELP_MSG
PWD:=$(shell pwd)

BRANCH:=$(shell git branch | grep \* | cut -d ' ' -f2)
control_image_name:=masalvar/ksalt
execution_image_name:=masalvar/ksalt-$(BRANCH):execution

DATA_DIR:=/mnt/ksalt
MODEL_DIR:=/mnt/models/ksalt
KAGGLE:=/home/mat/.kaggle
FLAGS:=

local_code_volume:=-v $(PWD):/workspace 

setup_volumes:=-v $(DATA_DIR):/mnt/data \
	-v $(MODEL_DIR):/mnt/models \
	-v $(KAGGLE):/kaggle

setup_environment:=--env DATA='/mnt/data' \
	--env MODELS='/mnt/models' \
	--env PYTHONPATH=$PYTHONPATH:/workspace/experiment/src \
	--env KAGGLE_CONFIG_DIR=/kaggle \
	--env TBOARD_LOGS=/mnt/models/logs \
	--env MODEL_CONFIG=experiment/configs/config.json


help:
	echo "$$PROJECT_HELP_MSG" | less

build: build-control build-execute
	@echo "Built control and execution images"

build-control:
	docker build --target control -t $(control_image_name) -f Docker/dockerfile . 

build-execute:
	docker build --target execution -t $(execution_image_name)  -f Docker/dockerfile . 

bash:
	nvidia-docker run $(local_code_volume) $(setup_volumes) $(setup_environment) -p 9999:9999 -p 6006:6006 -it $(control_image_name)
	
notebook:
	nvidia-docker run $(local_code_volume) $(setup_volumes) $(setup_environment) -p 9999:9999 -p 6006:6006 -it $(control_image_name) bash -c "jupyter notebook"
	
tensorboard:
	nvidia-docker run $(local_code_volume) $(setup_volumes) $(setup_environment) -p 6006:6006 -it $(control_image_name) bash -c "tensorboard $(FLAGS)"

run-dev:
	nvidia-docker run $(local_code_volume) $(setup_volumes) $(setup_environment) -it $(execution_image_name)

run:
	nvidia-docker run $(setup_volumes) $(setup_environment) -it $(execution_image_name)
	
push-control:
	docker push $(control_image_name)


.PHONY: help build push