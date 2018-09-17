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
MODEL_DIR:=/mnt/models/ksalt
KAGGLE:=/home/mat/.kaggle
FLAGS:=

setup_volumes:=-v $(PWD):/workspace  \
	-v $(DATA_DIR):/mnt/data \
	-v $(MODEL_DIR):/mnt/models \
	-v $(KAGGLE):/kaggle

setup_environment:=--env SCRIPTS='/workspace' \
	--env DATA='/mnt/data' \
	--env MODELS='/mnt/models' \
	--env PYTHONPATH=$PYTHONPATH:/workspace/src \
	--env KAGGLE_CONFIG_DIR=/kaggle 


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

### ONLY RUN THESE COMMANDS INSIDE THE DOCKER CONTAINER ###

$DATA/train.zip:
	kaggle competitions download -c tgs-salt-identification-challenge --path $DATA
	
$DATA/test.zip:
	kaggle competitions download -c tgs-salt-identification-challenge --path $DATA
	
$DATA/train: $DATA/train.zip
	cd $DATA
	mkdir -p train && unzip train.zip -d train
	
$DATA/test: $DATA/test.zip
	cd $DATA
	mkdir -p train && unzip train.zip -d train

download-data: $DATA/train $DATA/test $DATA/train.csv $DATA/depths.csv
	@echo Data dowloaded
	
run-model: $DATA/train $DATA/test
	@echo
	
submit:
	python src/submission.py $(FLAGS)
	
show-submissions:
	kaggle competitions submissions tgs-salt-identification-challenge
	

.PHONY: help build push