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

setup_environment:=--env DATA='/mnt/data' \
	--env MODELS='/mnt/models' \
	--env PYTHONPATH=$PYTHONPATH:/workspace/experiment/src \
	--env KAGGLE_CONFIG_DIR=/kaggle \
	--env TBOARD_LOGS=/mnt/models/logs


help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(image_name) Docker

run:
	nvidia-docker run $(setup_volumes) $(setup_environment) -p 9999:9999 -p 6006:6006 -it $(image_name)
	
notebook:
	nvidia-docker run $(setup_volumes) $(setup_environment) -p 9999:9999 -p 6006:6006 -it $(image_name) bash -c "jupyter notebook"
	
tensorboard:
	nvidia-docker run $(setup_volumes) $(setup_environment) -p 6006:6006 -it $(image_name) bash -c "tensorboard $(FLAGS)"

push:
	docker push $(image_name)

### ONLY RUN THESE COMMANDS INSIDE THE DOCKER CONTAINER ###

$(DATA)/train.zip:
	kaggle competitions download -c tgs-salt-identification-challenge --path $(DATA)
	
$(DATA)/test.zip:
	kaggle competitions download -c tgs-salt-identification-challenge --path $(DATA)
	
$(DATA)/train: $(DATA)/train.zip
	cd $(DATA)
	mkdir -p train && unzip train.zip -d train
	
$(DATA)/test: $(DATA)/test.zip
	cd $(DATA)
	mkdir -p train && unzip train.zip -d train

download-data: $DATA/train $(DATA)/test $DATA/train.csv $(DATA)/depths.csv
	@echo Data dowloaded

run-model: $(DATA)/train $(DATA)/test
	python experiment/src/nb.py execute experiment/notebooks/Model.ipynb experiment/notebooks/Model.ipynb
#	papermill experiment/notebooks/Model.ipynb notebooks/Model.ipynb --log-output $(FLAGS)

convert-jupytext:	
	jupytext --to notebook $(FLAGS)

submit:
	python src/submission.py $(FLAGS)
	
show-submissions:
	kaggle competitions submissions tgs-salt-identification-challenge

clean-output: clean-model clean-submission
	@echo cleaned model and submission output

clean-model:
	$(eval branch_name:=$(shell git branch | grep \* | cut -d ' ' -f2))
	rm $(MODELS)/$(branch_name)/*.model

clean-submission:
	$(eval branch_name:=$(shell git branch | grep \* | cut -d ' ' -f2))
	rm $(MODELS)/$(branch_name)/*.csv
	
update-description:
	python src/git_tools.py description set-from-file description
	
list-branches:
	python src/git_tools.py list

.PHONY: help build push