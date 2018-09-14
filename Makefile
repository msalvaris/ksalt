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

help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(image_name) Docker

run:
	nvidia-docker run -v $(PWD):/workspace -it $(image_name)
	
notebook:
	nvidia-docker run -v $(PWD):/workspace -p 9999:9999 -it $(image_name) bash -c "jupyter notebook --ip=* --port=9999 --no-browser --allow-root"

push:
	docker push $(image_name)



.PHONY: help build push