define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  build docker image
    make push					 push container
    make run					 run benchmarking container
endef
export PROJECT_HELP_MSG
PWD:=$(shell pwd)
SELECTED_SUBSCRIPTION:="Boston Team Danielle"
RESOURCE_GROUP:="msazmlksaltrg"
WORKSPACE:="msksalt"

## TODO: retag image and push to acr
## Need to also log in to acr
## https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli

### ONLY RUN THESE COMMANDS INSIDE THE DOCKER CONTAINER ###
select-subscription:
	az login -o table
	az account set --subscription $(SELECTED_SUBSCRIPTION)

create-workspace:
	python src/workspace.py create $(WORKSPACE) $(RESOURCE_GROUP) --subscription_id edf507a2-6235-46c5-b560-fd463ba2e771 --workspace_region eastus

show-workspaces:
	az ml workspace list -g $(RESOURCE_GROUP)
	
#az acr login --name msksaltacrahtgbdgx
#docker login myregistry.azurecr.io -u xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx -p myPassword	
#sudo az acr login --name msksaltacrahtgbdgx -g msazmlksaltrgUnable to load extension 'azure-cli-ml'. Use --debug for more information.Login Succeeded
#mat@msvole1:~/repos/_ksalt$ sudo docker tag masalvar/ksalt-pytorch_resnet34_unet_hyper:execution msksaltacrahtgbdgx.azurecr.io/ksalt-pytorch_resnet24_unet_hyper
#mat@msvole1:~/repos/_ksalt$ sudo docker push msksaltacrahtgbdgx.azurecr.io/ksalt-pytorch_resnet24_unet_hyper
	
.PHONY: help build push