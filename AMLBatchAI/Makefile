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
RESOURCE_GROUP:="msazmlrg"
WORKSPACE:="msksalt"

## TODO: retag image and push to acr
## Need to also log in to acr
## https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli

### ONLY RUN THESE COMMANDS INSIDE THE DOCKER CONTAINER ###
select-subscription:
	az login -o table
	az account set --subscription $(SELECTED_SUBSCRIPTION)

create-workspace:
	python src/workspace.py create $(WORKSPACE) $(RESOURCE_GROUP) --subscription_id $(SELECTED_SUBSCRIPTION)

show-workspaces:
	az ml workspace list -g $(RESOURCE_GROUP)
	
	
.PHONY: help build push