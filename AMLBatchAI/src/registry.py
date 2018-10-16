import fire
from azure.common.client_factory import get_client_from_cli_profile
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from azureml.core.runconfig import AzureContainerRegistry


def address_for(registry_client, resource_group, registry_name):
    mreg = registry_client.registries.get(resource_group, registry_name)
    return mreg.login_server


def username_password_for(
    registry_client, resource_group, registry_name, password_index=0
):
    creds = registry_client.registries.list_credentials(resource_group, registry_name)
    return creds.username, creds.passwords[password_index].value


def properties(resource_group, registry_name, subscription_id=None):
    if subscription_id is None:
        registry_client = get_client_from_cli_profile(ContainerRegistryManagementClient)
    else:
        registry_client = get_client_from_cli_profile(
            ContainerRegistryManagementClient, subscription_id=subscription_id
        )
        
    username, password = username_password_for(
        registry_client, resource_group, registry_name
    )
    
    properties_dict = {
        "address": address_for(
            registry_client, resource_group, registry_name
        ),
        "username": username,
        "password": password,
    }
    return properties_dict


def _print_properties(resource_group, registry_name, subscription_id=None):
    print(properties(resource_group, registry_name, subscription_id=subscription_id))


def azure_container_registry_for(resource_group, registry_name, subscription_id=None):
    properties_dict = properties(resource_group, registry_name, subscription_id=subscription_id)
    azr = AzureContainerRegistry()
    azr.address = properties_dict['address']
    azr.username = properties_dict['username']
    azr.password = properties_dict['password']
    return azr


if __name__ == "__main__":
    fire.Fire(_print_properties)
