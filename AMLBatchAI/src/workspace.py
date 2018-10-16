import logging

from azureml.core import Workspace

logger = logging.getLogger(__name__)


def create_workspace(
    workspace_name,
    resource_group,
    subscription_id=None,
    workspace_region="eastus2",
    config_path="configs",
    filename="azml_config.json",
):
    
    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        location=workspace_region,
        create_resource_group=True,
        exist_ok=True,
    )
    logger.info(ws.get_details())
    ws.write_config(path=config_path, file_name=filename)
        
    
if __name__ == "__main__":
    import fire
    fire.Fire({
        'create': create_workspace,
    })
