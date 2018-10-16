import logging

from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

logger = logging.getLogger(__name__)


def cluster(
    worskspace,
    name,
    vm_size="STANDARD_NC6",
    autoscale_enabled=True,
    vm_priority="dedicated",
    cluster_min_nodes=0,
    cluster_max_nodes=4,
):
    # choose a name for your cluster

    try:
        compute_target = ComputeTarget(workspace=worskspace, name=name)
        logger.info("Found existing compute target.")
    except ComputeTargetException:
        logger.info("Creating a new compute target...")
        compute_config = BatchAiCompute.provisioning_configuration(
            vm_size=vm_size,
            autoscale_enabled=autoscale_enabled,
            vm_priority=vm_priority,
            cluster_min_nodes=cluster_min_nodes,
            cluster_max_nodes=cluster_max_nodes,
        )

        # create the cluster
        compute_target = ComputeTarget.create(worskspace, name, compute_config)
        compute_target.wait_for_completion(show_output=True)

        # Use the 'status' property to get a detailed status for the current cluster.
        logger.info(compute_target.status.serialize())

    return compute_target
