import logging

from azure.common.client_factory import get_client_from_cli_profile
from azure.common.credentials import get_cli_profile
from azure.mgmt.resource import SubscriptionClient

logger = logging.getLogger(__name__)


def list_subscriptions():
    sub_client = get_client_from_cli_profile(SubscriptionClient)
    for sub in sub_client.subscriptions.list():
        logging.info("{} {}".format(sub.display_name, sub.subscription_id))


def select(sub_name_or_id):
    profile = get_cli_profile()
    profile.set_active_subscription(sub_name_or_id)


if __name__ == "__main__":
    import fire

    fire.Fire({"list": list_subscriptions, "select": select})
