import json
import logging
import os
import uuid

import fire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_DEFAULT_CONFIG = os.getenv("MODEL_CONFIG", os.path.join("../configs", "config.json"))


def _generate_id():
    return str(uuid.uuid4())


def _alter_id(id, config_json):
    if id is None:
        id = _generate_id()
    for key, step in config_json.items():
        step["id"] = id
    return config_json


def load_config(config=_DEFAULT_CONFIG):
    logger.info(f"Loading config {config}")
    with open(config) as f:
        return json.load(f)


def save_config(config_json, config_path=_DEFAULT_CONFIG):
    logger.info(f"Saving config {config_path}")
    with open(config_path, "w") as f:
        json_string = json.dumps(config_json, sort_keys=True, indent=4)
        logger.info(json_string)
        f.write(json_string)


def generate(id=None, config=_DEFAULT_CONFIG, config_template=_DEFAULT_CONFIG):
    logger.info("Generating config")
    config_json = load_config(config_template)
    config_json = _alter_id(id, config_json)
    save_config(config_json, config_path=config)


def default_config_path():
    return _DEFAULT_CONFIG


if __name__ == "__main__":
    fire.Fire(generate)
