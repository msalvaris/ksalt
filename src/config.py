import json
import os
import uuid
import fire

_DEFAULT_CONFIG = os.path.join("../configs", "config.json")


def _generate_id():
    return str(uuid.uuid4())


def _alter_id(id, config_json):
    if id is None:
        id = _generate_id()
    for key, step in config_json.items():
        step["id"] = id
    return config_json


def load_config(config=_DEFAULT_CONFIG):
    with open(config) as f:
        return json.load(f)


def _save_config(config_json, config_path):
    with open(config_path, "w") as f:
        json.dump(config_json, f)


def main(id=None, config=_DEFAULT_CONFIG):
    config_json = load_config(config)
    config_json = _alter_id(id, config_json)

    _save_config(config_json, config)


if __name__ == "__main__":
    fire.Fire(main())
