import logging
import os
import shutil

import papermill as pm

from config import default_config_path, generate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _execute(input_nb, output_nb, config_path=None, generate_config=True, **kwargs):
    logger.info(f"Executing notebook {input_nb} and saving to {output_nb}")

    if config_path is None:
        config_path = default_config_path()
    logger.info(f"Loading config from {config_path}")

    if generate_config:
        logger.info(f"Generating config")
        generate(config=config_path, config_template=config_path)

    params_dict = kwargs
    params_dict["config_path"] = config_path

    pm.execute_notebook(
        input_nb, output_nb, log_output=True, progress_bar=False, parameters=params_dict
    )

    logger.info(f"Saving config")
    shutil.copy(
        config_path, os.path.join(os.getenv("MODELS", "output"), "model_config.json")
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"execute": _execute})
