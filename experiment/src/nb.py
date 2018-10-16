import os
import shutil

import papermill as pm
from config import default_config_path, generate


def _execute(input_nb, output_nb, generate_config=True):
    if generate_config:
        generate()
    pm.execute_notebook(input_nb, output_nb, log_output=True, progress_bar=False)
    shutil.copy(
        default_config_path(),
        os.path.join(os.getenv("MODELS", "output"), "model_config.json"),
    )


if __name__ == "__main__":
    import fire

    fire.Fire({"execute": _execute})
