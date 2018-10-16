import papermill as pm
from config import generate


def _execute(input_nb, output_nb, generate_config=True):
    if generate_config:
        generate()
    pm.execute_notebook(input_nb, output_nb, log_output=True, progress_bar=False)
    # Save config to model directory


if __name__ == "__main__":
    import fire

    fire.Fire({"execute": _execute})
