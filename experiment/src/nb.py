import papermill as pm


def _execute(input_nb, output_nb):
    pm.execute_notebook(input_nb, output_nb, log_output=True, progress_bar=False)


if __name__ == "__main__":
    import fire

    fire.Fire({"execute": _execute})
