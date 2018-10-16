import logging
import os
import shutil
import uuid
from pprint import pformat

import numpy as np
import torch
from git import Repo, InvalidGitRepositoryError

logger = logging.getLogger(__name__)


def _generate_id():
    return str(uuid.uuid4())


def _model_identifier():
    try:
        repo = Repo(search_parent_directories=True)
        return repo.active_branch.name
    except InvalidGitRepositoryError:
        return os.getenv("MODEL_IDENTIFIER", _generate_id())


def model_path():
    model_path = os.path.join(os.getenv("MODELS"), _model_identifier())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


def save_checkpoint(
    state,
    outdir=model_path(),
    model_filename="model_state.pth",
    best_model_filename="model_best_state.pth",
):
    model_path = os.path.join(outdir, model_filename)
    best_model_path = os.path.join(outdir, best_model_filename)
    logger.debug(f"Saving to {model_path}")
    torch.save(state, model_path)
    if state["best_epoch"] == state["epoch"]:
        logger.debug(pformat(state, indent=2))
        logger.debug(f"Saving to {best_model_path}")
        shutil.copy(model_path, best_model_path)


def update_state(state, epoch, eval_metric, metric_value, model, optimizer):
    state["state_dict"] = model.state_dict()
    state["optimizer"] = optimizer.state_dict()
    state["epoch"] = epoch
    state[eval_metric] = metric_value
    best_key = f"best_{eval_metric}"

    # update
    if metric_value > state[best_key]:
        logger.info(f"{eval_metric} went from {state[best_key]} to {metric_value} >:)")
        state[best_key] = metric_value
        state["best_epoch"] = epoch

    return state


def predict_tta(model, image):  # predict both orginal and reflect x
    with torch.no_grad():
        image_reflect = np.flip(image.numpy(), axis=3).copy()
        with torch.cuda.device(0):
            image_gpu = image.type(torch.float).cuda()
            image_reflect_gpu = torch.as_tensor(image_reflect).type(torch.float).cuda()

        outputs = model(image_gpu)
        outputs_reflect = model(image_reflect_gpu)
        return (
            outputs.cpu().numpy() + np.flip(outputs_reflect.cpu().numpy(), axis=3)
        ) / 2
