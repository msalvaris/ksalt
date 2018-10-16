import os
import uuid

import numpy as np
import torch
from git import Repo, InvalidGitRepositoryError


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def _sgd(model_parameters, optim_config):
    return torch.optim.SGD(
        model_parameters,
        lr=optim_config["base_lr"],
        momentum=optim_config["momentum"],
        weight_decay=optim_config["weight_decay"],
        nesterov=optim_config["nesterov"],
    )


def _adam(model_parameters, optim_config):
    return torch.optim.Adam(
        model_parameters,
        lr=optim_config["base_lr"],
        betas=optim_config["betas"],
        weight_decay=optim_config["weight_decay"],
    )


class OptimizerUndefinedException(Exception):
    pass


def _no_optimizer(model_parameters, optim_config):
    raise OptimizerUndefinedException(
        "Optimizer {} not found. Please check it is defined".format(
            optim_config["type"]
        )
    )


_OPTIMIZER_DICT = {"sgd": _sgd, "adam": _adam}


def _get_optimizer(model_parameters, optim_config):
    return _OPTIMIZER_DICT.get(optim_config["type"], _no_optimizer)(
        model_parameters, optim_config
    )


def _multistep(optimizer, optim_config):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=optim_config["milestones"], gamma=optim_config["lr_decay"]
    )


def _cosine(optimizer, optim_config):
    total_steps = optim_config["epochs"] * optim_config["steps_per_epoch"]

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            optim_config["scheduler"]["lr_min"] / optim_config["base_lr"],
        ),
    )


def _no_scheduler(optimizer, optim_config):
    return None


_SCHEDULERS_DICT = {"multistep": _multistep, "cosine": _cosine}


def _get_scheduler(optimizer, optim_config):
    if optim_config["type"] == "sgd":
        scheduler = _SCHEDULERS_DICT.get(optim_config["scheduler"]["type"])(
            optimizer, optim_config
        )
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def _generate_id():
    return str(uuid.uuid4())


def _log_identifier():
    try:
        repo = Repo(search_parent_directories=True)
        return os.path.join(repo.active_branch.name, repo.active_branch.commit.hexsha)
    except InvalidGitRepositoryError:
        return os.getenv("TBOARD_LOG_IDENTIFIER", _generate_id())


def tboard_log_path():
    log_path = os.path.join(os.getenv("TBOARD_LOGS"), _log_identifier())
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path
