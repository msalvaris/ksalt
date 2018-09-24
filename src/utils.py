import os

import numpy as np
import torch
from git import Repo


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def _get_optimizer(model_parameters, optim_config):
    if optim_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=optim_config['base_lr'],
            momentum=optim_config['momentum'],
            weight_decay=optim_config['weight_decay'],
            nesterov=optim_config['nesterov'])
    elif optim_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=optim_config['base_lr'],
            betas=optim_config['betas'],
            weight_decay=optim_config['weight_decay'])
    return optimizer


def _get_scheduler(optimizer, optim_config):
    if optim_config['optimizer'] == 'sgd':
        if optim_config['scheduler'] == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=optim_config['milestones'],
                gamma=optim_config['lr_decay'])
        elif optim_config['scheduler'] == 'cosine':
            total_steps = optim_config['epochs'] * \
                          optim_config['steps_per_epoch']

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    total_steps,
                    1,  # since lr_lambda computes multiplicative factor
                    optim_config['lr_min'] / optim_config['base_lr']))
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def _ver():
    repo = Repo(search_parent_directories=True)
    # repo.active_branch.commit.hexsha
    return repo.active_branch.name


def tboard_log_path():
    log_path = os.path.join(os.getenv('TBOARD_LOGS'), _ver())
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path
