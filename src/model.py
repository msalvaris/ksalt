from git import Repo
import os
import torch
import logging
import shutil
from pprint import pformat

logger = logging.getLogger(__name__)

def _ver():
    repo = Repo(search_parent_directories=True)
    # repo.active_branch.commit.hexsha
    return repo.active_branch.name


def model_path():
    model_path = os.path.join(os.getenv('MODELS'), _ver())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


def save_checkpoint(state, outdir=model_path()):
    model_path = os.path.join(outdir, 'model_state.pth')
    best_model_path = os.path.join(outdir, 'model_best_state.pth')
    logger.debug(f"Saving to {model_path}")
    torch.save(state, model_path)
    if state['best_epoch'] == state['epoch']:
        logger.debug(pformat(state, indent=2))
        logger.debug(f"Saving to {best_model_path}")
        shutil.copy(model_path, best_model_path)
    

def update_state(state, epoch, eval_metric, metric_value, model, optimizer):
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['epoch'] = epoch
    state[eval_metric] = metric_value
    best_key = f'best_{eval_metric}'
    
    # update 
    if metric_value > state[best_key]:
        logger.info(f'{eval_metric} went from {state[best_key]} to {metric_value} >:)')
        state[best_key] = metric_value
        state['best_epoch'] = epoch

    return state
    