from git import Repo
import os


def _ver():
    repo = Repo()
    # repo.active_branch.commit.hexsha
    return repo.active_branch.name


def model_path():
    model_path = os.path.join(os.getenv('MODELS'), _ver())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path
    