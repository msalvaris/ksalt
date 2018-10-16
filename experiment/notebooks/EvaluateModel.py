# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   jupytext_formats: ipynb,py
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.6
# ---

# +
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
plt.style.use("seaborn-white")
import seaborn as sns
sns.set_style("white")
from torch import nn
from tqdm import tqdm
from toolz import compose
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils import data
import logging
import random
import os
# -

from data import (
    test_images_path,
    load_images_as_arrays,
    TGSSaltDataset,
    prepare_test_data,
)
from model import model_path, predict_tta
from data import rle_encode
from resnetlike import UNetResNet
from config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

now = datetime.datetime.now()

config = load_config()["EvaluateModel"]
logger.info(f"Loading config {config}")

locals().update(config)


torch.backends.cudnn.benchmark = True
logger.info(f"Started {now}")

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model = UNetResNet(1, base_channels)

device = torch.device("cuda:0")
model = nn.DataParallel(model)
model.to(device)

model.module.final_activation = nn.Sequential().to(device)

model_dir = os.path.join(model_path(), f"{id}")
filename = os.path.join(model_dir, initial_model_filename)
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint["state_dict"])

test_df = prepare_test_data()
x_test = np.array(load_images_as_arrays(test_df.index, test_images_path()))
x_test = x_test.reshape(-1, 1, img_target_size, img_target_size)

dataset_test = TGSSaltDataset(x_test, is_test=True)

test_data_loader = data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)

model.eval()
predictions = [predict_tta(model, image) for image in tqdm(test_data_loader)]

preds_test = np.concatenate(predictions, axis=0).squeeze()

transform = compose(rle_encode, np.round, lambda x: x > threshold)

pred_dict = {
    idx: transform(preds_test[i]) for i, idx in enumerate(tqdm(test_df.index.values))
}

sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]
filename = os.path.join(model_dir, f"submission_{now:%d%b%Y_%H}.csv")
sub.to_csv(filename)
