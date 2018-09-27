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
#   papermill:
#     duration: 1005.854818
#     end_time: '2018-09-17T13:18:30.811497'
#     environment_variables: {}
#     exception: false
#     output_path: notebooks/Model.ipynb
#     parameters: {}
#     start_time: '2018-09-17T13:01:44.956679'
#     version: 0.15.0
# ---

# + {"papermill": {"duration": 2.030166, "end_time": "2018-09-17T13:01:47.793244", "exception": false, "start_time": "2018-09-17T13:01:45.763078", "status": "completed"}, "tags": []}
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt

plt.style.use("seaborn-white")
import seaborn as sns

sns.set_style("white")

from sklearn.model_selection import train_test_split

from torch import nn

from tqdm import tqdm
from torch.nn import Sequential

# + {"papermill": {"duration": 0.098801, "end_time": "2018-09-17T13:01:47.892280", "exception": false, "start_time": "2018-09-17T13:01:47.793479", "status": "completed"}, "tags": []}
from image_processing import upsample, downsample
from data import prepare_data, test_images_path, load_images_as_arrays, TGSSaltDataset
from visualisation import (
    plot_coverage_and_coverage_class,
    scatter_coverage_and_coverage_class,
    plot_depth_distributions,
    plot_predictions,
    plot_images,
)
from model import model_path, save_checkpoint, update_state
from metrics import iou_metric_batch, my_iou_metric
from toolz import compose
from data import rle_encode
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.utils import data

from resnetlike import UNetResNet
from training import train, test
from collections import defaultdict
import logging
import random
from utils import create_optimizer, tboard_log_path
import uuid
import itertools as it
from operator import itemgetter
import shutil
from losses import lovasz_hinge

# -

now = datetime.datetime.now()

img_size_target = 101
batch_size = 128
learning_rate = 0.1
epochs = 70
num_workers = 0
seed = 42
num_cycles = (
    6
)  # Using Cosine Annealing with warm restarts, the number of times to oscillate
notebook_id = f"{now:%d%b%Y}_{uuid.uuid4()}"
base_channels = 32
config = {
    "run_config": {
        "arch": "shake_shake",
        "base_channels": 64,
        "depth": 26,
        "shake_forward": True,
        "shake_backward": True,
        "shake_image": True,
        "input_shape": (1, 1, img_size_target, img_size_target),
    },
    "optim_config": {
        "optimizer": "sgd",
        "base_lr": learning_rate,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "nesterov": True,
        "epochs": epochs,
        "scheduler": "cosine",
        "lr_min": 0,
    },
}

logging.basicConfig(level=logging.INFO)
torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)
logger.info(f"Started {now}")
tboard_log = os.path.join(tboard_log_path(), f"log_{notebook_id}")
logger.info(f"Writing TensorBoard logs to {tboard_log}")
summary_writer = SummaryWriter(log_dir=tboard_log)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model = UNetResNet(1, base_channels)

n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
logger.info("n_params: {}".format(n_params))

device = torch.device("cuda:0")
model = nn.DataParallel(model)
model.to(device)

# Test to check network
x = torch.randn(16, 1, img_size_target, img_size_target).cuda()
model.forward(x).shape

# + {"papermill": {"duration": 5.000668, "end_time": "2018-09-17T13:01:52.893036", "exception": false, "start_time": "2018-09-17T13:01:47.892368", "status": "completed"}, "tags": []}
train_df, test_df = prepare_data()

# + {"papermill": {"duration": 1.318465, "end_time": "2018-09-17T13:01:54.211706", "exception": false, "start_time": "2018-09-17T13:01:52.893241", "status": "completed"}, "tags": []}
train_df.head()

# + {"papermill": {"duration": 0.404575, "end_time": "2018-09-17T13:01:54.616484", "exception": false, "start_time": "2018-09-17T13:01:54.211909", "status": "completed"}, "tags": []}
plot_coverage_and_coverage_class(train_df.coverage, train_df.coverage_class)

# + {"papermill": {"duration": 0.207382, "end_time": "2018-09-17T13:01:54.824061", "exception": false, "start_time": "2018-09-17T13:01:54.616679", "status": "completed"}, "tags": []}
scatter_coverage_and_coverage_class(train_df.coverage, train_df.coverage_class)

# + {"papermill": {"duration": 0.369911, "end_time": "2018-09-17T13:01:55.194125", "exception": false, "start_time": "2018-09-17T13:01:54.824214", "status": "completed"}, "tags": []}
plot_depth_distributions(train_df.z, test_df.z)

# + {"papermill": {"duration": 1.807894, "end_time": "2018-09-17T13:01:57.002195", "exception": false, "start_time": "2018-09-17T13:01:55.194301", "status": "completed"}, "tags": []}
plot_images(train_df, max_images=15, grid_width=5, figsize=(16, 10))
# -

upsample_to = upsample(101, img_size_target)

# + {"papermill": {"duration": 10.282893, "end_time": "2018-09-17T13:02:07.336485", "exception": false, "start_time": "2018-09-17T13:01:57.053592", "status": "completed"}, "tags": []}
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample_to).tolist()).reshape(
        -1, 1, img_size_target, img_size_target
    ),
    np.array(train_df.masks.map(upsample_to).tolist()).reshape(
        -1, 1, img_size_target, img_size_target
    ),
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2,
    stratify=train_df.coverage_class,
    random_state=seed,
)

# + {"papermill": {"duration": 1.039783, "end_time": "2018-09-17T13:02:09.213911", "exception": false, "start_time": "2018-09-17T13:02:08.174128", "status": "completed"}, "tags": []}
# Augment data with flipped verisons
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
# -

x_train.shape

dataset = TGSSaltDataset(x_train, y_data=y_train)
dataset_val = TGSSaltDataset(x_valid, y_data=y_valid)

state = {
    "state_dict": None,
    "optimizer": None,
    "epoch": 0,
    "val_iou": 0,
    "best_val_iou": 0,
    "best_epoch": 0,
}

train_data_loader = data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
)
val_data_loader = data.DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)

config["optim_config"]["steps_per_epoch"] = len(train_data_loader)

# +
history = defaultdict(list)
loss_fn = torch.nn.BCELoss()

global_counter = it.count()
cumulative_epochs_counter = it.count()
cycle_best_val_iou = {}
for cycle in range(num_cycles):  # Cosine annealing with warm restarts
    optimizer, scheduler = create_optimizer(model.parameters(), config["optim_config"])
    for epoch in range(epochs):
        cum_epoch = next(cumulative_epochs_counter)
        train_metrics = train(
            cum_epoch,
            model,
            optimizer,
            scheduler,
            loss_fn,
            train_data_loader,
            config,
            summary_writer=summary_writer,
            global_counter=global_counter,
        )

        val_metrics = test(
            cum_epoch, model, loss_fn, val_data_loader, summary_writer=summary_writer
        )

        state = update_state(
            state, cum_epoch, "val_iou", np.mean(val_metrics["iou"]), model, optimizer
        )

        save_checkpoint(state, best_model_filename=f"model_{cycle}_best_state.pth")

        history["epoch"].append(cum_epoch)
        history["train_loss"].append(np.mean(train_metrics["loss"]))
        history["val_loss"].append(np.mean(val_metrics["loss"]))
        history["train_iou"].append(np.mean(train_metrics["iou"]))
        history["val_iou"].append(np.mean(val_metrics["iou"]))
    cycle_best_val_iou[cycle] = state["best_val_iou"]
# -

sorted_by_val_iou = sorted(cycle_best_val_iou.items(), key=itemgetter(1), reverse=True)
best_cycle, best_iou = sorted_by_val_iou[0]
logger.info(f"Best model cycle {best_cycle}: Validation IoU {best_iou}")
logger.info("Saving to model_best_state.pth")
shutil.copy(
    os.path.join(model_path(), f"model_{best_cycle}_best_state.pth"),
    os.path.join(model_path(), f"model_best_state.pth"),
)

# + {"papermill": {"duration": 0.371981, "end_time": "2018-09-17T13:10:01.845367", "exception": false, "start_time": "2018-09-17T13:10:01.473386", "status": "completed"}, "tags": []}
fig, (ax_loss, ax_iou) = plt.subplots(1, 2, figsize=(15, 5))
ax_loss.plot(history["epoch"], history["train_loss"], label="Train loss")
ax_loss.plot(history["epoch"], history["val_loss"], label="Validation loss")
ax_loss.legend()
ax_iou.plot(history["epoch"], history["train_iou"], label="Train IoU")
ax_iou.plot(history["epoch"], history["val_iou"], label="Validation IoU")
ax_iou.legend()
