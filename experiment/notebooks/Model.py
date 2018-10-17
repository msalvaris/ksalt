# ---
# jupyter:
#   celltoolbar: Tags
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.2
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
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import logging
import random
import itertools as it
from operator import itemgetter
import shutil
from tensorboardX import SummaryWriter
import json

# + {"papermill": {"duration": 0.098801, "end_time": "2018-09-17T13:01:47.892280", "exception": false, "start_time": "2018-09-17T13:01:47.793479", "status": "completed"}, "tags": []}
from image_processing import upsample
from data import prepare_data, TGSSaltDataset
from visualisation import (
    plot_coverage_and_coverage_class,
    scatter_coverage_and_coverage_class,
    plot_depth_distributions,
    plot_images,
)
from model import model_path, save_checkpoint, update_state, model_identifier
from training import train, test, TrainStep, TestStep
from collections import defaultdict
from utils import create_optimizer, tboard_log_path
from config import load_config, default_config_path
from resnet34_unet_hyper import UNetResNetSCSE
# -

try:
    from azureml.core.run import Run, RunEnvironmentException
    run = Run.get_context()
except (ModuleNotFoundError, RunEnvironmentException):
    from run_mock import RunMock
    run = RunMock()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

now = datetime.datetime.now()

# + {"tags": ["parameters"]}
config_path=default_config_path()
# -

config = load_config(config=config_path)["Model"]
logger.info(f"Loading config {json.dumps(config, indent=4)}")

locals().update(config)

torch.backends.cudnn.benchmark = True
logger.info(f"Started {now}")
tboard_log = os.path.join(tboard_log_path(), f"log_{id}")
logger.info(f"Writing TensorBoard logs to {tboard_log}")
summary_writer = SummaryWriter(log_dir=tboard_log)
run.tag('id', value=id)
run.tag('model_id', value=model_identifier())

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model = UNetResNetSCSE()

n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
logger.info("n_params: {}".format(n_params))

device = torch.device("cuda:0")
# model = nn.DataParallel(model)
model.to(device)

# Test to check network
x = torch.randn(16, 1, img_target_size, img_target_size).cuda()
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

upsample_to = upsample(101, img_target_size)

# + {"papermill": {"duration": 10.282893, "end_time": "2018-09-17T13:02:07.336485", "exception": false, "start_time": "2018-09-17T13:01:57.053592", "status": "completed"}, "tags": []}
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample_to).tolist()).reshape(
        -1, 1, img_target_size, img_target_size
    ),
    np.array(train_df.masks.map(upsample_to).tolist()).reshape(
        -1, 1, img_target_size, img_target_size
    ),
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.1,
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

optimization_config["steps_per_epoch"] = len(train_data_loader)
optimization_config["epochs"] = epochs

model_dir = os.path.join(model_path(), f"{id}")
os.makedirs(model_dir, exist_ok=True)

# +
history = defaultdict(list)
loss_fn = torch.nn.BCEWithLogitsLoss()

cumulative_epochs_counter = it.count()
cycle_best_val_iou = {}

optimizer, scheduler = create_optimizer(model.parameters(), optimization_config)
step_func = TrainStep(loss_fn, scheduler, optimizer, summary_writer=summary_writer)
test_step_func = TestStep(loss_fn, summary_writer=summary_writer)

for cycle in range(
    optimization_config["scheduler"]["num_cycles"]
):  # Cosine annealing with warm restarts
    optimizer, scheduler = create_optimizer(model.parameters(), optimization_config)
    step_func.set_optimizer(optimizer)
    step_func.set_scheduler(scheduler)
    for epoch in range(epochs):
        cum_epoch = next(cumulative_epochs_counter)
        train_metrics = train(
            cum_epoch,
            model,
            train_data_loader,
            step_func,
            summary_writer=summary_writer,
        )

        val_metrics = test(
            cum_epoch, model, val_data_loader, test_step_func, summary_writer=summary_writer
        )

        state = update_state(
            state, cum_epoch, "val_iou", np.mean(val_metrics["iou"]), model, optimizer
        )

        save_checkpoint(
            state,
            outdir=model_dir,
            best_model_filename=model_filename.format(cycle=cycle),
        )

        history["epoch"].append(cum_epoch)
        history["train_loss"].append(np.mean(train_metrics["loss"]))
        history["val_loss"].append(np.mean(val_metrics["loss"]))
        history["train_iou"].append(np.mean(train_metrics["iou"]))
        history["val_iou"].append(np.mean(val_metrics["iou"]))
        run.log('val_loss', history["val_loss"][-1])
        run.log('val_iou', history["val_iou"][-1])
    cycle_best_val_iou[cycle] = state["best_val_iou"]
# -

sorted_by_val_iou = sorted(cycle_best_val_iou.items(), key=itemgetter(1), reverse=True)
best_cycle, best_iou = sorted_by_val_iou[0]
logger.info(f"Best model cycle {best_cycle}: Validation IoU {best_iou}")
run.log('best_val_iou', best_iou)
logger.info(f"Saving to {best_model_filename}")
shutil.copy(
    os.path.join(model_dir, model_filename.format(cycle=best_cycle)),
    os.path.join(model_dir, best_model_filename),
)

# + {"papermill": {"duration": 0.371981, "end_time": "2018-09-17T13:10:01.845367", "exception": false, "start_time": "2018-09-17T13:10:01.473386", "status": "completed"}, "tags": []}
fig, (ax_loss, ax_iou) = plt.subplots(1, 2, figsize=(15, 5))
ax_loss.plot(history["epoch"], history["train_loss"], label="Train loss")
ax_loss.plot(history["epoch"], history["val_loss"], label="Validation loss")
ax_loss.legend()
ax_iou.plot(history["epoch"], history["train_iou"], label="Train IoU")
ax_iou.plot(history["epoch"], history["val_iou"], label="Validation IoU")
ax_iou.legend()
