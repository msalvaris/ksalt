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
# ---

# +
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
plt.style.use("seaborn-white")
import seaborn as sns
sns.set_style("white")
from sklearn.model_selection import train_test_split
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import logging
import random
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import defaultdict
import json
# -

from image_processing import upsample
from data import prepare_data, TGSSaltDataset
from model import model_path, save_checkpoint, update_state, predict_tta, model_identifier
from resnet34_unet_hyper import UNetResNetSCSE
from training import train, test, RefineStep, RefineTestStep
from utils import tboard_log_path
from losses import lovasz_hinge
from metrics import my_iou_metric, iou_metric_batch
from visualisation import plot_poor_predictions, plot_predictions
from config import load_config, save_config, default_config_path
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

config = load_config(config=config_path)["RefineModel"]
logger.info(f"Loading config {json.dumps(config, indent=4)}")

locals().update(config)


torch.backends.cudnn.benchmark = True
logger.info(f"Started {now}")
tboard_log = os.path.join(tboard_log_path(), f"log_refine_{id}")
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

model_dir = os.path.join(model_path(), f"{id}")
filename = os.path.join(model_dir, initial_model_filename)
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint["state_dict"])

train_df, test_df = prepare_data()
train_df.head()

upsample_to = upsample(101, img_target_size)

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

# Augment data with flipped verisons
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
x_train.shape

dataset = TGSSaltDataset(x_train, y_data=y_train)
dataset_val = TGSSaltDataset(x_valid, y_data=y_valid)

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

state = {
    "state_dict": None,
    "optimizer": None,
    "epoch": 0,
    "val_iou": 0,
    "best_val_iou": 0,
    "best_epoch": 0,
}

metrics = (("iou", my_iou_metric(threshold=0)),)
loss_fn = lovasz_hinge
optimizer = torch.optim.Adam(
    model.parameters(), lr=optimization_config["learning_rate"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10, verbose=True, threshold=0.0001
)


# +
step_func = RefineStep(
    loss_fn,
    scheduler,
    optimizer,
    summary_writer=summary_writer,
    metrics_func=metrics
)

test_step_func = RefineTestStep(loss_fn, summary_writer=summary_writer)
# -

lovasz_history = defaultdict(list)
for epoch in range(epochs):
    train_metrics = train(
        epoch, model, train_data_loader, step_func, summary_writer=summary_writer
    )

    val_metrics = test(
        epoch,
        model,
        val_data_loader,
        test_step_func,
        summary_writer=summary_writer
    )
    scheduler.step(np.mean(val_metrics["loss"]))
    state = update_state(
        state, epoch, "val_iou", np.mean(val_metrics["iou"]), model, optimizer
    )

    save_checkpoint(
        state,
        outdir=model_dir,
        model_filename=model_filename,
        best_model_filename=best_model_filename,
    )

    lovasz_history["epoch"].append(epoch)
    lovasz_history["train_loss"].append(np.mean(train_metrics["loss"]))
    lovasz_history["val_loss"].append(np.mean(val_metrics["loss"]))
    lovasz_history["train_iou"].append(np.mean(train_metrics["iou"]))
    lovasz_history["val_iou"].append(np.mean(val_metrics["iou"]))

fig, (ax_loss, ax_iou) = plt.subplots(1, 2, figsize=(15, 5))
ax_loss.plot(lovasz_history["epoch"], lovasz_history["train_loss"], label="Train loss")
ax_loss.plot(
    lovasz_history["epoch"], lovasz_history["val_loss"], label="Validation loss"
)
ax_loss.legend()
ax_iou.plot(lovasz_history["epoch"], lovasz_history["train_iou"], label="Train IoU")
ax_iou.plot(lovasz_history["epoch"], lovasz_history["val_iou"], label="Validation IoU")
ax_iou.legend()

# ### Find Optimal Threshold

filename = os.path.join(model_dir, best_model_filename)
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint["state_dict"])

# +
model.eval()
predictions = [predict_tta(model, image) for image, _ in tqdm(val_data_loader)]
preds_valid = np.concatenate(predictions, axis=0).squeeze()

preds_thresh_iter = map(
    lambda pred: np.array(np.round(pred > 0), dtype=np.float32), preds_valid
)

preds_thresh = np.array(list(preds_thresh_iter))


plot_predictions(
    train_df, preds_thresh, ids_valid, max_images=15, grid_width=5, figsize=(16, 10)
)

# +
## Scoring for last model, choose threshold using validation data
thresholds = np.linspace(0.3, 0.7, 31)
y_valid_down = y_valid.squeeze()
thresholds = np.log(thresholds / (1 - thresholds))

ious = list(
    map(
        lambda th: iou_metric_batch(y_valid_down, np.int32(preds_valid > th)),
        tqdm(thresholds),
    )
)

threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
# -

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))

# +
preds_thresh_iter = map(
    lambda pred: np.array(np.round(pred > threshold_best), dtype=np.float32),
    preds_valid,
)
preds_thresh = np.array(list(preds_thresh_iter))

plot_predictions(
    train_df, preds_thresh, ids_valid, max_images=15, grid_width=5, figsize=(16, 10)
)

plt.legend()
# -

plot_poor_predictions(
    train_df,
    preds_thresh,
    y_valid_down,
    ids_valid,
    max_images=15,
    grid_width=5,
    figsize=(16, 10),
)

dd = train_df.loc[ids_valid]

dd.groupby("coverage_class").mean().iou.plot(kind="bar")

dd.groupby("coverage_class").count().z.plot(kind="bar")

dd["iou"].mean()

# write best threshold to config
config = load_config()
config["EvaluateModel"]["threshold"] = threshold_best
save_config(config)


