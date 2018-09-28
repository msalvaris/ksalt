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

# +
from image_processing import downsample
from data import test_images_path, load_images_as_arrays, TGSSaltDataset, prepare_test_data
from visualisation import (
    plot_predictions,
)
from model import model_path
from metrics import iou_metric_batch
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
import logging
import random
import uuid

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
optim_config = {
    "optimizer": "sgd",
    "base_lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "nesterov": True,
    "epochs": epochs,
    "scheduler": "cosine",
    "lr_min": 0,
}

logging.basicConfig(level=logging.INFO)
torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)
logger.info(f"Started {now}")

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model = UNetResNet(1, base_channels)

device = torch.device("cuda:0")
model = nn.DataParallel(model)
model.to(device)

filename = os.path.join(model_path(), "model_lovasz_best_state.pth")
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint["state_dict"])

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

# +
model.eval()
predictions = [predict_tta(model, image) for image, _ in tqdm(val_data_loader)]

preds_valid = np.concatenate(predictions, axis=0).squeeze()
downsample_to = downsample(128, 101)
preds_valid = np.array(list(map(downsample_to, preds_valid)))
plot_predictions(
    train_df, preds_valid, ids_valid, max_images=15, grid_width=5, figsize=(16, 10)
)

# +
## Scoring for last model, choose threshold using validation data
thresholds = np.linspace(0.3, 0.7, 31)
y_valid_down = np.array(list(map(downsample_to, y_valid.squeeze())))
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

plot_predictions(
    train_df, preds_thresh, ids_valid, max_images=15, grid_width=5, figsize=(16, 10)
)
preds_thresh = np.array(list(map(downsample_to, preds_thresh_iter)))
plt.legend()
# -
test_df = prepare_test_data()
x_test = load_images_as_arrays(test_df.index, test_images_path())
x_test = x_test.reshape(-1, 1, img_size_target, img_size_target)

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

transform = compose(rle_encode, np.round, downsample_to, lambda x: x > threshold_best)

pred_dict = {
    idx: transform(preds_test[i]) for i, idx in enumerate(tqdm(test_df.index.values))
}

sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]
filename = os.path.join(model_path(), f"submission_{now:%d%b%Y_%H}.csv")
sub.to_csv(filename)
