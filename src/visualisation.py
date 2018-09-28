import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from metrics import _thresholded_iou_for


def plot_coverage_and_coverage_class(coverage, coverage_class):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.distplot(coverage, kde=False, ax=axs[0])
    sns.distplot(coverage_class, bins=10, kde=False, ax=axs[1])
    plt.suptitle("Salt coverage")
    axs[0].set_xlabel("Coverage")
    axs[1].set_xlabel("Coverage class")


def scatter_coverage_and_coverage_class(coverage, coverage_class):
    plt.scatter(coverage, coverage_class)
    plt.xlabel("Coverage")
    plt.ylabel("Coverage class")


def plot_depth_distributions(train_depth, test_depth):
    sns.distplot(train_depth, label="Train")
    sns.distplot(test_depth, label="Test")
    plt.legend()
    plt.title("Depth distribution")


def plot_images(train_df, img_size_ori=101, max_images=60, grid_width=15, figsize=()):
    grid_height = int(max_images / grid_width)
    figsize = (grid_width, grid_height) if figsize == () else figsize
    fig, axs = plt.subplots(grid_height, grid_width, figsize=figsize)
    for i, idx in enumerate(train_df.index[:max_images]):
        img = train_df.loc[idx].images
        mask = train_df.loc[idx].masks
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.text(1, img_size_ori - 1, train_df.loc[idx].z, color="black")
        ax.text(
            img_size_ori - 1,
            1,
            round(train_df.loc[idx].coverage, 2),
            color="black",
            ha="right",
            va="top",
        )
        ax.text(
            1, 1, train_df.loc[idx].coverage_class, color="black", ha="left", va="top"
        )
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.suptitle(
        "Green: salt. Top-left: coverage class, top-right: salt coverage, bottom-left: depth"
    )


def plot_predictions(
    train_df,
    predictions,
    image_ids,
    img_size_ori=101,
    max_images=60,
    grid_width=15,
    figsize=(),
):
    grid_height = int(max_images / grid_width)
    figsize = (grid_width, grid_height) if figsize == () else figsize
    fig, axs = plt.subplots(grid_height, grid_width, figsize=figsize)
    for i, idx in enumerate(image_ids[:max_images]):
        img = train_df.loc[idx].images
        mask = train_df.loc[idx].masks
        pred = predictions[i]
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.imshow(pred, alpha=0.3, cmap="OrRd")
        ax.text(1, img_size_ori - 1, train_df.loc[idx].z, color="black")
        ax.text(
            img_size_ori - 1,
            1,
            round(train_df.loc[idx].coverage, 2),
            color="black",
            ha="right",
            va="top",
        )
        ax.text(
            1, 1, train_df.loc[idx].coverage_class, color="black", ha="left", va="top"
        )
        if 'iou' in train_df:
            ax.text(
                1, 1, np.round(train_df.loc[idx].iou, decimals=2), color="black", ha="right", verticalalignment="baseline"
            )
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.suptitle(
        "Green: salt, Red: prediction. Top-left: coverage class, top-right: salt coverage, bottom-left: depth"
    )


def plot_poor_predictions(
    train_df,
    predictions,
    labels,
    image_ids,
    iou_threshold=0.0,
    img_size_ori=101,
    max_images=60,
    grid_width=15,
    figsize=(),
):
    iou_iter = (
        _thresholded_iou_for(1, label, pred) for pred, label in zip(predictions, labels)
    )
    iou_series = pd.Series(list(iou_iter), index=image_ids)
    reindex_series = pd.Series(range(len(image_ids)), index=image_ids)
    image_ids = iou_series[iou_series <= iou_threshold].index
    predictions = np.array([predictions[i] for i in reindex_series.loc[image_ids]])
    train_df['iou'] = iou_series
    plot_predictions(
        train_df,
        predictions,
        image_ids,
        img_size_ori=img_size_ori,
        max_images=max_images,
        grid_width=grid_width,
        figsize=figsize,
    )
