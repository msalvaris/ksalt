import logging
import time
from collections import defaultdict

import numpy as np
import torch
import torchvision

from metrics import my_iou_metric

logger = logging.getLogger(__name__)


def create_image_writer(summary_writer):
    def write_to(data_tensor, label, epoch, normalize=False):
        image_grid = torchvision.utils.make_grid(
            data_tensor, normalize=normalize, scale_each=True
        )
        summary_writer.add_image(label, image_grid, epoch)

    return write_to


def _add_metrics(metric_collector_dict, metrics_funcs, mask_array, pred_array):
    for metric, metric_func in metrics_funcs:
        metric_collector_dict[metric] = metric_func(mask_array, pred_array)
    return metric_collector_dict


class TrainingStep(object):
    def __init__(self):
        super(TrainingStep).__init__()

    def __call__(self, epoch, step, image, mask):
        raise NotImplementedError("Please implement __call__ method")


class CycleStep(TrainingStep):
    def __init__(
        self,
        model,
        criterion,
        scheduler,
        optimizer,
        summary_writer=None,
        global_counter=None,
        metrics_func=(("iou", my_iou_metric),),
    ):
        super(TrainingStep).__init__()
        self._scheduler = scheduler
        self._optimizer = optimizer
        self._summary_writer = summary_writer
        self._global_counter = global_counter
        self._model = model
        self._criterion = criterion

        self._image_writer = create_image_writer(summary_writer)
        self._metrics_func = metrics_func

    def _optimize(self, image, mask):
        with torch.cuda.device(0):
            image = image.type(torch.float).cuda(async=True)
            mask_gpu = mask.type(torch.float).cuda(async=True)

        self._optimizer.zero_grad()
        output = self._model(image)
        loss = self._criterion(output, mask_gpu)

        loss.backward()
        self._optimizer.step()
        return output, loss

    def _to_tensorboard(self, epoch, step, image, mask, metrics, output_cpu):
        global_step = (
            next(self._global_counter) if self._global_counter is not None else step
        )
        if self._summary_writer is not None:
            lr = self._optimizer.param_groups[0]["lr"]  # scheduler.get_lr()[0]
            self._summary_writer.add_scalar("Train/LearningRate", lr, global_step)
            self._summary_writer.add_scalar(
                "Train/RunningLoss", metrics["loss"], global_step
            )
            self._summary_writer.add_scalar(
                "Train/RunningIoU", metrics["iou"], global_step
            )
            if step == 0:
                self._image_writer(image, "Train/Image", epoch, normalize=True)
                self._image_writer(mask, "Train/Mask", epoch)
                self._image_writer(output_cpu, "Train/Prediction", epoch)

    def _metrics(self, output_cpu, loss, mask):
        train_metrics = {}
        train_metrics["loss"] = loss.item()
        train_metrics = _add_metrics(
            train_metrics, self._metrics_func, mask.numpy(), output_cpu.data.numpy()
        )
        return train_metrics

    def __call__(self, epoch, step, image, mask):
        self._scheduler.step()
        output, loss = self._optimize(image, mask)
        output_cpu = output.cpu()
        train_metrics = self._metrics(output_cpu, loss, mask)
        self._to_tensorboard(epoch, step, image, mask, train_metrics, output_cpu)
        return train_metrics


class RefineStep(CycleStep):
    def __init__(
        self,
        model,
        criterion,
        scheduler,
        optimizer,
        summary_writer=None,
        global_counter=None,
        metrics_func=(("iou", my_iou_metric),),
    ):
        super(RefineStep).__init__(
            model,
            criterion,
            scheduler,
            optimizer,
            summary_writer=summary_writer,
            global_counter=global_counter,
            metrics_func=metrics_func,
        )

    def __call__(self, epoch, step, image, mask):
        output, loss = self._optimize(image, mask)
        output_cpu = output.cpu()
        train_metrics = self._metrics(output_cpu, loss, mask)
        self._to_tensorboard(epoch, step, image, mask, train_metrics, output_cpu)
        return train_metrics


def train(epoch, model, train_loader, tain_step_func, summary_writer=None):
    logger.info("Train {}".format(epoch))
    model.train()

    train_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(train_loader):
        metrics = tain_step_func(epoch, step, image, mask)

        for key, item in metrics.items():
            train_metrics[key].append(item)

        if step % 100 == 0:
            message = (
                f"Epoch: {epoch},"
                f"Step: {step},"
                f"Train: loss {np.mean(train_metrics['loss']):.3f},  IoU {np.mean(train_metrics['iou']):.3f}"
            )
            logger.info(message)

    elapsed = time.time() - start
    message = (
        f"Epoch: {epoch},"
        f"Step: {step},"
        f"Train: loss {np.mean(train_metrics['loss']):.3f},  IoU {np.mean(train_metrics['iou']):.3f}"
    )
    logger.info(message)
    logger.info("Elapsed {:.2f}".format(elapsed))

    if summary_writer:
        summary_writer.add_scalar("Train/Loss", np.mean(train_metrics["loss"]), epoch)
        summary_writer.add_scalar("Train/IoU", np.mean(train_metrics["iou"]), epoch)
        summary_writer.add_scalar("Train/Time", elapsed, epoch)

    return train_metrics


def test(
    epoch,
    model,
    criterion,
    test_loader,
    summary_writer=None,
    metrics_funcs=(("iou", my_iou_metric),),
):
    logger.info("Test {}".format(epoch))

    image_writer = create_image_writer(summary_writer)

    model.eval()

    val_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(test_loader):

        if summary_writer is not None and step == 0:
            image_writer(image, "Test/Image", epoch, normalize=True)

        with torch.cuda.device(0):
            image = image.type(torch.float).cuda(async=True)
            mask_gpu = mask.type(torch.float).cuda(async=True)

        with torch.no_grad():
            output = model(image)
        loss = criterion(output, mask_gpu)

        output_cpu = output.cpu()
        if summary_writer is not None and step == 0:
            image_writer(mask, "Test/Mask", epoch)
            image_writer(output_cpu, "Test/Prediction", epoch)

        val_metrics["loss"].append(loss.item())
        val_metrics = _add_metrics(
            val_metrics, metrics_funcs, mask.numpy(), output_cpu.data.numpy()
        )

    message = (
        f"Epoch: {epoch},"
        f"Val: loss {np.mean(val_metrics['loss']):.3f}, IoU {np.mean(val_metrics['iou']):.3f} ** "
    )
    logger.info(message)

    elapsed = time.time() - start
    logger.info("Elapsed {:.2f}".format(elapsed))

    if summary_writer:
        summary_writer.add_scalar("Test/Loss", np.mean(val_metrics["loss"]), epoch)
        summary_writer.add_scalar("Test/IoU", np.mean(val_metrics["iou"]), epoch)
        summary_writer.add_scalar("Test/Time", elapsed, epoch)

    return val_metrics
