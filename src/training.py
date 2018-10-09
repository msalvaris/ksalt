import itertools as it
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




class TrainStep(object):
    def __init__(
        self,
        criterion,
        scheduler,
        optimizer,
        summary_writer=None,
        metrics_func=(("iou", my_iou_metric),),
    ):
        super().__init__()
        self._scheduler = scheduler
        self._optimizer = optimizer
        self._summary_writer = summary_writer
        self._criterion = criterion

        self._image_writer = create_image_writer(summary_writer)
        self._metrics_func = metrics_func
        self._step_counter = it.count()
        self._previous_epoch = None

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def set_scheduler(self, scheduler):
        self._scheduler = scheduler

    def _optimize(self, model, image, mask):
        with torch.cuda.device(0):
            image = image.type(torch.float).cuda(async=True)
            mask_gpu = mask.type(torch.float).cuda(async=True)

        self._optimizer.zero_grad()
        output = model(image)
        loss = self._criterion(output, mask_gpu)

        loss.backward()
        self._optimizer.step()
        return output, loss

    def _to_tensorboard(self, epoch, image, mask, metrics, output_cpu):
        global_step = next(self._step_counter)
        if self._summary_writer is not None:
            lr = self._optimizer.param_groups[0]["lr"]  # scheduler.get_lr()[0]
            self._summary_writer.add_scalar("Train/LearningRate", lr, global_step)
            self._summary_writer.add_scalar(
                "Train/RunningLoss", metrics["loss"], global_step
            )
            self._summary_writer.add_scalar(
                "Train/RunningIoU", metrics["iou"], global_step
            )
            if self._previous_epoch != epoch:
                self._previous_epoch = epoch
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

    def __call__(self, model, image, mask, epoch):
        self._scheduler.step()
        output, loss = self._optimize(model, image, mask)
        output_cpu = torch.sigmoid(output).cpu()
        train_metrics = self._metrics(output_cpu, loss, mask)
        self._to_tensorboard(epoch, image, mask, train_metrics, output_cpu)
        return train_metrics


class RefineStep(TrainStep):
    def __init__(
        self,
        criterion,
        scheduler,
        optimizer,
        summary_writer=None,
        metrics_func=(("iou", my_iou_metric(threshold=0)),),
    ):
        super().__init__(criterion, scheduler, optimizer, summary_writer, metrics_func)

    def __call__(self, model, image, mask, epoch):
        output, loss = self._optimize(model, image, mask)
        output_cpu = output.cpu()
        train_metrics = self._metrics(output_cpu, loss, mask)
        self._to_tensorboard(
            epoch, image, mask, train_metrics, torch.sigmoid(output_cpu)
        )
        return train_metrics


def train(epoch, model, train_loader, tain_step_func, summary_writer=None):
    logger.info("Train {}".format(epoch))
    model.train()

    train_metrics = defaultdict(list)
    start = time.time()
    for image, mask in train_loader:
        metrics = tain_step_func(model, image, mask, epoch)

        for key, item in metrics.items():
            train_metrics[key].append(item)

    elapsed = time.time() - start
    message = (
        f"Epoch: {epoch},"
        f"Train: loss {np.mean(train_metrics['loss']):.3f},  IoU {np.mean(train_metrics['iou']):.3f}"
    )
    logger.info(message)
    logger.info("Elapsed {:.2f}".format(elapsed))

    if summary_writer:
        summary_writer.add_scalar("Train/Loss", np.mean(train_metrics["loss"]), epoch)
        summary_writer.add_scalar("Train/IoU", np.mean(train_metrics["iou"]), epoch)
        summary_writer.add_scalar("Train/Time", elapsed, epoch)

    return train_metrics


class TestStep(object):
    def __init__(
        self, criterion, summary_writer=None, metrics_func=(("iou", my_iou_metric),)
    ):
        super(TestStep).__init__()
        self._metrics_func = metrics_func
        self._summary_writer = summary_writer
        self._criterion = criterion
        self._image_writer = create_image_writer(summary_writer)
        self._previous_epoch=None

    def _evaluate(self, model, image, mask):
        with torch.cuda.device(0):
            image = image.type(torch.float).cuda(async=True)
            mask_gpu = mask.type(torch.float).cuda(async=True)

        with torch.no_grad():
            output = model(image)
        loss = self._criterion(output, mask_gpu)
        return output, loss

    def _to_tensorboard(self, epoch, image, mask, output_cpu):
        if self._summary_writer is not None and self._previous_epoch != epoch:
            self._previous_epoch = epoch
            self._image_writer(image, "Test/Image", epoch, normalize=True)
            self._image_writer(mask, "Test/Mask", epoch)
            self._image_writer(output_cpu, "Test/Prediction", epoch)

    def _metrics(self, output_cpu, loss, mask):
        train_metrics = {}
        train_metrics["loss"] = loss.item()
        train_metrics = _add_metrics(
            train_metrics, self._metrics_func, mask.numpy(), output_cpu.data.numpy()
        )
        return train_metrics

    def __call__(self, model, image, mask, epoch):
        output, loss = self._evaluate(model, image, mask)
        output_cpu = torch.sigmoid(output).cpu()
        test_metrics = self._metrics(output_cpu, loss, mask)
        self._to_tensorboard(epoch, image, mask, output_cpu)
        return test_metrics


class RefineTestStep(TestStep):
    def __init__(
        self, criterion, summary_writer=None, metrics_func=(("iou", my_iou_metric(threshold=0)),)
    ):
        super().__init__(criterion, summary_writer, metrics_func)

    def __call__(self, model, image, mask, epoch):
        output, loss = self._evaluate(model, image, mask)
        output_cpu = output.cpu()
        test_metrics = self._metrics(output_cpu, loss, mask)
        self._to_tensorboard(epoch, image, mask, torch.sigmoid(output_cpu))
        return test_metrics


def test(epoch, model, test_loader, test_step_func, summary_writer=None):
    logger.info("Test {}".format(epoch))
    model.eval()

    val_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(test_loader):

        metrics = test_step_func(model, image, mask, epoch)

        for key, item in metrics.items():
            val_metrics[key].append(item)

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
