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
            data_tensor, normalize=normalize, scale_each=True)
        summary_writer.add_image(label, image_grid, epoch)

    return write_to


def _add_metrics(metric_collector_dict, metrics_funcs, mask_array, pred_array):
    for metric, metric_func in metrics_funcs:
        metric_collector_dict[metric].append(metric_func(mask_array, pred_array))
    return metric_collector_dict


def train(epoch,
          model,
          optimizer,
          scheduler,
          criterion,
          train_loader,
          config,
          summary_writer=None,
          global_counter=None,
          metrics_funcs=(('iou', my_iou_metric),)):
    logger.info('Train {}'.format(epoch))

    image_writer = create_image_writer(summary_writer)
    model.train()

    train_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(train_loader):
        global_step = next(global_counter) if global_counter is not None else step
        scheduler.step()
        
        if summary_writer is not None:
            summary_writer.add_scalar('Train/LearningRate',
                                      scheduler.get_lr()[0], global_step)
            if step == 0:
                image_writer(image, 'Train/Image', epoch, normalize=True)


        with torch.cuda.device(0):
            image = image.type(torch.float).cuda(async=True)
            mask_gpu = mask.type(torch.float).cuda(async=True)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask_gpu)

        loss.backward()
        optimizer.step()

        output_cpu = output.cpu()
        train_metrics['loss'].append(loss.item())
        train_metrics = _add_metrics(train_metrics, metrics_funcs, mask.numpy(), output_cpu.data.numpy())

        if summary_writer is not None:
            summary_writer.add_scalar('Train/RunningLoss', train_metrics['loss'][-1], global_step)
            summary_writer.add_scalar('Train/RunningIoU', train_metrics['iou'][-1], global_step)
            if step == 0:
                image_writer(mask, 'Train/Mask', epoch)
                image_writer(output_cpu, 'Train/Prediction', epoch)

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
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if summary_writer:
        summary_writer.add_scalar('Train/Loss', np.mean(train_metrics['loss']), epoch)
        summary_writer.add_scalar('Train/IoU', np.mean(train_metrics['iou']), epoch)
        summary_writer.add_scalar('Train/Time', elapsed, epoch)

    return train_metrics


def test(epoch, 
         model, 
         criterion, 
         test_loader, 
         summary_writer=None, 
         metrics_funcs=(('iou', my_iou_metric),)):
    logger.info('Test {}'.format(epoch))

    image_writer = create_image_writer(summary_writer)

    model.eval()

    val_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(test_loader):

        if summary_writer is not None and step == 0:
            image_writer(image, 'Test/Image', epoch, normalize=True)

        with torch.cuda.device(0):
            image = image.type(torch.float).cuda(async=True)
            mask_gpu = mask.type(torch.float).cuda(async=True)

        with torch.no_grad():
            output = model(image)
        loss = criterion(output, mask_gpu)

        output_cpu = output.cpu()
        if summary_writer is not None and step == 0:
            image_writer(mask, 'Test/Mask', epoch)
            image_writer(output_cpu, 'Test/Prediction', epoch)

        val_metrics['loss'].append(loss.item())
        val_metrics = _add_metrics(val_metrics, metrics_funcs, mask.numpy(), output_cpu.data.numpy())

    message = (
        f"Epoch: {epoch},"
        f"Val: loss {np.mean(val_metrics['loss']):.3f}, IoU {np.mean(val_metrics['iou']):.3f} ** "
    )
    logger.info(message)

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if summary_writer:
        summary_writer.add_scalar('Test/Loss', np.mean(val_metrics['loss']), epoch)
        summary_writer.add_scalar('Test/IoU', np.mean(val_metrics['iou']), epoch)
        summary_writer.add_scalar('Test/Time', elapsed, epoch)

    return val_metrics
