import logging
from collections import defaultdict
import time
import numpy as np
from metrics import my_iou_metric
import torch
import torchvision

logger = logging.getLogger(__name__)


def train(epoch, model, optimizer, scheduler, criterion, train_loader, config, summary_writter=None):
    logger.info('Train {}'.format(epoch))

    run_config = config['run_config']
    optim_config = config['optim_config']
    
    model.train()

    train_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(train_loader):
        train.global_step+=1
        if summary_writter is not None and step == 0:
            image_grid = torchvision.utils.make_grid(
                image, normalize=True, scale_each=True)
            summary_writter.add_image('Train/Image', image_grid, epoch)

        if optim_config['scheduler'] == 'multistep':
            scheduler.step(epoch - 1)
        elif optim_config['scheduler'] == 'cosine':
            scheduler.step()

        if summary_writter is not None:
            summary_writter.add_scalar('Train/LearningRate',
                              scheduler.get_lr()[0], train.global_step)
            
        with torch.cuda.device(0):
            image = image.type(torch.float).cuda(async=True)
            mask_gpu = mask.type(torch.float).cuda(async=True)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask_gpu)

        loss.backward()
        optimizer.step()
        
        train_metrics['loss'].append(loss.item())
        train_metrics['iou'].append(my_iou_metric(mask.numpy(), output.cpu().data.numpy()))

        if summary_writter:
            summary_writter.add_scalar('Train/RunningLoss', train_metrics['loss'][-1], train.global_step)
            summary_writter.add_scalar('Train/RunningIoU', train_metrics['iou'][-1], train.global_step)

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

    if summary_writter:
        summary_writter.add_scalar('Train/Loss', np.mean(train_metrics['loss']), epoch)
        summary_writter.add_scalar('Train/IoU', np.mean(train_metrics['iou']), epoch)
        summary_writter.add_scalar('Train/Time', elapsed, epoch)
        
    return train_metrics


def test(epoch, model, criterion, test_loader, summary_writter=None):
    logger.info('Test {}'.format(epoch))

    model.eval()

    val_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(test_loader):

        if summary_writter and epoch == 0 and step == 0:
            image_grid = torchvision.utils.make_grid(
                image, normalize=True, scale_each=True)
            summary_writter.add_image('Test/Image', image_grid, epoch)
        
        with torch.cuda.device(0):
            image = image.type(torch.float).cuda(async=True)
            mask_gpu = mask.type(torch.float).cuda(async=True)
        
        with torch.no_grad():
            outputs = model(image)
        loss = criterion(outputs, mask_gpu)

        val_metrics['loss'].append(loss.item())
        val_metrics['iou'].append(my_iou_metric(mask.numpy(), outputs.cpu().data.numpy()))

    message = (
        f"Epoch: {epoch},"
        f"Val: loss {np.mean(val_metrics['loss']):.3f}, IoU {np.mean(val_metrics['iou']):.3f} ** "
    )    
    logger.info(message)

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if summary_writter:
        summary_writter.add_scalar('Test/Loss', np.mean(val_metrics['loss']), epoch)
        summary_writter.add_scalar('Test/IoU', np.mean(val_metrics['iou']), epoch)
        summary_writter.add_scalar('Test/Time', elapsed, epoch)
    
    return val_metrics