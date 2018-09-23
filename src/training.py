import logging
from collections import defaultdict
import time
import numpy as np
from metrics import my_iou_metric
import torch

logger = logging.getLogger(__name__)


def train(epoch, model, optimizer, scheduler, criterion, train_loader, config):
    logger.info('Train {}'.format(epoch))

    run_config = config['run_config']
    optim_config = config['optim_config']
    
    model.train()

    train_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(train_loader):

        if optim_config['scheduler'] == 'multistep':
            scheduler.step(epoch - 1)
        elif optim_config['scheduler'] == 'cosine':
            scheduler.step()
            
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
    return train_metrics


def test(epoch, model, criterion, test_loader):
    logger.info('Test {}'.format(epoch))

    model.eval()

    val_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(test_loader):
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
    return val_metrics