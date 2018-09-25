import logging
from collections import defaultdict
import time
import numpy as np
from metrics import my_iou_metric
import torch
import torchvision

logger = logging.getLogger(__name__)

def create_image_writer(summary_writer):
    def write_to(data_tensor, label, epoch, normalize=False):
        image_grid = torchvision.utils.make_grid(
            data_tensor, normalize=normalize, scale_each=True)
        summary_writer.add_image(label, image_grid, epoch)
    return write_to
    

def train(epoch, model, optimizer, scheduler, criterion, train_loader, config, summary_writer=None, global_counter=None):
    logger.info('Train {}'.format(epoch))

    run_config = config['run_config']
    optim_config = config['optim_config']

    image_writer = create_image_writer(summary_writer)
    model.train()

    train_metrics = defaultdict(list)
    start = time.time()
    for step, (image, mask) in enumerate(train_loader):
        global_step = next(global_counter) if global_counter is not None else step
       
        if summary_writer is not None and step == 0:
            image_writer(image, 'Train/Image', epoch, normalize=True)

        if optim_config['scheduler'] == 'multistep':
            scheduler.step(epoch - 1)
        elif optim_config['scheduler'] == 'cosine':
            scheduler.step()

        if summary_writer is not None:
            summary_writer.add_scalar('Train/LearningRate',
                                      scheduler.get_lr()[0], global_step)
            
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
        train_metrics['iou'].append(my_iou_metric(mask.numpy(), output_cpu.data.numpy()))
        
        if summary_writer is not None and step == 0:
            image_writer(mask, 'Train/Mask', epoch)
            image_writer(output_cpu, 'Train/Prediction', epoch)

        if summary_writer:
            summary_writer.add_scalar('Train/RunningLoss', train_metrics['loss'][-1], global_step)
            summary_writer.add_scalar('Train/RunningIoU', train_metrics['iou'][-1], global_step)

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


def test(epoch, model, criterion, test_loader, summary_writer=None):
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
        val_metrics['iou'].append(my_iou_metric(mask.numpy(), output_cpu.data.numpy()))

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