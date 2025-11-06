# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/28 16:11
@Auth ： keevinzha
@File ：engine.py
@IDE ：PyCharm
"""
import math
import time
from typing import Dict

import torch

from ..utils.loggers import MetricLogger, SmoothedValue


def train_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        log_writer=None,
        start_steps=None,
        lr_schedule_values=None,
        wd_schedule_values=None,
        num_training_steps_per_epoch=None,
        update_freq=None,
) -> Dict[str, float]:
    """
    basic training function for one epoch in training loop
    Args:
    :param model: nn model
    :param dataloader: dataloader
    :param criterion: loss function
    :param optimizer: optimizer
    :param device: device (cpu or gpu)
    :param epoch:

    Returns:
    a dict contains metrics
    """
    model.train()

    metric_logger = MetricLogger(delimiter="   ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    optimizer.zero_grad()

    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        step = batch_idx // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        if lr_schedule_values is not None or wd_schedule_values is not None and batch_idx % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


        outputs = model(samples)
        loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            assert math.isfinite(loss_value)

        loss /= update_freq
        loss.backward()
        if (batch_idx + 1) % update_freq == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10
        max_lr = 0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device:torch.device
) -> Dict[str, float]:
    model.eval()

    metric_logger = MetricLogger(delimiter="   ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(dataloader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print(f'* Loss {metric_logger.loss.global_avg:.4f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



