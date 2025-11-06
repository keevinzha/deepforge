# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/29 13:53
@Auth ： keevinzha
@File ：main.py
@IDE ：PyCharm
"""
# train.py
"""
主训练脚本（使用模型和数据集注册系统）
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import time
import argparse

from src.core.engine import train_one_epoch, evaluate
from src.utils.checkpoint import (
    save_checkpoint,
    auto_resume,
    load_pretrained,
    has_checkpoint
)
from src.utils.helpers import (
    load_config,
    set_seed,
    create_model,
    create_optimizer,
    create_losses,
    cosine_scheduler,
    parse_args,
    create_dataset,
    setup_device,
)
from src.utils.checkpoint import auto_resume






def main():
    # load config
    args = parse_args()
    cfg = load_config(args.config)

    print("Training Configuration")
    print(f"Config file: {args.config}")
    print(f"Dataset: {cfg['data']['name']}")
    print(f"Epochs: {cfg['train']['epochs']}")
    print(f"Batch size: {cfg['train']['batch_size']}")
    print(f"Learning rate: {cfg['train']['lr']}")
    print(f"Optimizer: {cfg['train']['opt']}")

    # set random seed
    if 'seed' in cfg['train']:
        set_seed(cfg['train']['seed'])
    # setup device
    device = setup_device(cfg)

    # create model
    model = create_model(cfg)
    model = model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['name']}")
    print(f"Trainable parameters: {n_parameters:,}")

    # create dataset and dataloader
    enable_eval = not cfg['train']['disable_eval']
    train_set = create_dataset(cfg, is_train=True)
    val_set = create_dataset(cfg, is_train=False)

    # calculate batch size
    total_batch_size = cfg['data']['batch_size'] * cfg['train']['update_freq']
    num_training_steps_per_epoch = len(train_set) // total_batch_size

    # todo add sampler
    data_loader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_mem'],
        drop_last=True
    )

    if enable_eval:
        data_loader_val = torch.utils.data.DataLoader(
            val_set,
            batch_size=int(1.5*cfg['data']['batch_size']), # this magic number comes from https://github.com/facebookresearch/ConvNeXt.git
            num_workers=cfg['data']['num_workers'],
            pin_memory=cfg['data']['pin_mem'],
            drop_last=False
        )
    else:
        data_loader_val = None

    print(f"LR={cfg['train']['lr']}")
    print(f"Batch size = {total_batch_size}")
    print(f"Update frequent = {cfg['train']['update_freq']}")
    print(f"Number of training examples = {len(train_set)}")
    print(f"Number of training training per epoch = {num_training_steps_per_epoch}")

    # create optimizer
    optimizer = create_optimizer(cfg, model)

    # create lr schedular
    steps_per_epoch = len(data_loader_train)
    lr_schedule_values = cosine_scheduler(
        cfg['train']['lr'], cfg['train']['min_lr'], cfg['train']['epochs'], num_training_steps_per_epoch,
        warmup_epochs=cfg['train']['warmup_epochs'], warmup_steps=cfg['train']['warmup_steps']
    )

    if cfg['train']['weight_decay_end'] is None:
        cfg['train']['weight_decay_end'] = cfg['train']['weight_decay']
    wd_schedule_values = cosine_scheduler(
        cfg['train']['weight_decay'], cfg['train']['weight_decay_end'], cfg['train']['epochs'], num_training_steps_per_epoch,
    )

    # create loss function
    criterion = create_losses(cfg)



    # load checkpoint
    checkpoint_info = auto_resume(save_dir=cfg['train']['output_dir'], model=model, optimizer=optimizer,
                scheduler=lr_schedule_values, device=device)
    start_epoch = checkpoint_info['start_epoch']

    print("Starting Training")
    print(f"Start epoch: {start_epoch}")
    print(f"End epoch: {cfg['train']['epochs']}")
    print(f"Best metric so far: {checkpoint_info['best_metric']:.4f}")

    monitor_mode = cfg['train'].get('monitor_mode', 'max')

    start_time = time.time()

    best_metric = float('inf')

    for epoch in range(start_epoch, cfg['train']['epochs']):
        epoch_start_time = time.time()

        print(f"Epoch [{epoch}/{cfg['train']['epochs']}]")

        # training
        # todo add tensorboard
        train_stats = train_one_epoch(
            model=model,
            dataloader=data_loader_train,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=cfg['train']['update_freq']
        )
        if cfg['train']['output_dir'] and cfg['train']['save_ckpt']:
            if (epoch + 1) % cfg['train']['save_ckpt_freq'] == 0 or epoch + 1 == cfg['train']['epochs']:
                save_checkpoint(epoch=epoch, model=model, optimizer=optimizer,
                                cfg=cfg)

        # evaluate
        val_stats = None
        if not cfg['train']['disable_eval']:
            val_stats = evaluate(model=model, dataloader=data_loader_val, device=device,
                                 criterion=criterion)
            current_metric = val_stats.get('loss', float('inf'))

            if monitor_mode == 'max':
                is_best = current_metric > best_metric
            else:  # min
                is_best = current_metric < best_metric

            if is_best:
                best_metric = current_metric
                print(f"New best loss: {best_metric:.4f}")

            # save best
            if cfg['train']['ckpt'] and cfg['train']['output_dir'] and is_best:
                save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    is_best=True
                )

        # save frequently
        if (epoch + 1) % cfg['train'].get('save_ckpt_freq', 10) == 0:
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                best_metric=best_metric,
                is_best=False
            )

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch}] Summary:")

        train_metrics_str = ' | '.join([
            f"{k}: {v:.4f}" for k, v in train_stats.items()
        ])
        print(f"Train -> {train_metrics_str}")

        if val_stats:
            val_metrics_str = ' | '.join([
                f"{k}: {v:.4f}" for k, v in val_stats.items()
            ])
            print(f"Val -> {val_metrics_str}")

        print(f"Time: {epoch_time}")
        print(f"Best loss: {best_metric:.4f}")

    total_time = time.time() - start_time
    print("Training Completed!")
    print(f"Total time: {total_time}")
    print(f"Best loss: {best_metric:.4f}")
    print(f"Checkpoints saved to: {cfg['train']['save_dir']}")
    print("=" * 80)


if __name__ == '__main__':
    main()