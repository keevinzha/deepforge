# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/29 15:48
@Auth ： keevinzha
@File ：helpers.py
@IDE ：PyCharm
"""

import os
import yaml
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import math
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..models import build_model, list_models
from ..data import build_dataset, list_datasets
from ..utils.losses import build_loss, list_losses
from ..utils.optim_factory import create_optimizer as fac_create_optimizer


def load_config(config_name: str) -> Dict[str, Any]:
    """
    yaml config loader
    :param config_path:
    :return config dict
    """

    config_path = convert_config_name2path(config_name)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path:str):
    """
    save config to yaml
    :param config: config dict
    :param save_path:
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {save_path}")


def set_seed(seed: int, deterministic: bool=False):
    """
    set random seed
    :param seed: random seed
    :param deterministic: if using deterministic algorithm
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Using deterministic mode (may reduce performance)")
    else:
        torch.backends.cudnn.benchmark = True
    print(f"Random seed set to {seed}")


def create_model(config: dict) -> nn.Module:
    model_name = config['model']['name']
    input_size = config['data']['input_size']
    return build_model(model_name, input_size)


def create_dataset(config: dict, is_train: bool) -> Dataset:

    return build_dataset(config, is_train)


def create_optimizer(config: Dict[str, Any], model) -> torch.optim.Optimizer:
    opt = fac_create_optimizer(config, model)
    return opt


def create_losses(config:dict) -> nn.Module:
    loss_name = config['train']['loss']
    return build_loss(loss_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', type=str, help='config file name')
    args = parser.parse_args()
    return args


def convert_config_name2path(config_name):
    script_dir = Path(__file__).parent.parent.parent.resolve()
    config_dir = script_dir / 'configs'


    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"

    config_path = config_dir / config_name

    if not config_path.exists():
        available = [f.stem for f in config_dir.glob('*.yaml')] if config_dir.exists() else []
        raise argparse.ArgumentTypeError(
            f"Config file not exist: {config_path}\n"
            f"Available config files: {', '.join(available) if available else 'No config file'}"
        )

    return config_path


def setup_device(config):
    if 'gpu' not in config['device'] or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("GPU unset, using default config")
        return device

    gpu_ids = config['device']['gpu']

    if isinstance(gpu_ids, int):
        device = torch.device(f'cuda:{gpu_ids}')
        torch.cuda.set_device(gpu_ids)
        print(f"Using GPU: {gpu_ids}")

    elif isinstance(gpu_ids, list):
        # todo multi gpu
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(gpu_ids[0])
        print(f"Using multiple GPUs: {gpu_ids}, main GPU: {gpu_ids[0]}")

    return device


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule