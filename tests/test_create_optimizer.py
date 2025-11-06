# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/2 21:12
@Auth ： keevinzha
@File ：test_create_optimizer.py
@IDE ：PyCharm
"""
import torch

from src.utils.helpers import setup_gpu, load_config, convert_config_name2path, create_model
from src.utils.optim_factory import create_optimizer


def test_create_optimizer():
    config_path = convert_config_name2path('config')
    config = load_config(config_path)
    model = create_model(config)
    opt = create_optimizer(config, model)
    print(type(opt))