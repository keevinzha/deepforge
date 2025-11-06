# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/1 21:56
@Auth ： keevinzha
@File ：test_registry.py
@IDE ：PyCharm
"""
import pytest
from nibabel.nicom.dicomwrappers import none_or_close

from src.utils.helpers import create_dataset, create_model, create_losses, convert_config_name2path, load_config


def test_dataset_register():
    config_path = convert_config_name2path('config')
    config = load_config(config_path)
    dataset = create_dataset(config)
    assert dataset is not None
    assert len(dataset) == 100


def test_model_register():
    config_path = convert_config_name2path('config')
    config = load_config(config_path)
    model = create_model(config)
    assert model is not None

def test_loss_register():
    config_path = convert_config_name2path('config')
    config = load_config(config_path)
    loss = create_losses(config)
    assert loss is not None