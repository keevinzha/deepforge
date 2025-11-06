# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/3 17:47
@Auth ： keevinzha
@File ：test_build_dataset.py
@IDE ：PyCharm
"""
import pytest

from src.utils.helpers import create_dataset, load_config

def test_build_dataset():
    config = load_config('config')
    dataset = create_dataset(config, is_train=True)
    assert len(dataset) == 50000
    assert dataset is not None
