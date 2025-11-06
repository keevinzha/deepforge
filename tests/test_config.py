# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/2 14:31
@Auth ： keevinzha
@File ：test_config.py
@IDE ：PyCharm
"""
import pytest
from src.utils.helpers import load_config, convert_config_name2path

def test_load_config():
    config_path = convert_config_name2path('config')
    # this is a dict
    config = load_config(config_path)
    print(config)
    print(config.__class__)