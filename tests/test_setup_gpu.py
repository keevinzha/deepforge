# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/2 16:39
@Auth ： keevinzha
@File ：test_setup_gpu.py
@IDE ：PyCharm
"""
import pytest
import time

import torch

from src.utils.helpers import setup_gpu, load_config, convert_config_name2path


def test_setup_gpu():
    """测试 GPU 占用 60 秒"""
    config_path = convert_config_name2path('config')
    config = load_config(config_path)

    # 设置 GPU
    device = setup_gpu(config)

    if device is None:
        print("No GPU available")
        return



    print("\n" + "=" * 60)
    print("Using GPU 60 秒...")
    print("=" * 60 + "\n")

    # 在每个 GPU 上创建大张量
    tensors = []
    t = torch.randn(15000, 15000, device=device)
    tensors.append(t)

    start_time = time.time()
    iteration = 0

    while time.time() - start_time < 60:
        iteration += 1

        for i in range(len(tensors)):
            tensors[i] = torch.matmul(tensors[i], tensors[i].t())

        if iteration % 20 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Iteration {iteration}, 剩余: {60 - elapsed:.1f}s")

    print(f"finish")
    print("=" * 60)