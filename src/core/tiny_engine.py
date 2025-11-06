# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/1 22:06
@Auth ： keevinzha
@File ：tiny_engine.py
@IDE ：PyCharm
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision import transforms

from src.utils.helpers import (
    load_config,
    create_model,
    create_dataset,
    parse_args,
    convert_config_name2path,
    setup_gpu
)



def main():
    # load config
    args = parse_args()
    config_path = convert_config_name2path(args.config)
    config = load_config(config_path)

    device = setup_gpu(config)
    model = create_model(config['model']['name'])
    dataset = create_dataset(config['data']['name'])



    dataloader = DataLoader(dataset, batch_size=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 训练循环
    for epoch in range(2):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")
            break  # 只跑一个batch测试
        break

if __name__ == '__main__':
    main()