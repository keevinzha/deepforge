# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/27 17:48
@Auth ： keevinzha
@File ：simple_dataset.py
@IDE ：PyCharm
"""
import numpy as np

import torch
from torchvision import datasets

from . import register_dataset
from .base_dataset import BaseDataset

@register_dataset('simple')
class SimpleDataset(BaseDataset):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train, normalizer=None)
        self.cifar_dataset = datasets.CIFAR10(
            root=self.root,
            train=is_train,
            download=False,  # 不自动下载，使用已有数据
            transform=None  # 先不做变换，获取原始数据
        )


    def __len__(self):
        return len(self.cifar_dataset)


    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return image, label