# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/27 17:32
@Auth ： keevinzha
@File ：mlp_model.py
@IDE ：PyCharm
"""

import torch.nn as nn

from . import register_model

@register_model('mlp')
class MLPModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=256, output_size=10):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x