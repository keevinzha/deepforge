# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/27 15:40
@Auth ： keevinzha
@File ：losses.py.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


LOSS_REGISTRY = {}


def register_loss(name):
    """
    Loss register decorator

    Usage:
        @register_loss('mse')
        class MSELoss(nn.Module):
            ...
    """

    def decorator(cls):
        if name in LOSS_REGISTRY:
            raise ValueError(
                f"Loss '{name}' is already registered!\n"
                f"Please use a different name."
            )
        LOSS_REGISTRY[name] = cls
        return cls

    return decorator


def build_loss(name, **params):
    """
        build model from registered loss dict

        Args:
            name: register name of the loss
            **params: (optional) model params

        Returns:
            loss instance
        """
    if name not in LOSS_REGISTRY:
        available = ', '.join(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown loss: '{name}'\n"
            f"Available losses: {available}\n"
            f"Did you forget to use @register_loss decorator?"
        )

    loss_class = LOSS_REGISTRY[name]
    return loss_class(**params)


def list_losses():
    return list(LOSS_REGISTRY.keys())


@register_loss('mse')
class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        return F.mse_loss(pred, target, reduction=self.reduction)


@register_loss('mae')
class MAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        return F.l1_loss(pred, target, reduction=self.reduction)