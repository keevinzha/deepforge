# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/3 14:43
@Auth ： keevinzha
@File ：normalizer.py
@IDE ：PyCharm
"""

import numpy as np
import torch
from typing import Union, Tuple, Callable


class Normalizer:
    '''
    A class to normalize and denormalize data using different methods.
    Supports both NumPy arrays and PyTorch tensors.
    '''
    MIN_MAX = 'min_max'
    Z_SCORE = 'z_score'
    SUPPORTED_METHODS = [MIN_MAX, Z_SCORE]

    def __init__(self, method: str = MIN_MAX, normalize_fn: Callable = None, denormalize_fn: Callable = None):
        if method not in self.SUPPORTED_METHODS and not (normalize_fn and denormalize_fn):
            raise ValueError(f"Unsupported normalization method: {method}")

        self.method = method
        self.params = None
        self.normalize_fn = normalize_fn
        self.denormalize_fn = denormalize_fn

    def normalize(self, data: Union[np.ndarray, torch.Tensor], axis: Union[int, Tuple[int, ...]] = None) -> Union[
        np.ndarray, torch.Tensor]:
        '''
        Normalize the data using the provided normalizer.
        '''
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise TypeError("Input data must be a numpy array or torch tensor")

        is_torch = isinstance(data, torch.Tensor)

        if self.normalize_fn:
            normalized_data, self.params = self.normalize_fn(data, axis)
        elif self.method == self.MIN_MAX:
            if is_torch:
                if axis is None:
                    min_val = torch.min(data)
                    max_val = torch.max(data)
                else:
                    min_val = torch.min(data, dim=axis, keepdim=True).values
                    max_val = torch.max(data, dim=axis, keepdim=True).values
                self.params = (min_val, max_val)
                normalized_data = (data - min_val) / (max_val - min_val)
            else:
                if axis is None:
                    min_val = np.min(data)
                    max_val = np.max(data)
                else:
                    min_val = np.min(data, axis=axis, keepdims=True)
                    max_val = np.max(data, axis=axis, keepdims=True)
                self.params = (min_val, max_val)
                normalized_data = (data - min_val) / (max_val - min_val)
        elif self.method == self.Z_SCORE:
            if is_torch:
                if axis is None:
                    mean = torch.mean(data)
                    std = torch.std(data)
                else:
                    mean = torch.mean(data, dim=axis, keepdim=True)
                    std = torch.std(data, dim=axis, keepdim=True)
                self.params = (mean, std)
                normalized_data = (data - mean) / std
            else:
                if axis is None:
                    mean = np.mean(data)
                    std = np.std(data)
                else:
                    mean = np.mean(data, axis=axis, keepdims=True)
                    std = np.std(data, axis=axis, keepdims=True)
                self.params = (mean, std)
                normalized_data = (data - mean) / std
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

        return normalized_data

    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        '''
        Denormalize the data using the provided normalizer.

        This function denormalizes the data using the method and parameters provided during initialization.
        '''
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise TypeError("Input data must be a numpy array or torch tensor")

        is_torch = isinstance(data, torch.Tensor)

        if self.denormalize_fn:
            denormalized_data = self.denormalize_fn(data, self.params)
        elif self.method == self.MIN_MAX and self.params is not None:
            min_val, max_val = self.params
            denormalized_data = data * (max_val - min_val) + min_val
        elif self.method == self.Z_SCORE and self.params is not None:
            mean, std = self.params
            denormalized_data = data * std + mean
        else:
            raise ValueError(f"Unsupported denormalization method or missing parameters for {self.method}")

        return denormalized_data