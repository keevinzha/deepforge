# -*- coding: utf-8 -*-
"""
@Time ： 2025/11/3 17:59
@Auth ： keevinzha
@File ：base_dataset.py.py
@IDE ：PyCharm
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from src.utils.normalizer import Normalizer

class BaseDataset(Dataset, ABC):
    """
    Abstract base class for datasets.

    To create a subclass, implement the following methods:
    -- __init__: Initialize the class, first call BaseDataset.__init__(self, opt).
    -- __len__: Return the size of the dataset.
    -- __getitem__: Get a data point.
    -- modify_commandline_options: (optional) Add dataset-specific options and set default options.
    """

    def __init__(self, cfg: Dict, is_train: bool, normalizer: Union[Normalizer, None] = None):
        """
        Initialize the BaseDataset class.

        :param opt: Option class that stores all the experiment flags; needs to be a subclass of BaseOptions.
        :param is_train: Whether in training or evaluation mode.
        :param normalizer: Normalizer instance for data normalization and denormalization.
        """
        if is_train :
            if 'train_path' not in cfg['data']:
                raise ValueError("opt must have 'data_path' and 'eval_data_path' attributes")
        else:
            if not cfg['train'].get('disable_eval', False):
                if 'val_path' not in cfg['data']:
                    raise ValueError("cfg['data'] must have 'val_path' when disable_eval=False")

        self.cfg = cfg
        self.root = cfg['data']['train_path'] if is_train else cfg['data']['val_path']
        self.is_train = is_train
        self.normalizer = normalizer


    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of images in the dataset.

        :return: The number of images in the dataset.
        """
        pass


    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return a data point and its metadata information.

        :param index: A random integer for data indexing.
        :return: A dictionary of data with their names.
        """
        pass

    def apply_normalization(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize the data using the provided normalizer.
        No normalization is applied by default.
        :param data: Data to be normalized.
        :return: Normalized data.
        """

        return data


    def apply_denormalization(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize the data using the provided normalizer.
        No denormalization is applied by default.
        :param data: Data to be denormalized.
        :return: Denormalized data.
        """

        return data

    def collate_fn(self, batch):
        return default_collate(batch)