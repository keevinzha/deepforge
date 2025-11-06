# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/27 17:39
@Auth ： keevinzha
@File ：__init__.py
@IDE ：PyCharm
"""

import os
import importlib
import pkgutil


DATASET_REGISTRY = {} # dataset dict for dataset register


def register_dataset(name):
    """
    dataset register decorator

    usage：
        @register_dataset('medical')
        class MedicalDataset(Dataset):
            ...

    Args:
        name: registered name
    """

    def decorator(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{name}' is already registered!\n"
                f"Please use a different name."
            )
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def build_dataset(config, is_train, **params):
    """
    build dataset from registered dataset dict

    Args:
        name: register name of the dataset
        **params: (optional) dataset params

    Returns:
        dataset instance
    """
    name = config['data']['name']
    if name not in DATASET_REGISTRY:
        available = ', '.join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{name}'\n"
            f"Available datasets: {available}\n"
            f"Did you forget to use @register_dataset decorator?"
        )

    dataset_class = DATASET_REGISTRY[name]
    return dataset_class(config, is_train, **params)


def list_datasets():
    """
    list all registered datasets

    Returns:
        list of registered dataset names
    """
    return list(DATASET_REGISTRY.keys())


def _auto_import_datasets():
    """
    automatically import all dataset modules in src/data,
    and trigger the @register_model decorator
    """
    current_dir = os.path.dirname(__file__)

    # traverse all modules in the current directory
    for finder, name, ispkg in pkgutil.iter_modules([current_dir]):
        # skip packages
        if ispkg:
            continue

        # skip private modules
        if name.startswith('_'):
            continue

        # import module
        try:
            importlib.import_module(f'.{name}', package=__package__)
        except Exception as e:
            err_msg = f"Failed to import dataset module '{name}': {e}"
            raise ImportError(err_msg)


_auto_import_datasets()


__all__ = [
    'build_dataset',
    'register_dataset',
    'list_datasets',
    'DATASET_REGISTRY',
]