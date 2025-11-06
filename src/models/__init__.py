# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/27 15:57
@Auth ： keevinzha
@File ：__init__.py
@IDE ：PyCharm
"""

import os
import importlib
import pkgutil


MODEL_REGISTRY = {} # model dict for model register


def register_model(name):
    """
    model register decorator

    usage:
        @register_model('resnet')
        class ResNet(nn.Module):
            ...

    Args:
        name: registered name
    """

    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered!\n"
                f"Please use a different name."
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name, input_size, **params):
    """
    build model from registered model dict

    Args:
        name: register name of the model
        **params: (optional) model params

    Returns:
        model instance
    """
    if name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{name}'\n"
            f"Available models: {available}\n"
            f"Did you forget to use @register_model decorator?"
        )

    model_class = MODEL_REGISTRY[name]
    return model_class(input_size, **params)


def list_models():
    """
    list all the registered models

    Returns:
        list of registered model names
    """
    return list(MODEL_REGISTRY.keys())


def _auto_import_models():
    """
    automatically import all models in src/models,
    and trigger the @register_model decorator
    """
    current_dir = os.path.dirname(__file__)

    # traverse all modules in the current directory
    for finder, name, ispkg in pkgutil.iter_modules([current_dir]):
        # skip packages
        if ispkg:
            continue

        # ignore private modules, including __init__.py
        if name.startswith('_'):
            continue

        # import module
        try:
            importlib.import_module(f'.{name}', package=__package__)
        except Exception as e:
            err_msg = f"Error importing module {name}: {e}"
            raise ImportError(err_msg)


_auto_import_models()


__all__ = [
    'build_model',
    'register_model',
    'list_models',
    'MODEL_REGISTRY',
]