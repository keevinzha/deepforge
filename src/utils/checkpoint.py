# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/29 14:02
@Auth ： keevinzha
@File ：checkpoint.py
@IDE ：PyCharm
"""

import os
import glob
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    is_best: bool = False,
    **kwargs
) -> str:
    save_dir = Path(cfg['train']['output_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': cfg,
    }

    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()

    checkpoint.update(kwargs)

    epoch_name = str(epoch)
    checkpoint_path = save_dir / f'checkpoint-{epoch_name}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    if is_best:
        best_path = save_dir / 'checkpoint-best.pth'
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")

    if isinstance(epoch, int) and cfg.get('checkpoint', {}).get('keep_last_n', 0) > 0:
        cleanup_old_checkpoints(
            save_dir,
            keep_last_n=cfg['checkpoint']['keep_last_n'],
            current_epoch=epoch
        )

    return str(checkpoint_path)


def cleanup_old_checkpoints(save_dir: Path, keep_last_n: int, current_epoch: int):
    all_checkpoints = glob.glob(str(save_dir / 'checkpoint-*.pth'))

    epoch_checkpoints = []
    for ckpt in all_checkpoints:
        epoch_str = ckpt.split('-')[-1].split('.')[0]
        if epoch_str.isdigit():
            epoch_checkpoints.append((int(epoch_str), ckpt))

    epoch_checkpoints.sort(key=lambda x: x[0])

    if len(epoch_checkpoints) > keep_last_n:
        to_delete = epoch_checkpoints[:-keep_last_n]
        for epoch_num, ckpt_path in to_delete:
            if epoch_num < current_epoch:
                try:
                    os.remove(ckpt_path)
                    print(f"Removed old checkpoint: {ckpt_path}")
                except OSError as e:
                    print(f"Failed to remove {ckpt_path}: {e}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = True,
    load_optimizer: bool = True,
    device: str = 'cpu'
) -> Dict[str, Any]:
    if checkpoint_path.startswith('https'):
        print(f"Loading checkpoint from URL: {checkpoint_path}")
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path,
            map_location=device,
            check_hash=True
        )
    else:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    else:
        # in case of save state_dict as checkpoint
        model_state_dict = checkpoint

    load_state_dict(model, model_state_dict, strict=strict)

    start_epoch = 0
    if load_optimizer and optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded optimizer state")

    # load epoch information
    if 'epoch' in checkpoint:
        if isinstance(checkpoint['epoch'], int):
            start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint from epoch {checkpoint['epoch']}")

    # load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Loaded scheduler state")

    # load scaler state
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        print("Loaded AMP scaler state")

    info = {
        'start_epoch': start_epoch,
        'best_metric': checkpoint.get('best_metric', 0.0),
        'config': checkpoint.get('config', None),
    }

    for key in checkpoint:
        if key not in ['model', 'optimizer', 'scheduler', 'scaler', 'epoch', 'best_metric', 'config']:
            info[key] = checkpoint[key]

    return info


def load_pretrained(
        model: torch.nn.Module,
        pretrained_path: str,
        model_key: str = 'model|model_ema|state_dict',
        remove_prefix: str = '',
        strict: bool = False
) -> None:
    """
    加载预训练模型，支持多种格式
    参考ConvNeXt的finetune加载逻辑

    Args:
        model: 模型
        pretrained_path: 预训练模型路径或URL
        model_key: state_dict的key，用|分隔多个候选
        remove_prefix: 要移除的key前缀
        strict: 是否严格匹配

    Examples:
        # 从本地文件加载
        load_pretrained(model, './pretrained/resnet50.pth')

        # 从URL加载
        load_pretrained(model, 'https://download.pytorch.org/models/resnet50.pth')

        # 指定key
        load_pretrained(model, './checkpoint.pth', model_key='model_ema')
    """
    print(f"Loading pretrained model from: {pretrained_path}")

    # 加载checkpoint
    if pretrained_path.startswith('https') or pretrained_path.startswith('http'):
        print("  Downloading from URL...")
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained_path,
            map_location='cpu',
            check_hash=True
        )
    else:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')

    # 尝试不同的key
    state_dict = None
    used_key = None

    for key in model_key.split('|'):
        if key in checkpoint:
            state_dict = checkpoint[key]
            used_key = key
            print(f"✓ Found state_dict with key: '{key}'")
            break

    if state_dict is None:
        # 假设整个checkpoint就是state_dict
        state_dict = checkpoint
        used_key = 'root'
        print("✓ Using entire checkpoint as state_dict")

    # 移除前缀
    if remove_prefix:
        print(f"  Removing prefix: '{remove_prefix}'")
        state_dict = {
            k.replace(remove_prefix, ''): v
            for k, v in state_dict.items()
        }

    # 处理分类头不匹配的情况
    model_state = model.state_dict()
    keys_to_remove = []

    # 常见的分类头key
    classifier_keys = [
        'head.weight', 'head.bias',
        'fc.weight', 'fc.bias',
        'classifier.weight', 'classifier.bias',
        'classifier.1.weight', 'classifier.1.bias',  # 有些模型的classifier是Sequential
    ]

    for key in classifier_keys:
        if key in state_dict and key in model_state:
            if state_dict[key].shape != model_state[key].shape:
                print(f"✓ Removing key '{key}' (shape mismatch: "
                      f"{state_dict[key].shape} vs {model_state[key].shape})")
                keys_to_remove.append(key)

    # 移除不匹配的key
    for key in keys_to_remove:
        del state_dict[key]

    # 加载权重
    load_state_dict(model, state_dict, strict=strict)
    print("✓ Pretrained model loaded successfully")


def load_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
    prefix: str = '',
    ignore_missing: str = "relative_position_index",
    strict: bool = True
) -> None:
    if prefix:
         state_dict = {prefix + k: v for k, v in state_dict.items()}

    # Remove `module.` prefix when loading a checkpoint saved from DataParallel/DistributedDataParallel into an unwrapped (single-GPU/CPU) model.
    if list(state_dict.keys())[0].startswith('module.') and not hasattr(model, 'module'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    if hasattr(model, 'module') and not list(state_dict.keys())[0].startswith('module.'):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        should_ignore = False
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                should_ignore = True
                break
        if should_ignore:
            ignore_missing_keys.append(key)
        else:
            warn_missing_keys.append(key)

    if len(warn_missing_keys) > 0:
        print(f"Weights not loaded from checkpoint: {warn_missing_keys}")

    if len(unexpected_keys) > 0:
        print(f"Weights in checkpoint but not used: {unexpected_keys}")

    if len(ignore_missing_keys) > 0:
        print(f"Ignored missing keys: {ignore_missing_keys}")

    if strict and (len(warn_missing_keys) > 0 or len(unexpected_keys) > 0):
        raise RuntimeError(
            f"Error loading state_dict: "
            f"Missing keys: {warn_missing_keys}, "
            f"Unexpected keys: {unexpected_keys}"
        )

    if len(warn_missing_keys) == 0 and len(unexpected_keys) == 0:
        print("All model weights loaded successfully")


def auto_resume(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    save_dir = Path(save_dir)

    if not save_dir.exists():
        print(f"Checkpoint directory {save_dir} does not exist")
        return {'start_epoch': 0, 'best_metric': 0.0}

    all_checkpoints = glob.glob(str(save_dir / 'checkpoint-*.pth'))

    if not all_checkpoints:
        print(f"No checkpoint found in {save_dir}")
        return {'start_epoch': 0, 'best_metric': 0.0}

    latest_epoch = -1
    latest_checkpoint = None

    for ckpt_path in all_checkpoints:
        epoch_str = ckpt_path.split('-')[-1].split('.')[0]
        if epoch_str.isdigit():
            epoch_num = int(epoch_str)
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint = ckpt_path

    if latest_checkpoint is None:
        print(f"No valid checkpoint found in {save_dir}")
        return {'start_epoch': 0, 'best_metric': 0.0}

    print(f"Auto resume from: {latest_checkpoint}")

    info = load_checkpoint(
        checkpoint_path=latest_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device
    )

    return info


def get_latest_checkpoint(save_dir: str) -> Optional[str]:
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None

    all_checkpoints = glob.glob(str(save_dir / 'checkpoint-*.pth'))
    if not all_checkpoints:
        return None

    latest_epoch = -1
    latest_checkpoint = None

    for ckpt_path in all_checkpoints:
        epoch_str = ckpt_path.split('-')[-1].split('.')[0]
        if epoch_str.isdigit():
            epoch_num = int(epoch_str)
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint = ckpt_path

    return latest_checkpoint


def has_checkpoint(save_dir: str) -> bool:
    return get_latest_checkpoint(save_dir) is not None