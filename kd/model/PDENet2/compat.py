# compat.py
from __future__ import annotations

import contextlib

import torch


def get_device(device: str | None = None) -> torch.device:
    """
    device: e.g. 'cuda:0', 'cpu', None
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@contextlib.contextmanager
def inference_mode():
    """
    PyTorch 1.13: inference_mode is faster & safer than no_grad for inference.
    """
    with torch.inference_mode():
        yield


def to_device(batch, device: torch.device):
    """
    Recursively move tensors in (list/tuple/dict) to device.
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        t = [to_device(x, device) for x in batch]
        return type(batch)(t)
    return batch


def save_checkpoint(path: str, obj):
    # New zipfile serialization is default; keeping default is fine.
    torch.save(obj, path)


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)
