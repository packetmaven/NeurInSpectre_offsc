"""Input transformation defenses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_resize_pad(x: torch.Tensor, size: tuple[int, int] = (32, 32)) -> torch.Tensor:
    """
    Random resize + pad (simple stochastic transform).
    """
    resized = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return resized


def random_crop(x: torch.Tensor, crop_size: tuple[int, int] = (32, 32)) -> torch.Tensor:
    """
    Random crop (assumes input spatial dims >= crop size).
    """
    h, w = x.shape[-2:]
    ch, cw = crop_size
    if h <= ch or w <= cw:
        return x
    top = torch.randint(0, h - ch + 1, (1,), device=x.device).item()
    left = torch.randint(0, w - cw + 1, (1,), device=x.device).item()
    return x[..., top : top + ch, left : left + cw]
