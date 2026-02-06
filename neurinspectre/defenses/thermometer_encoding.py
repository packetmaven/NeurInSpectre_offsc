"""Thermometer encoding defense."""

from __future__ import annotations

import torch


def thermometer_defense(x: torch.Tensor, levels: int = 16) -> torch.Tensor:
    """
    Simple quantization-based thermometer encoding approximation.
    """
    levels = int(levels)
    return torch.round(x * levels) / levels
