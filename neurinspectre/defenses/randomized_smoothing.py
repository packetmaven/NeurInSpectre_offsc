"""Randomized smoothing defense."""

from __future__ import annotations

import torch


def randomized_smoothing(x: torch.Tensor, sigma: float = 0.25) -> torch.Tensor:
    """
    Additive Gaussian noise as a randomized smoothing defense.
    """
    return x + torch.randn_like(x) * float(sigma)
