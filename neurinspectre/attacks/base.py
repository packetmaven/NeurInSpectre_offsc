"""Base class for adversarial attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .numerics import safe_flat_norm


class Attack(ABC):
    """
    Base class for adversarial attacks with support for different norms.

    All attacks inherit from this class and implement `forward()`.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the attack and return adversarial inputs."""
        raise NotImplementedError

    def __call__(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x, y, **kwargs)

    def _project(self, delta: torch.Tensor, eps: float, norm: str = "linf") -> torch.Tensor:
        if norm == "linf":
            return delta.clamp(-eps, eps)
        if norm == "l2":
            norms = safe_flat_norm(delta, p=2)  # [B, 1]
            factors = torch.min(eps / norms, torch.ones_like(norms))
            view_shape = (-1,) + (1,) * (delta.ndim - 1)
            return delta * factors.view(view_shape)
        raise ValueError(f"Unsupported norm: {norm}")
