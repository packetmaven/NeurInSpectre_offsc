"""Hybrid BPDA + EOT attack."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .base import Attack
from .eot import EOT


class HybridBPDAEOT(Attack):
    """
    Hybrid attack combining BPDA (for shattered gradients) and EOT (stochasticity).
    """

    def __init__(
        self,
        model: nn.Module,
        defense,
        *,
        approx_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_samples: int = 20,
        importance_sampling: bool = True,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 40,
        norm: str = "linf",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.defense = defense
        self.approx_fn = approx_fn or defense.get_bpda_approximation()
        self.num_samples = int(num_samples)
        self.importance_sampling = bool(importance_sampling)

        self.eot = EOT(
            model,
            transform_fn=self._bpda_transform,
            num_samples=self.num_samples,
            importance_sampling=self.importance_sampling,
            eps=eps,
            alpha=alpha,
            steps=steps,
            norm=norm,
            device=device,
        )

    def _bpda_transform(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_actual = self.defense.transform(x)
        x_approx = self.approx_fn(x)
        return x_actual + (x_approx - x_approx.detach())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.eot(x, y)
