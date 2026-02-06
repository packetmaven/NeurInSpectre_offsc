"""Temporal momentum PGD (momentum over gradient history)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Attack
from .numerics import infer_data_range, clamp_to_range, check_grad_sanity


class TemporalMomentumPGD(Attack):
    """
    PGD variant with temporal momentum over gradients (MI-FGSM style).
    """

    def __init__(
        self,
        model,
        eps: float = 0.031,
        alpha: float = 0.003,
        steps: int = 40,
        norm: str = "linf",
        momentum: float = 0.9,
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.norm = norm
        self.momentum = float(momentum)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        x_min, x_max = infer_data_range(x)

        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = self._project(delta, self.eps, self.norm)
        velocity = torch.zeros_like(delta)

        for _ in range(self.steps):
            delta = delta.detach()
            delta.requires_grad_(True)

            logits = self.model(x + delta)
            loss = F.cross_entropy(logits, y)
            self.model.zero_grad(set_to_none=True)
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            grad = delta.grad

            if not check_grad_sanity(grad, "TemporalMomentumPGD"):
                grad = torch.sign(torch.randn_like(delta))

            grad_normed = grad / (
                grad.abs().mean(dim=list(range(1, grad.ndim)), keepdim=True).clamp_min(1e-12)
            )
            velocity = self.momentum * velocity + grad_normed

            with torch.no_grad():
                delta = delta + self.alpha * velocity.sign()
                delta = self._project(delta, self.eps, self.norm)
                delta = clamp_to_range(x + delta, x_min, x_max) - x

        return (x + delta).detach()
