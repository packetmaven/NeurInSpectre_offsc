"""Memory-augmented PGD using Volterra-weighted gradient history."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Attack
from .numerics import infer_data_range, clamp_to_range, check_grad_sanity, transformed_gradient


class MemoryAugmentedPGD(Attack):
    """
    PGD variant using Volterra kernel weighting of gradient history.
    alpha_volterra comes from NeurInSpectre Layer 2.
    """

    def __init__(
        self,
        model,
        alpha_volterra: float,
        memory_length: int = 10,
        eps: float = 0.031,
        alpha: float = 0.003,
        steps: int = 40,
        norm: str = "linf",
        kernel: str = "power_law",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.alpha_volterra = float(alpha_volterra)
        self.memory_length = int(memory_length)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.norm = norm
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        x_min, x_max = infer_data_range(x)

        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = self._project(delta, self.eps, self.norm)
        grad_history: list[torch.Tensor] = []

        for _ in range(self.steps):
            delta = delta.detach()
            delta.requires_grad_(True)

            logits = self.model(x + delta)
            loss = F.cross_entropy(logits, y)
            self.model.zero_grad(set_to_none=True)
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            current_grad = delta.grad.detach().clone()
            if not check_grad_sanity(current_grad, "MA-PGD"):
                current_grad = torch.sign(torch.randn_like(delta))

            grad_history.append(current_grad)
            if len(grad_history) > self.memory_length:
                grad_history.pop(0)

            if self.alpha_volterra < 0.8 and len(grad_history) > 1:
                weighted_grad = self._volterra_weighted_grad(grad_history)
            else:
                weighted_grad = current_grad

            weighted_grad = transformed_gradient(weighted_grad)

            with torch.no_grad():
                delta = delta + self.alpha * weighted_grad.sign()
                delta = self._project(delta, self.eps, self.norm)
                delta = clamp_to_range(x + delta, x_min, x_max) - x

        return (x + delta).detach()

    def _volterra_weighted_grad(self, grad_history: list[torch.Tensor]) -> torch.Tensor:
        k = len(grad_history)
        device = grad_history[0].device

        if self.kernel == "power_law":
            weights = torch.tensor([(i + 1) ** (-self.alpha_volterra) for i in range(k)], device=device)
        elif self.kernel == "exp":
            weights = torch.tensor([torch.exp(-self.alpha_volterra * i) for i in range(k)], device=device)
        else:  # uniform decay
            weights = torch.tensor([1.0 / (1 + self.alpha_volterra * i) for i in range(k)], device=device)

        weights = weights / weights.sum().clamp_min(1e-12)
        weighted_grad = torch.zeros_like(grad_history[0])
        for w, g in zip(weights, grad_history):
            weighted_grad = weighted_grad + w * g
        return weighted_grad
