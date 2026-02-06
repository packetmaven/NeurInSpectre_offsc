"""Projected Gradient Descent (PGD) baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Attack
from .numerics import infer_data_range, clamp_to_range, check_grad_sanity


class PGD(Attack):
    """
    Projected Gradient Descent (PGD) attack.

    Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018).
    """

    def __init__(
        self,
        model,
        eps: float = 0.031,
        alpha: float = 0.003,
        steps: int = 40,
        norm: str = "linf",
        random_start: bool = True,
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.norm = norm
        self.random_start = bool(random_start)
        if self.steps < 10:
            raise ValueError("PGD requires steps >= 10 for stable convergence.")

    def forward(self, x: torch.Tensor, y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        x_min, x_max = infer_data_range(x)

        delta = torch.zeros_like(x)
        if self.random_start:
            delta.uniform_(-self.eps, self.eps)
            delta = self._project(delta, self.eps, self.norm)

        for step in range(self.steps):
            delta = delta.detach()
            delta.requires_grad_(True)

            logits = self.model(x + delta)
            loss = F.cross_entropy(logits, y)
            if targeted:
                loss = -loss

            self.model.zero_grad(set_to_none=True)
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            grad = delta.grad

            if not check_grad_sanity(grad, f"PGD-step{step}"):
                grad = torch.sign(torch.randn_like(delta)) * 1e-3

            with torch.no_grad():
                delta = delta + self.alpha * grad.sign()
                delta = self._project(delta, self.eps, self.norm)
                delta = clamp_to_range(x + delta, x_min, x_max) - x

        return (x + delta).detach()


class PGDWithRestarts(Attack):
    """
    PGD with random restarts for improved attack success rates.
    """

    def __init__(
        self,
        model,
        n_restarts: int = 10,
        eps: float = 0.031,
        alpha: float = 0.003,
        steps: int = 40,
        norm: str = "linf",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.n_restarts = int(n_restarts)
        self.base_attack = PGD(
            model,
            eps=eps,
            alpha=alpha,
            steps=steps,
            norm=norm,
            random_start=True,
            device=device,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        best_adv = x.clone()
        best_loss = torch.full((x.size(0),), -1e10, device=self.device)

        for _ in range(self.n_restarts):
            x_adv = self.base_attack(x, y)
            with torch.no_grad():
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y, reduction="none")
                improved = loss > best_loss
                if improved.any():
                    best_adv[improved] = x_adv[improved]
                    best_loss[improved] = loss[improved]

        return best_adv
