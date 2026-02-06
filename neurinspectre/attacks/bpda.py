"""Backward Pass Differentiable Approximation (BPDA) attack."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Attack
from .numerics import infer_data_range, clamp_to_range, check_grad_sanity
from .bpda_registry import BPDA_REGISTRY, apply_bpda


class BPDA(Attack):
    """
    Backward Pass Differentiable Approximation (BPDA) attack.

    For defenses with non-differentiable components g(x), BPDA computes
    gradients through a differentiable approximation g̃(x).
    """

    def __init__(
        self,
        model: nn.Module,
        defense: Callable,
        approx_name: str = "identity",
        approx_fn: Optional[Callable] = None,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 40,
        norm: str = "linf",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.defense = defense
        if approx_fn is not None:
            self.approx_fn = approx_fn
        elif approx_name in BPDA_REGISTRY:
            self.approx_fn = BPDA_REGISTRY[approx_name]
        else:
            raise ValueError(f"Unknown approximation: {approx_name}. Available: {list(BPDA_REGISTRY.keys())}")

        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.norm = norm

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        x_min, x_max = infer_data_range(x)

        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = self._project(delta, self.eps, self.norm)
        delta.requires_grad = True

        for step in range(self.steps):
            x_adv = x + delta
            x_bpda = apply_bpda(x_adv, self.defense, self.approx_fn)

            logits = self.model(x_bpda)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            grad = delta.grad

            if not check_grad_sanity(grad, f"BPDA-step{step}"):
                grad = torch.sign(torch.randn_like(delta)) * 1e-3

            with torch.no_grad():
                delta.data = delta + self.alpha * grad.sign()
                delta.data = self._project(delta, self.eps, self.norm)
                delta.data = clamp_to_range(x + delta.data, x_min, x_max) - x

            if delta.grad is not None:
                delta.grad.zero_()

        return (x + delta).detach()


class LearnedBPDA(Attack):
    """
    BPDA with learned neural network approximation.
    """

    def __init__(
        self,
        model: nn.Module,
        defense: Callable,
        approx_network: Optional[nn.Module] = None,
        lambda_jacobian: float = 0.01,
        train_steps: int = 1000,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 40,
        norm: str = "linf",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.defense = defense
        self.lambda_jacobian = float(lambda_jacobian)
        self.train_steps = int(train_steps)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.norm = norm

        if approx_network is None:
            self.approx_network = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1),
            ).to(device)
        else:
            self.approx_network = approx_network.to(device)

        self.optimizer = torch.optim.Adam(self.approx_network.parameters(), lr=1e-3)

    def train_approximation(self, train_loader: torch.utils.data.DataLoader):
        self.approx_network.train()

        for step, (x, _) in enumerate(train_loader):
            if step >= self.train_steps:
                break

            x = x.to(self.device)
            with torch.no_grad():
                x_defended = self.defense(x)

            x_approx = self.approx_network(x)
            recon_loss = F.mse_loss(x_approx, x_defended)

            x = x.detach().requires_grad_(True)
            x_approx_grad = self.approx_network(x)

            jacobian_norm = 0.0
            for i in range(x_approx_grad.size(1)):
                grad_i = torch.autograd.grad(
                    x_approx_grad[:, i].sum(), x, create_graph=True, retain_graph=True
                )[0]
                jacobian_norm += grad_i.pow(2).sum()

            jacobian_loss = self.lambda_jacobian * jacobian_norm
            loss = recon_loss + jacobian_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 100 == 0:
                print(
                    f"[Learned BPDA] Step {step}/{self.train_steps}: "
                    f"Recon={recon_loss.item():.4f}, Jac={jacobian_loss.item():.4f}"
                )

        self.approx_network.eval()
        print("[Learned BPDA] Approximation training complete")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        x_min, x_max = infer_data_range(x)

        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta.requires_grad = True

        for step in range(self.steps):
            x_adv = x + delta
            x_defended = self.defense(x_adv.detach())
            x_approx = self.approx_network(x_adv)
            x_bpda = x_defended + (x_approx - x_approx.detach())

            logits = self.model(x_bpda)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            grad = delta.grad
            if not check_grad_sanity(grad, f"LearnedBPDA-step{step}"):
                grad = torch.sign(torch.randn_like(delta)) * 1e-3

            with torch.no_grad():
                delta.data = delta + self.alpha * grad.sign()
                delta.data = self._project(delta, self.eps, self.norm)
                delta.data = clamp_to_range(x + delta.data, x_min, x_max) - x

            if delta.grad is not None:
                delta.grad.zero_()

        return (x + delta).detach()
