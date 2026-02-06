"""Expectation Over Transformation (EOT) attack."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .base import Attack
from .numerics import infer_data_range, clamp_to_range, check_grad_sanity


class EOT(Attack):
    """
    Expectation Over Transformation (EOT) for stochastic defenses.
    """

    def __init__(
        self,
        model,
        transform_fn: Callable,
        num_samples: int = 20,
        importance_sampling: bool = True,
        temperature: float = 0.1,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 40,
        norm: str = "linf",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.transform_fn = transform_fn
        self.num_samples = int(num_samples)
        self.importance_sampling = bool(importance_sampling)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.norm = norm

        assert self.num_samples >= 10, "EOT requires ≥10 samples for reliable gradient estimates"

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        x_min, x_max = infer_data_range(x)

        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = self._project(delta, self.eps, self.norm)

        for step in range(self.steps):
            delta.requires_grad_(True)

            if self.importance_sampling:
                grad_acc, grad_count = self._importance_weighted_gradients(x, delta, y)
            else:
                grad_acc, grad_count = self._uniform_gradients(x, delta, y)

            avg_grad = grad_acc / max(grad_count, 1)
            if not check_grad_sanity(avg_grad, f"EOT-step{step}"):
                avg_grad = torch.sign(torch.randn_like(delta)) * 1e-3

            with torch.no_grad():
                delta = delta + self.alpha * avg_grad.sign()
                delta = self._project(delta, self.eps, self.norm)
                delta = clamp_to_range(x + delta, x_min, x_max) - x

            delta = delta.detach()

        return (x + delta).detach()

    def _uniform_gradients(self, x: torch.Tensor, delta: torch.Tensor, y: torch.Tensor) -> tuple:
        grad_acc = torch.zeros_like(delta)
        for i in range(self.num_samples):
            x_t = self.transform_fn(x + delta)
            logits = self.model(x_t)
            loss = F.cross_entropy(logits, y)
            g = torch.autograd.grad(loss, delta, retain_graph=(i < self.num_samples - 1))[0]
            grad_acc += g
        return grad_acc, self.num_samples

    def _importance_weighted_gradients(self, x: torch.Tensor, delta: torch.Tensor, y: torch.Tensor) -> tuple:
        losses = []
        grads = []

        for i in range(self.num_samples):
            x_t = self.transform_fn(x + delta)
            logits = self.model(x_t)
            loss = F.cross_entropy(logits, y)
            g = torch.autograd.grad(loss, delta, retain_graph=(i < self.num_samples - 1))[0]
            losses.append(loss.item())
            grads.append(g)

        losses_tensor = torch.tensor([-l for l in losses], device=self.device)
        weights = F.softmax(losses_tensor / self.temperature, dim=0)

        grad_weighted = torch.zeros_like(delta)
        for w, g in zip(weights, grads):
            grad_weighted += w.item() * g

        effective_n = 1.0 / (weights**2).sum().item()
        return grad_weighted, effective_n


class AdaptiveEOT(Attack):
    """
    EOT with adaptive sample count selection.
    """

    def __init__(
        self,
        model,
        transform_fn: Callable,
        target_variance: float = 0.01,
        min_samples: int = 10,
        max_samples: int = 100,
        confidence: float = 0.95,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 40,
        norm: str = "linf",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.transform_fn = transform_fn
        self.target_variance = float(target_variance)
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)
        self.confidence = float(confidence)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.norm = norm

        try:
            from scipy.stats import norm as _norm

            self.z_score = float(_norm.ppf((1 + confidence) / 2))
        except Exception:
            normal = torch.distributions.Normal(0.0, 1.0)
            self.z_score = float(normal.icdf(torch.tensor((1 + confidence) / 2)))

    def _estimate_required_samples(self, x: torch.Tensor, delta: torch.Tensor, y: torch.Tensor) -> int:
        pilot_grads = []
        for i in range(self.min_samples):
            x_t = self.transform_fn(x + delta)
            logits = self.model(x_t)
            loss = F.cross_entropy(logits, y)
            g = torch.autograd.grad(loss, delta, retain_graph=(i < self.min_samples - 1))[0]
            pilot_grads.append(g.flatten())

        pilot_grads_tensor = torch.stack(pilot_grads)
        grad_var = pilot_grads_tensor.var(dim=0).mean().item()
        required_n = int(torch.ceil(torch.tensor((grad_var / self.target_variance) * (self.z_score**2))).item())
        return max(self.min_samples, min(required_n, self.max_samples))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        x_min, x_max = infer_data_range(x)

        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = self._project(delta, self.eps, self.norm)

        num_samples = self.min_samples
        for step in range(self.steps):
            delta.requires_grad_(True)

            if step % 10 == 0:
                num_samples = self._estimate_required_samples(x, delta, y)

            grad_acc = torch.zeros_like(delta)
            for i in range(num_samples):
                x_t = self.transform_fn(x + delta)
                logits = self.model(x_t)
                loss = F.cross_entropy(logits, y)
                g = torch.autograd.grad(loss, delta, retain_graph=(i < num_samples - 1))[0]
                grad_acc += g

            avg_grad = grad_acc / float(num_samples)
            if not check_grad_sanity(avg_grad, f"AdaptiveEOT-step{step}"):
                avg_grad = torch.sign(torch.randn_like(delta)) * 1e-3

            with torch.no_grad():
                delta = delta + self.alpha * avg_grad.sign()
                delta = self._project(delta, self.eps, self.norm)
                delta = clamp_to_range(x + delta, x_min, x_max) - x

            delta = delta.detach()

        return (x + delta).detach()
