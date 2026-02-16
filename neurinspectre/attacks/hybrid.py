"""Hybrid BPDA + EOT attack."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Attack
from .eot import EOT
from .memory_gradient import MemoryAugmentedGradient
from .numerics import clamp_to_range, check_grad_sanity, infer_data_range


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


class HybridBPDAEOTVolterra(Attack):
    """
    Hybrid BPDA+EOT attack augmented with Volterra-weighted gradient memory.

    This is the "BPDA+EOT + memory" variant needed for RL-style / temporally
    correlated obfuscation where per-step EOT gradients remain noisy.
    """

    def __init__(
        self,
        model: nn.Module,
        defense,
        *,
        approx_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_samples: int = 20,
        importance_sampling: bool = True,
        temperature: float = 0.1,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 40,
        norm: str = "linf",
        # Volterra memory parameters
        alpha_volterra: float = 0.5,
        memory_length: int = 20,
        kernel_type: str = "power_law",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.defense = defense
        self.approx_fn = approx_fn or defense.get_bpda_approximation()
        self.num_samples = int(num_samples)
        self.importance_sampling = bool(importance_sampling)
        self.temperature = float(temperature)

        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.norm = str(norm)

        self.alpha_volterra = float(alpha_volterra)
        self.memory_length = int(memory_length)
        self.kernel_type = str(kernel_type)

        if self.num_samples < 10:
            raise ValueError(
                f"HybridBPDAEOTVolterra requires num_samples >= 10, got {self.num_samples}"
            )
        if self.memory_length < 2:
            raise ValueError(f"memory_length must be >= 2, got {self.memory_length}")

        self._memory = MemoryAugmentedGradient(
            memory_length=self.memory_length,
            kernel_type=self.kernel_type,
            alpha=self.alpha_volterra,
            device=device,
        )

    def _bpda_transform(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_actual = self.defense.transform(x)
        x_approx = self.approx_fn(x)
        return x_actual + (x_approx - x_approx.detach())

    def _uniform_gradients(self, x: torch.Tensor, delta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        grad_acc = torch.zeros_like(delta)
        for i in range(self.num_samples):
            x_t = self._bpda_transform(x + delta)
            logits = self.model(x_t)
            loss = F.cross_entropy(logits, y)
            g = torch.autograd.grad(loss, delta, retain_graph=(i < self.num_samples - 1))[0]
            grad_acc += g
        return grad_acc / float(self.num_samples)

    def _importance_weighted_gradients(
        self, x: torch.Tensor, delta: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        losses: list[float] = []
        grads: list[torch.Tensor] = []
        for i in range(self.num_samples):
            x_t = self._bpda_transform(x + delta)
            logits = self.model(x_t)
            loss = F.cross_entropy(logits, y)
            g = torch.autograd.grad(loss, delta, retain_graph=(i < self.num_samples - 1))[0]
            losses.append(float(loss.detach().item()))
            grads.append(g)
        weights = EOT._compute_importance_weights(  # pylint: disable=protected-access
            losses,
            temperature=self.temperature,
            device=self.device,
        )
        grad_weighted = torch.zeros_like(delta)
        for w, g in zip(weights, grads):
            grad_weighted += float(w.item()) * g
        return grad_weighted

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        x_min, x_max = infer_data_range(x)

        self._memory.reset()
        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = self._project(delta, self.eps, self.norm)

        for step in range(self.steps):
            delta.requires_grad_(True)

            if self.importance_sampling:
                grad = self._importance_weighted_gradients(x, delta, y)
            else:
                grad = self._uniform_gradients(x, delta, y)

            if not check_grad_sanity(grad, f"HybridVolterra-step{step}"):
                grad = torch.sign(torch.randn_like(delta)) * 1e-3

            grad_mem = self._memory.update_and_weight(grad)
            if not check_grad_sanity(grad_mem, f"HybridVolterraMem-step{step}"):
                grad_mem = grad

            with torch.no_grad():
                delta = delta + self.alpha * grad_mem.sign()
                delta = self._project(delta, self.eps, self.norm)
                delta = clamp_to_range(x + delta, x_min, x_max) - x
            delta = delta.detach()

        return (x + delta).detach()
