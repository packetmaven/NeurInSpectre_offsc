"""
Memory-Augmented Projected Gradient Descent (MA-PGD) attack.

MA-PGD extends standard PGD by exploiting temporal correlations in gradient
sequences via Volterra-weighted gradient history.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Attack
from .memory_gradient import MemoryAugmentedGradient, memory_length_schedule
from .numerics import clamp_to_range, check_grad_sanity, infer_data_range, transformed_gradient
from ..mathematical.volterra import fit_volterra_kernel


class MAPGD(Attack):
    """
    Memory-Augmented PGD with Volterra-weighted gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 100,
        norm: str = "linf",
        alpha_volterra: Optional[float] = None,
        memory_length: Optional[int] = None,
        kernel_type: str = "power_law",
        use_tg: bool = True,
        tg_scale: float = 1.5,
        tg_clip: float = 3.0,
        use_momentum: bool = False,
        momentum_beta: float = 0.75,
        auto_detect_alpha: bool = False,
        n_detection_steps: int = 30,
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.eps = eps
        self.alpha_step = alpha
        self.steps = int(steps)
        self.norm = norm
        self.use_tg = use_tg
        self.tg_scale = tg_scale
        self.tg_clip = tg_clip
        self.use_momentum = use_momentum
        self.momentum_beta = momentum_beta
        self.auto_detect_alpha = auto_detect_alpha
        self.n_detection_steps = n_detection_steps

        self.alpha_volterra = alpha_volterra
        self.memory_length_config = memory_length
        self.kernel_type = kernel_type

        self.memory_grad: Optional[MemoryAugmentedGradient] = None
        self.stats: Dict[str, object] = {
            "alpha_detected": None,
            "memory_length_used": None,
            "gradient_variance": [],
            "memory_contribution": [],
            "final_asr": None,
        }

    def _detect_alpha_from_gradients(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, int]:
        """
        Auto-detect alpha by collecting initial gradient samples and fitting
        a Volterra kernel.
        """
        print("[MA-PGD] Auto-detecting alpha via gradient sampling...")
        grad_history: List[np.ndarray] = []

        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = self._project(delta, self.eps, self.norm)
        delta.requires_grad = True

        for _ in range(self.n_detection_steps):
            logits = self.model(x + delta)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            grad = delta.grad.detach().clone()
            grad_flat = grad.view(grad.size(0), -1).cpu().numpy()
            grad_history.append(grad_flat.mean(axis=0))

            with torch.no_grad():
                delta.data = delta + self.alpha_step * grad.sign()
                delta.data = self._project(delta, self.eps, self.norm)

            if delta.grad is not None:
                delta.grad.zero_()

        grad_history_np = np.array(grad_history)
        try:
            kernel, rmse, info = fit_volterra_kernel(
                grad_history_np,
                kernel_type="power_law",
                method="L-BFGS-B",
                verbose=False,
            )
            alpha_detected = float(kernel.alpha)
            if not info.get("success", False):
                print("[MA-PGD] Kernel fitting failed, using default alpha=0.5")
                alpha_detected = 0.5
        except Exception as exc:
            print(f"[MA-PGD] Kernel fitting exception: {exc}. Using default alpha=0.5")
            alpha_detected = 0.5

        memory_len = memory_length_schedule(alpha_detected, max_length=50)
        print(f"[MA-PGD] Detected alpha={alpha_detected:.4f}, scheduled k={memory_len}")

        self.stats["alpha_detected"] = alpha_detected
        self.stats["memory_length_used"] = memory_len

        return alpha_detected, memory_len

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        x_min, x_max = infer_data_range(x)

        if self.auto_detect_alpha:
            alpha_volterra, memory_length = self._detect_alpha_from_gradients(x, y)
        else:
            alpha_volterra = self.alpha_volterra if self.alpha_volterra is not None else 0.5
            memory_length = self.memory_length_config if self.memory_length_config is not None else 20
            self.stats["alpha_detected"] = alpha_volterra
            self.stats["memory_length_used"] = memory_length

        self.memory_grad = MemoryAugmentedGradient(
            memory_length=memory_length,
            kernel_type=self.kernel_type,
            alpha=alpha_volterra,
            device=self.device,
        )

        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = self._project(delta, self.eps, self.norm)
        delta.requires_grad = True

        momentum = torch.zeros_like(delta) if self.use_momentum else None

        for step in range(self.steps):
            logits = self.model(x + delta)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            grad = delta.grad.detach().clone()
            if not check_grad_sanity(grad, f"MA-PGD-step{step}"):
                print(f"[MA-PGD] Gradient sanity check failed at step {step}")
                grad = torch.sign(torch.randn_like(delta)) * 1e-3

            if self.use_tg:
                grad = transformed_gradient(grad, s=self.tg_scale, clip=self.tg_clip)

            grad_weighted = self.memory_grad.update_and_weight(grad)

            if step > memory_length:
                grad_var = grad.var().item()
                self.stats["gradient_variance"].append(grad_var)
                cos_sim = F.cosine_similarity(
                    grad.flatten(),
                    grad_weighted.flatten(),
                    dim=0,
                ).item()
                self.stats["memory_contribution"].append(cos_sim)

            if self.use_momentum:
                momentum = self.momentum_beta * momentum + grad_weighted
                final_grad = momentum
            else:
                final_grad = grad_weighted

            with torch.no_grad():
                delta.data = delta + self.alpha_step * final_grad.sign()
                delta.data = self._project(delta, self.eps, self.norm)
                delta.data = clamp_to_range(x + delta.data, x_min, x_max) - x

            if delta.grad is not None:
                delta.grad.zero_()

        with torch.no_grad():
            final_logits = self.model(x + delta)
            final_preds = final_logits.argmax(1)
            final_asr = (final_preds != y).float().mean().item()
            self.stats["final_asr"] = final_asr

        x_adv = (x + delta).detach()
        if return_stats:
            return x_adv, self.stats
        return x_adv


class MAPGDEnsemble(Attack):
    """
    MA-PGD ensemble with multiple kernel types and alpha values.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alphas: List[float] | Tuple[float, ...] = (0.3, 0.5, 0.7),
        kernel_types: List[str] | Tuple[str, ...] = ("power_law",),
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.eps = eps

        self.attacks: List[Tuple[float, str, MAPGD]] = []
        for alpha in alphas:
            for kernel_type in kernel_types:
                attack = MAPGD(
                    model,
                    eps=eps,
                    alpha_volterra=alpha,
                    memory_length=memory_length_schedule(alpha),
                    kernel_type=kernel_type,
                    device=device,
                )
                self.attacks.append((alpha, kernel_type, attack))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        best_adv = x.clone()
        best_loss = torch.full((x.size(0),), -1e10, device=self.device)

        for alpha, kernel_type, attack in self.attacks:
            print(f"[MA-PGD Ensemble] Running alpha={alpha:.2f}, kernel={kernel_type}")
            x_adv = attack(x, y)
            with torch.no_grad():
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y, reduction="none")
                improved = loss > best_loss
                best_adv[improved] = x_adv[improved]
                best_loss[improved] = loss[improved]

        return best_adv.detach()
