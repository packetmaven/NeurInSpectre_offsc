"""
Memory-augmented gradient estimation for MA-PGD.

Standard PGD treats gradient samples independently:
    delta_{t+1} = delta_t + alpha * sign(grad_t)

MA-PGD exploits temporal correlations via Volterra weighting:
    delta_{t+1} = delta_t + alpha * sign(grad_tilde)

where grad_tilde = sum_i w_i(alpha) * grad_{t-i} is the memory-weighted gradient.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from ..mathematical.volterra import ExponentialKernel, PowerLawKernel, UniformKernel


class MemoryAugmentedGradient:
    """
    Memory-augmented gradient estimator using Volterra weighting.

    Maintains a sliding window of past gradients and computes a weighted
    average using kernel-defined weights.
    """

    def __init__(
        self,
        memory_length: int,
        kernel_type: str = "power_law",
        alpha: Optional[float] = None,
        lambda_: Optional[float] = None,
        device: str = "cuda",
    ):
        self.memory_length = int(memory_length)
        kernel_type = str(kernel_type).lower()
        if kernel_type in ("exp", "exponential"):
            kernel_type = "exponential"
        elif kernel_type in ("power", "powerlaw", "power_law", "power-law"):
            kernel_type = "power_law"
        elif kernel_type in ("uniform",):
            kernel_type = "uniform"
        self.kernel_type = kernel_type
        self.device = device

        if self.kernel_type == "power_law":
            if alpha is None:
                alpha = 0.5
            self.kernel = PowerLawKernel(alpha=alpha)
        elif self.kernel_type == "exponential":
            if lambda_ is None:
                lambda_ = 1.0
            self.kernel = ExponentialKernel(lambda_=lambda_)
        elif self.kernel_type == "uniform":
            self.kernel = UniformKernel()
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        self.weights = torch.from_numpy(self.kernel.compute_weights(self.memory_length)).float().to(device)
        self.history: List[torch.Tensor] = []
        self.history_full = False

    def update_and_weight(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Add new gradient to history and compute weighted average.

        Args:
            gradient: Current gradient (B, C, H, W) or (B, D)

        Returns:
            Memory-weighted gradient (same shape as input)
        """
        self.history.insert(0, gradient.detach().clone())
        if len(self.history) > self.memory_length:
            self.history.pop()
        self.history_full = len(self.history) >= self.memory_length

        if not self.history_full:
            weighted_grad = torch.stack(self.history).mean(dim=0)
        else:
            grad_stack = torch.stack(self.history)
            weights_expanded = self.weights.view(self.memory_length, *([1] * (grad_stack.ndim - 1)))
            weighted_grad = (grad_stack * weights_expanded).sum(dim=0)

        return weighted_grad

    def reset(self) -> None:
        """Clear gradient history."""
        self.history.clear()
        self.history_full = False

    def set_alpha(self, alpha: float) -> None:
        """Update alpha parameter for power-law kernel."""
        if self.kernel_type != "power_law":
            raise ValueError(f"Cannot set alpha for {self.kernel_type} kernel")
        self.kernel = PowerLawKernel(alpha=alpha)
        self.weights = torch.from_numpy(self.kernel.compute_weights(self.memory_length)).float().to(self.device)

    def get_effective_memory_depth(self) -> float:
        """Compute effective memory depth (cumulative weight threshold)."""
        cumsum = torch.cumsum(self.weights, dim=0)
        threshold_idx = (cumsum >= 0.95).nonzero(as_tuple=True)[0]
        if len(threshold_idx) > 0:
            return threshold_idx[0].item() + 1
        return self.memory_length

    def __repr__(self) -> str:
        return (
            "MemoryAugmentedGradient("
            f"length={self.memory_length}, kernel={self.kernel}, "
            f"filled={len(self.history)}/{self.memory_length})"
        )


def memory_length_schedule(alpha: float, max_length: int = 50) -> int:
    """
    Determine optimal memory length based on detected alpha.

    k = max_length * (1 - alpha), clamped to [10, max_length].
    """
    k = max_length * (1.0 - float(alpha))
    return int(np.clip(k, 10, max_length))


def adaptive_kernel_selection(
    gradient_history: torch.Tensor,
    candidate_kernels: List[str] | Tuple[str, ...] = ("power_law", "exponential", "uniform"),
) -> Tuple[str, dict]:
    """
    Automatically select best kernel type based on reconstruction error.
    """
    from ..mathematical.volterra import fit_volterra_kernel

    if isinstance(gradient_history, torch.Tensor):
        grad_np = gradient_history.detach().cpu().numpy()
    else:
        grad_np = np.asarray(gradient_history)

    best_kernel = None
    best_rmse = float("inf")
    best_info: dict = {}

    for kernel_type in candidate_kernels:
        try:
            kernel, rmse, info = fit_volterra_kernel(
                grad_np,
                kernel_type=kernel_type,
                method="L-BFGS-B",
                verbose=False,
            )
            if rmse < best_rmse:
                best_rmse = rmse
                best_kernel = kernel_type
                best_info = {"kernel": kernel, "rmse": rmse, "fit_info": info}
        except Exception as exc:
            print(f"[Warning] Failed to fit {kernel_type}: {exc}")
            continue

    if best_kernel is None:
        best_kernel = "power_law"
        best_info = {
            "kernel": PowerLawKernel(alpha=0.5),
            "rmse": np.nan,
            "fit_info": {"success": False, "fallback": True},
        }

    return best_kernel, best_info
