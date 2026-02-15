"""Numerical helpers for robust attack implementations."""

from __future__ import annotations

import torch


_GRAD_SANITY_WARN_COUNTS: dict[str, int] = {}
_GRAD_SANITY_MAX_WARNINGS_PER_KEY = 3


def _warn_throttled(key: str, message: str) -> None:
    """
    Print at most N warnings per logical attack key.

    Many call sites include per-step suffixes (e.g. "PGD-step17"), which can
    otherwise spam stdout and drown out the CLI report.
    """
    count = _GRAD_SANITY_WARN_COUNTS.get(key, 0)
    if count < _GRAD_SANITY_MAX_WARNINGS_PER_KEY:
        _GRAD_SANITY_WARN_COUNTS[key] = count + 1
        print(message)
        return
    if count == _GRAD_SANITY_MAX_WARNINGS_PER_KEY:
        _GRAD_SANITY_WARN_COUNTS[key] = count + 1
        print(f"[WARN] Further gradient warnings suppressed for {key}")


def infer_data_range(x: torch.Tensor) -> tuple[float, float]:
    """
    Infer min/max of data once for clamping across attack steps.

    Args:
        x: Input tensor of shape (B, C, H, W) or generic batch tensors.

    Returns:
        (x_min, x_max): Data range tuple.

    Example:
        >>> x = torch.rand(10, 3, 32, 32)
        >>> x_min, x_max = infer_data_range(x)
        >>> assert x_min <= x_max
    """
    x_min = float(x.detach().min().item())
    x_max = float(x.detach().max().item())
    return x_min, x_max


def clamp_to_range(x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
    """
    Clamp tensor to valid data range.
    """
    return x.clamp(x_min, x_max)


def safe_flat_norm(delta: torch.Tensor, p: int = 2, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute per-sample p-norm with numerical safety.

    Prevents division-by-zero in L2 projection.

    Args:
        delta: Perturbation tensor (B, C, H, W)
        p: Norm order (2 for L2, float("inf") for L-infinity)
        eps: Minimum norm value to prevent div-by-zero

    Returns:
        Per-sample norms of shape (B, 1)
    """
    flat = delta.view(delta.size(0), -1)
    return flat.norm(p=p, dim=1, keepdim=True).clamp_min(eps)


def check_grad_sanity(grad: torch.Tensor | None, name: str = "") -> bool:
    """
    Return False if gradient is NaN/Inf; caller decides fallback.

    Args:
        grad: Gradient tensor
        name: Debug identifier (e.g., "PGD", "APGD")
    """
    warn_key = str(name or "grad")
    # Collapse per-step names ("PGD-step12" -> "PGD") to avoid spamming output.
    step_idx = warn_key.find("-step")
    if step_idx != -1:
        warn_key = warn_key[:step_idx]
    if grad is None:
        tag = f" in {name}" if name else ""
        _warn_throttled(warn_key, f"[WARN] Gradient is None{tag}")
        return False
    if torch.isnan(grad).any():
        tag = f" in {name}" if name else ""
        _warn_throttled(warn_key, f"[WARN] NaN detected in gradient{tag}")
        return False
    if torch.isinf(grad).any():
        tag = f" in {name}" if name else ""
        _warn_throttled(warn_key, f"[WARN] Inf detected in gradient{tag}")
        return False
    return True


def transformed_gradient(g: torch.Tensor, s: float = 1.5, clip: float = 3.0) -> torch.Tensor:
    """
    Transformed gradient (TG) preprocessing for improved transferability.

    Scale gradients by mean absolute value, then truncate to [-clip, clip].
    """
    dims = list(range(1, g.ndim))
    mean_abs = g.abs().mean(dim=dims, keepdim=True).clamp_min(1e-12)
    g_scaled = s * g / mean_abs
    return g_scaled.clamp(-clip, clip)
