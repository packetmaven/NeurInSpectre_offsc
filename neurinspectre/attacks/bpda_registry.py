"""
BPDA approximation registry for common defenses.

This registry maps defense types to differentiable approximations
for use in Backward Pass Differentiable Approximation (BPDA) attacks.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F

BPDA_REGISTRY: Dict[str, Callable] = {}


def register_bpda(name: str):
    """Decorator to register a BPDA approximation."""

    def decorator(fn: Callable) -> Callable:
        BPDA_REGISTRY[name] = fn
        return fn

    return decorator


@register_bpda("identity")
def bpda_identity(x: torch.Tensor) -> torch.Tensor:
    """Identity approximation: g̃(x) = x."""
    return x


@register_bpda("jpeg")
def bpda_jpeg(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Straight-through approximation for JPEG compression defenses."""
    return x


@register_bpda("thermometer")
def bpda_thermometer(x: torch.Tensor, levels: int = 16, tau: float = 10.0) -> torch.Tensor:
    """
    Shape-preserving thermometer-style approximation with STE.

    This is a *shape-preserving* approximation (returns same shape as `x`) by
    averaging thermometer bits across levels. It is appropriate for defenses
    that behave like quantization / feature squeezing.

    For true thermometer encoding defenses that *expand channels* (e.g.,
    (B,C,H,W)->(B,C*L,H,W)), use `bpda_thermometer_bits`.
    """
    levels = int(max(2, levels))
    x_clamped = torch.clamp(x, 0.0, 1.0)

    if x_clamped.ndim >= 4:
        # (B, C, H, W) -> (B, C, L, H, W)
        thresholds = torch.arange(1, levels + 1, device=x.device, dtype=x.dtype) / levels
        thresholds = thresholds.view(1, 1, levels, 1, 1)
        x_expanded = x_clamped.unsqueeze(2)
        hard = (x_expanded >= thresholds).to(x.dtype)
        soft = torch.sigmoid((x_expanded - thresholds) * float(tau))
        encoded = soft + (hard - soft).detach()
        return encoded.mean(dim=2)

    # Generic path for tabular/flat inputs, e.g., (B, D)
    thresholds = torch.arange(1, levels + 1, device=x.device, dtype=x.dtype) / levels
    view_shape = (1,) * x_clamped.ndim + (levels,)
    thresholds = thresholds.view(*view_shape)
    x_expanded = x_clamped.unsqueeze(-1)
    hard = (x_expanded >= thresholds).to(x.dtype)
    soft = torch.sigmoid((x_expanded - thresholds) * float(tau))
    encoded = soft + (hard - soft).detach()
    return encoded.mean(dim=-1)


@register_bpda("thermometer_bits")
def bpda_thermometer_bits(x: torch.Tensor, levels: int = 16, tau: float = 10.0) -> torch.Tensor:
    """
    Channel-expanding thermometer encoding approximation with STE.

    Returns:
      - Image-like input (B, C, H, W) -> (B, C*levels, H, W)
      - Flat input (B, D) -> (B, D*levels)
    """
    levels = int(max(2, levels))
    x_clamped = torch.clamp(x, 0.0, 1.0)

    if x_clamped.ndim >= 4:
        thresholds = torch.arange(1, levels + 1, device=x.device, dtype=x.dtype) / levels
        thresholds = thresholds.view(1, 1, levels, 1, 1)
        x_expanded = x_clamped.unsqueeze(2)
        hard = (x_expanded >= thresholds).to(x.dtype)
        soft = torch.sigmoid((x_expanded - thresholds) * float(tau))
        encoded = soft + (hard - soft).detach()  # (B, C, L, H, W)
        b, c, _l, h, w = encoded.shape
        return encoded.view(b, c * levels, h, w)

    if x_clamped.ndim == 2:
        thresholds = torch.arange(1, levels + 1, device=x.device, dtype=x.dtype) / levels
        thresholds = thresholds.view(1, 1, levels)
        x_expanded = x_clamped.unsqueeze(-1)  # (B, D, 1)
        hard = (x_expanded >= thresholds).to(x.dtype)
        soft = torch.sigmoid((x_expanded - thresholds) * float(tau))
        encoded = soft + (hard - soft).detach()  # (B, D, L)
        return encoded.reshape(int(x.shape[0]), int(x.shape[1]) * levels)

    # Other shapes are unusual for thermometer encoding; keep the failure explicit.
    raise ValueError(f"bpda_thermometer_bits expects (B,C,H,W) or (B,D), got shape={tuple(x.shape)}")


@register_bpda("quantization")
def bpda_quantization(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Bit-depth reduction approximation via STE."""
    scale = 2**bits - 1
    x_quantized = torch.round(x * scale) / scale
    return x + (x_quantized - x).detach()


@register_bpda("median_filter")
def bpda_median_filter(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Median filter approximation via average pooling."""
    padding = kernel_size // 2
    return F.avg_pool2d(x, kernel_size, stride=1, padding=padding)


@register_bpda("gaussian_blur")
def bpda_gaussian_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian blur approximation via differentiable convolution."""
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    kernel = _gaussian_kernel_2d(kernel_size, sigma).to(x.device)
    kernel = kernel.expand(x.size(1), 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=x.size(1))


@register_bpda("random_resizing")
def bpda_random_resizing(x: torch.Tensor, scale_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """Approximation for random resizing (identity; use EOT for stochasticity)."""
    return x


@register_bpda("rand_resize_pad")
def bpda_rand_resize_pad(x: torch.Tensor, size: tuple[int, int] = (32, 32)) -> torch.Tensor:
    """Alias for fixed resize/pad approximation."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Generate 2D Gaussian kernel."""
    ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()
    kernel = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    return kernel.unsqueeze(0).unsqueeze(0)


class BPDAFunction(torch.autograd.Function):
    """Custom autograd function for BPDA with separate forward/backward."""

    @staticmethod
    def forward(ctx, x, defense_fn, approx_fn):
        ctx.approx_fn = approx_fn
        ctx.save_for_backward(x)
        with torch.no_grad():
            x_defended = defense_fn(x)
        return x_defended

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        approx_fn = ctx.approx_fn
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            x_approx = approx_fn(x)
            grad_input = torch.autograd.grad(
                x_approx, x, grad_output, only_inputs=True, retain_graph=False
            )[0]
        return grad_input, None, None


def apply_bpda(x: torch.Tensor, defense_fn: Callable, approx_fn: Callable) -> torch.Tensor:
    """Apply BPDA with custom forward/backward functions."""
    return BPDAFunction.apply(x, defense_fn, approx_fn)
