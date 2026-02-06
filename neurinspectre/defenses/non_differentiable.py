"""
Non-differentiable defense implementations (shattered gradients).

These defenses contain non-differentiable operations that break gradient flow,
requiring BPDA approximations for effective attacks.
"""

from __future__ import annotations

import io
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    warnings.warn("PIL not available, JPEG defense will use approximation")


class JPEGCompression(nn.Module):
    """
    JPEG compression defense.

    Applies lossy JPEG compression to input images, introducing quantization
    artifacts that destroy adversarial perturbations but also gradient information.
    """

    def __init__(self, quality: int = 75, differentiable: bool = False):
        super().__init__()
        if not 0 <= quality <= 100:
            raise ValueError(f"quality must be in [0, 100], got {quality}")
        self.quality = int(quality)
        self.differentiable = bool(differentiable)
        if not PIL_AVAILABLE and not self.differentiable:
            warnings.warn("PIL not available, using differentiable approximation")
            self.differentiable = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.differentiable:
            return self._differentiable_jpeg(x)
        return self._true_jpeg(x)

    def _true_jpeg(self, x: torch.Tensor) -> torch.Tensor:
        if not PIL_AVAILABLE:
            return self._differentiable_jpeg(x)

        device = x.device
        dtype = x.dtype
        compressed = []

        for img in x:
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
            if img_np.shape[-1] == 1:
                img_np = img_np[:, :, 0]
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)

            img_compressed = Image.open(buffer)
            img_compressed_np = np.array(img_compressed).astype(np.float32) / 255.0
            if img_compressed_np.ndim == 2:
                img_compressed_np = img_compressed_np[:, :, None]
            img_tensor = torch.from_numpy(img_compressed_np.transpose(2, 0, 1))
            compressed.append(img_tensor)

        result = torch.stack(compressed).to(device=device, dtype=dtype)
        return result

    def _differentiable_jpeg(self, x: torch.Tensor) -> torch.Tensor:
        if self.quality >= 75:
            return x
        q_step = (100 - self.quality) / 100.0 * 0.1
        if q_step <= 0:
            return x
        x_quantized = x + torch.tanh((torch.round(x / q_step) * q_step - x) / 0.01) * 0.01
        return x_quantized

    def get_bpda_approximation(self) -> nn.Module:
        return JPEGCompression(quality=self.quality, differentiable=True)


class BitDepthReduction(nn.Module):
    """Bit-depth reduction defense."""

    def __init__(self, bits: int = 4, differentiable: bool = False):
        super().__init__()
        if not 1 <= bits <= 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        self.bits = int(bits)
        self.differentiable = bool(differentiable)
        self.levels = 2 ** self.bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * (self.levels - 1)
        if self.differentiable:
            x_quantized = torch.round(x_scaled)
            x_quantized = x_scaled + (x_quantized - x_scaled).detach()
        else:
            x_quantized = torch.round(x_scaled)
        return x_quantized / (self.levels - 1)

    def get_bpda_approximation(self) -> nn.Module:
        return BitDepthReduction(bits=self.bits, differentiable=True)


class ThermometerEncoding(nn.Module):
    """Thermometer encoding defense."""

    def __init__(self, levels: int = 16, differentiable: bool = False):
        super().__init__()
        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")
        self.levels = int(levels)
        self.differentiable = bool(differentiable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * (self.levels - 1)
        if self.differentiable:
            x_quantized = torch.round(x_scaled)
            x_quantized = x_scaled + (x_quantized - x_scaled).detach()
        else:
            x_quantized = torch.round(x_scaled)
        return x_quantized / (self.levels - 1)

    def get_bpda_approximation(self) -> nn.Module:
        return ThermometerEncoding(levels=self.levels, differentiable=True)


class MedianFilter(nn.Module):
    """Median filtering defense."""

    def __init__(self, kernel_size: int = 3, differentiable: bool = False):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.kernel_size = int(kernel_size)
        self.differentiable = bool(differentiable)
        self.padding = self.kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.differentiable:
            return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        return self._true_median_filter(x)

    def _true_median_filter(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        x_unfold = x_unfold.view(b, c, self.kernel_size ** 2, h * w)
        x_median = x_unfold.median(dim=2)[0]
        return x_median.view(b, c, h, w)

    def get_bpda_approximation(self) -> nn.Module:
        return MedianFilter(kernel_size=self.kernel_size, differentiable=True)


class TotalVariationMinimization(nn.Module):
    """Total variation (TV) minimization defense."""

    def __init__(self, weight: float = 0.1, n_iter: int = 10, differentiable: bool = False):
        super().__init__()
        self.weight = float(weight)
        self.n_iter = int(n_iter)
        self.differentiable = bool(differentiable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.differentiable:
            tv_grad = self._compute_tv_gradient(x)
            return x - self.weight * tv_grad
        return self._iterative_tv(x)

    def _compute_tv_gradient(self, x: torch.Tensor) -> torch.Tensor:
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dx = F.pad(dx, (0, 1, 0, 0), mode="constant", value=0)
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = F.pad(dy, (0, 0, 0, 1), mode="constant", value=0)
        epsilon = 1e-8
        tv_norm = torch.sqrt(dx ** 2 + dy ** 2 + epsilon)
        grad = (dx + dy) / tv_norm
        return grad

    def _iterative_tv(self, x: torch.Tensor) -> torch.Tensor:
        x_tv = x.clone()
        for _ in range(self.n_iter):
            tv_grad = self._compute_tv_gradient(x_tv)
            x_tv = x_tv - self.weight * tv_grad
            x_tv = 0.9 * x_tv + 0.1 * x
            x_tv = torch.clamp(x_tv, 0.0, 1.0)
        return x_tv

    def get_bpda_approximation(self) -> nn.Module:
        return TotalVariationMinimization(
            weight=self.weight,
            n_iter=self.n_iter,
            differentiable=True,
        )
