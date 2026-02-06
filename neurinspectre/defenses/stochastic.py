"""
Stochastic defense implementations (randomized transformations).

These defenses apply random transformations at inference time, requiring
EOT (Expectation Over Transformation) for effective attacks.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomResizing(nn.Module):
    """Random resizing and padding defense."""

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), deterministic: bool = False):
        super().__init__()
        self.scale_min, self.scale_max = scale_range
        self.deterministic = bool(deterministic)
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min must be < scale_max")
        if self.scale_min <= 0:
            raise ValueError("scale_min must be positive")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deterministic:
            scale = (self.scale_min + self.scale_max) / 2.0
        else:
            scale = torch.empty(x.size(0), device=x.device).uniform_(self.scale_min, self.scale_max)

        b, c, h, w = x.shape
        x_resized = []
        for i in range(b):
            s = scale if self.deterministic else float(scale[i].item())
            new_h = max(1, int(round(h * s)))
            new_w = max(1, int(round(w * s)))

            x_i = F.interpolate(x[i : i + 1], size=(new_h, new_w), mode="bilinear", align_corners=False)

            if new_h < h or new_w < w:
                pad_h = max(0, h - new_h)
                pad_w = max(0, w - new_w)
                x_i = F.pad(x_i, (0, pad_w, 0, pad_h), mode="constant", value=0)
            elif new_h > h or new_w > w:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                x_i = x_i[:, :, start_h : start_h + h, start_w : start_w + w]

            x_resized.append(x_i)
        return torch.cat(x_resized, dim=0)


class RandomPadding(nn.Module):
    """Random padding defense."""

    def __init__(self, max_pad: int = 4, deterministic: bool = False):
        super().__init__()
        self.max_pad = int(max_pad)
        self.deterministic = bool(deterministic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deterministic:
            pad_left = pad_right = pad_top = pad_bottom = self.max_pad // 2
        else:
            device = x.device
            pad_left = torch.randint(0, self.max_pad + 1, (x.size(0),), device=device)
            pad_right = torch.randint(0, self.max_pad + 1, (x.size(0),), device=device)
            pad_top = torch.randint(0, self.max_pad + 1, (x.size(0),), device=device)
            pad_bottom = torch.randint(0, self.max_pad + 1, (x.size(0),), device=device)

        b, c, h, w = x.shape
        x_padded = []
        for i in range(b):
            pl = pad_left if self.deterministic else int(pad_left[i].item())
            pr = pad_right if self.deterministic else int(pad_right[i].item())
            pt = pad_top if self.deterministic else int(pad_top[i].item())
            pb = pad_bottom if self.deterministic else int(pad_bottom[i].item())

            x_i = F.pad(x[i : i + 1], (pl, pr, pt, pb), mode="constant", value=0)

            max_h_shift = pt + pb
            max_w_shift = pl + pr
            if self.deterministic:
                start_h = max_h_shift // 2
                start_w = max_w_shift // 2
            else:
                start_h = int(torch.randint(0, max_h_shift + 1, (1,), device=x.device).item())
                start_w = int(torch.randint(0, max_w_shift + 1, (1,), device=x.device).item())

            x_i = x_i[:, :, start_h : start_h + h, start_w : start_w + w]
            x_padded.append(x_i)
        return torch.cat(x_padded, dim=0)


class RandomNoise(nn.Module):
    """Random Gaussian noise defense."""

    def __init__(self, std: float = 0.05, deterministic: bool = False):
        super().__init__()
        self.std = float(std)
        self.deterministic = bool(deterministic)
        self.noise_generator = torch.Generator()
        if deterministic:
            self.noise_generator.manual_seed(42)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deterministic and x.device.type == "cpu":
            noise = torch.randn_like(x, generator=self.noise_generator) * self.std
        else:
            noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)


class RandomSmoothing(nn.Module):
    """Randomized smoothing defense (simplified)."""

    def __init__(self, sigma: float = 0.25, n_samples: int = 100, alpha: float = 0.001):
        super().__init__()
        self.sigma = float(sigma)
        self.n_samples = int(n_samples)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor, return_cert: bool = False) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return torch.clamp(x + noise, 0.0, 1.0)

        noises = [torch.randn_like(x) * self.sigma for _ in range(self.n_samples)]
        x_noisy_samples = [torch.clamp(x + n, 0.0, 1.0) for n in noises]
        return x_noisy_samples[0]
