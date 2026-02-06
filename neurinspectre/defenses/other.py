"""
Miscellaneous defense implementations.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefensiveDistillation(nn.Module):
    """Defensive distillation defense."""

    def __init__(self, model: nn.Module, temperature: float = 20.0, inference_temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = float(temperature)
        self.inference_temperature = float(inference_temperature)

    def forward(self, x: torch.Tensor, use_temp: bool = True) -> torch.Tensor:
        logits = self.model(x)
        if use_temp:
            if self.training:
                return logits / self.temperature
            return logits / self.inference_temperature
        return logits


class FeatureSqueezing(nn.Module):
    """Feature squeezing defense."""

    def __init__(self, bit_depth: int = 5, spatial_filter: bool = True, filter_size: int = 2):
        super().__init__()
        self.bit_depth = int(bit_depth)
        self.spatial_filter = bool(spatial_filter)
        self.filter_size = int(filter_size)
        self.levels = 2 ** self.bit_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * (self.levels - 1)
        x_quantized = torch.round(x_scaled) / (self.levels - 1)
        if self.spatial_filter:
            x_quantized = F.avg_pool2d(
                x_quantized,
                kernel_size=self.filter_size,
                stride=1,
                padding=self.filter_size // 2,
            )
        return x_quantized


class GradientMasking(nn.Module):
    """Gradient masking defense."""

    def __init__(self, masking_type: str = "saturate"):
        super().__init__()
        if masking_type not in ["saturate", "smooth", "shatter"]:
            raise ValueError(f"Unknown masking type: {masking_type}")
        self.masking_type = masking_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.masking_type == "saturate":
            return torch.tanh(x * 10)
        if self.masking_type == "smooth":
            return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        if self.masking_type == "shatter":
            return torch.round(x * 255) / 255
        return x


class EnsembleDefense(nn.Module):
    """Ensemble of multiple defenses."""

    def __init__(self, defenses: list):
        super().__init__()
        self.defenses = nn.ModuleList(defenses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.randint(0, len(self.defenses), (1,), device=x.device).item()
        return self.defenses[idx](x)
