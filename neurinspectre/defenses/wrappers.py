"""
Unified defense wrappers with obfuscation metadata and BPDA/EOT support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..characterization.defense_analyzer import ObfuscationType
from .non_differentiable import (
    JPEGCompression,
    BitDepthReduction,
    MedianFilter,
    TotalVariationMinimization,
)
from .stochastic import RandomPadding, RandomNoise
from .other import DefensiveDistillation

logger = logging.getLogger(__name__)


@dataclass
class DefenseSpec:
    """
    Specification for a defense wrapper.
    """

    name: str
    domain: str
    obfuscation_types: List[ObfuscationType]
    params: Dict[str, Any] = field(default_factory=dict)

    requires_bpda: bool = False
    bpda_approximation: str = "identity"

    is_stochastic: bool = False
    eot_samples: int = 1

    expected_gradient_norm: str = "normal"
    gradient_variance: str = "low"

    claimed_robust_accuracy: float = 0.0

    def get_key(self) -> str:
        return f"{self.domain}_{self.name}"


class DefenseWrapper(nn.Module):
    """
    Unified defense wrapper base class.
    """

    def __init__(self, base_model: nn.Module, spec: DefenseSpec, device: str = "cuda"):
        super().__init__()
        self.spec = spec
        self.device = device
        self.base_model = base_model.to(device)
        self.base_model.eval()

        self._eot_enabled = False
        self._eot_samples = int(spec.eot_samples)

        logger.debug(
            "[DefenseWrapper] %s obf=%s bpda=%s eot=%s",
            spec.name,
            [o.value for o in spec.obfuscation_types],
            spec.requires_bpda,
            spec.is_stochastic,
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda x: x

    def apply_defense(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    def _bpda_forward(self, x: torch.Tensor) -> torch.Tensor:
        approx_fn = self.get_bpda_approximation()
        with torch.no_grad():
            x_actual = self.transform(x)
        x_approx = approx_fn(x)
        return x_actual + (x_approx - x_approx.detach())

    def forward(self, x: torch.Tensor, use_approximation: bool = False) -> torch.Tensor:
        if use_approximation and self.spec.requires_bpda:
            x_defended = self._bpda_forward(x)
        else:
            x_defended = self.transform(x)
        return self.base_model(x_defended)

    def enable_eot(self, n_samples: Optional[int] = None) -> None:
        self._eot_enabled = True
        if n_samples is not None:
            self._eot_samples = int(n_samples)

    def disable_eot(self) -> None:
        self._eot_enabled = False

    @property
    def obfuscation_types(self) -> List[ObfuscationType]:
        return self.spec.obfuscation_types

    @property
    def obfuscation_type(self) -> ObfuscationType:
        if len(self.spec.obfuscation_types) == 1:
            return self.spec.obfuscation_types[0]
        return ObfuscationType.HYBRID

    @property
    def requires_bpda(self) -> bool:
        return bool(self.spec.requires_bpda)

    @property
    def requires_eot(self) -> bool:
        return bool(self.spec.is_stochastic)


class JPEGCompressionDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, quality: int = 75, device: str = "cuda"):
        spec = DefenseSpec(
            name="jpeg_compression",
            domain="vision",
            obfuscation_types=[ObfuscationType.SHATTERED],
            params={"quality": int(quality)},
            requires_bpda=True,
            bpda_approximation="jpeg",
            claimed_robust_accuracy=0.89,
        )
        super().__init__(base_model, spec, device)
        self.defense = JPEGCompression(quality=quality, differentiable=False)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.defense(x)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.defense.get_bpda_approximation()


class BitDepthReductionDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, bits: int = 4, device: str = "cuda"):
        spec = DefenseSpec(
            name="bit_depth_reduction",
            domain="vision",
            obfuscation_types=[ObfuscationType.SHATTERED],
            params={"bits": int(bits)},
            requires_bpda=True,
            bpda_approximation="identity",
            claimed_robust_accuracy=0.91,
        )
        super().__init__(base_model, spec, device)
        self.defense = BitDepthReduction(bits=bits, differentiable=False)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.defense(x)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.defense.get_bpda_approximation()


class ThermometerEncodingDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, levels: int = 16, device: str = "cuda"):
        spec = DefenseSpec(
            name="thermometer_encoding",
            domain="av_perception",
            obfuscation_types=[ObfuscationType.SHATTERED],
            params={"levels": int(levels)},
            requires_bpda=True,
            bpda_approximation="soft_thermometer",
            claimed_robust_accuracy=0.95,
        )
        self.levels = int(levels)
        super().__init__(base_model, spec, device)
        self._adapt_model_input()

    def _adapt_model_input(self) -> None:
        if hasattr(self.base_model, "conv1"):
            conv1 = self.base_model.conv1
            out_channels = conv1.out_channels
            kernel_size = conv1.kernel_size
            stride = conv1.stride
            padding = conv1.padding
            new_conv1 = nn.Conv2d(
                in_channels=3 * self.levels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            with torch.no_grad():
                original_weight = conv1.weight.data
                new_weight = original_weight.repeat(1, self.levels, 1, 1) / self.levels
                new_conv1.weight.data = new_weight
            self.base_model.conv1 = new_conv1

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        thresholds = torch.linspace(0, 1, self.levels + 1, device=x.device)[1:]
        thresholds = thresholds.view(1, 1, -1, 1, 1)
        x_expanded = x.unsqueeze(2)
        thermometer = (x_expanded > thresholds).float()
        return thermometer.view(b, c * self.levels, h, w)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        levels = self.levels

        class SoftThermometer(nn.Module):
            def __init__(self, levels: int, tau: float = 10.0):
                super().__init__()
                self.levels = int(levels)
                self.tau = float(tau)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, h, w = x.shape
                thresholds = torch.linspace(0, 1, self.levels + 1, device=x.device)[1:]
                thresholds = thresholds.view(1, 1, -1, 1, 1)
                x_expanded = x.unsqueeze(2)
                soft = torch.sigmoid((x_expanded - thresholds) * self.tau)
                return soft.view(b, c * self.levels, h, w)

        return SoftThermometer(levels=levels).to(self.device)


class FeatureSqueezingDefense(DefenseWrapper):
    def __init__(
        self,
        base_model: nn.Module,
        bit_depth: int = 5,
        kernel_size: int = 3,
        device: str = "cuda",
    ):
        if kernel_size % 2 == 0:
            kernel_size += 1
        spec = DefenseSpec(
            name="feature_squeezing",
            domain="malware",
            obfuscation_types=[ObfuscationType.SHATTERED],
            params={"bit_depth": int(bit_depth), "kernel_size": int(kernel_size)},
            requires_bpda=True,
            bpda_approximation="identity",
            claimed_robust_accuracy=0.93,
        )
        super().__init__(base_model, spec, device)
        self.squeeze = BitDepthReduction(bits=bit_depth, differentiable=False)
        self.median = MedianFilter(kernel_size=kernel_size, differentiable=False)
        self.squeeze_bpda = self.squeeze.get_bpda_approximation()
        self.median_bpda = self.median.get_bpda_approximation()
        self.kernel_size = int(kernel_size)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x_squeezed = self.squeeze(x)
        if x_squeezed.ndim == 2:
            return self._median_filter_1d(x_squeezed)
        return self.median(x_squeezed)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        class _Approx(nn.Module):
            def __init__(self, levels: int, kernel_size: int, squeeze_bpda: nn.Module, median_bpda: nn.Module):
                super().__init__()
                self.levels = int(levels)
                self.kernel_size = int(kernel_size)
                self.squeeze_bpda = squeeze_bpda
                self.median_bpda = median_bpda

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.ndim == 2:
                    return self._approx_1d(x)
                return self.median_bpda(self.squeeze_bpda(x))

            def _approx_1d(self, x: torch.Tensor) -> torch.Tensor:
                levels = self.levels
                x_scaled = x * (levels - 1)
                x_q = torch.round(x_scaled)
                x_q = x_scaled + (x_q - x_scaled).detach()
                x_q = x_q / (levels - 1)
                if self.kernel_size <= 1:
                    return x_q
                pad = self.kernel_size // 2
                x_pad = F.pad(x_q.unsqueeze(1), (pad, pad), mode="reflect")
                x_avg = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1).squeeze(1)
                return x_avg

        return _Approx(self.squeeze.levels, self.kernel_size, self.squeeze_bpda, self.median_bpda).to(self.device)

    def _median_filter_1d(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        pad = self.kernel_size // 2
        x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
        x_unfold = x_pad.unfold(dimension=2, size=self.kernel_size, step=1)
        x_med = x_unfold.median(dim=3)[0].squeeze(1)
        return x_med


class RandomizedSmoothingDefense(DefenseWrapper):
    def __init__(
        self,
        base_model: nn.Module,
        sigma: float = 0.25,
        n_samples: int = 100,
        device: str = "cuda",
    ):
        spec = DefenseSpec(
            name="randomized_smoothing",
            domain="vision",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"sigma": float(sigma), "n_samples": int(n_samples)},
            is_stochastic=True,
            eot_samples=50,
            claimed_robust_accuracy=0.85,
        )
        super().__init__(base_model, spec, device)
        self.sigma = float(sigma)
        self.n_samples = int(n_samples)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        return torch.clamp(x + noise, 0.0, 1.0)

    def forward(self, x: torch.Tensor, use_approximation: bool = False) -> torch.Tensor:
        if self._eot_enabled:
            return self.base_model(self.transform(x))
        logits_acc = None
        for _ in range(self.n_samples):
            logits = self.base_model(self.transform(x))
            logits_acc = logits if logits_acc is None else logits_acc + logits
        return logits_acc / float(self.n_samples)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.transform

    def certified_radius(self, x: torch.Tensor, n_samples: int = 1000) -> float:
        try:
            from scipy.stats import norm as _norm
        except Exception as exc:
            raise ImportError("scipy is required for certified radius computation") from exc
        counts = None
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.base_model(self.transform(x))
                preds = logits.argmax(dim=1)
                if counts is None:
                    counts = torch.zeros(logits.size(1), device=x.device)
                counts[preds] += 1
        p_a = (counts.max() / n_samples).item() if counts is not None else 0.0
        return self.sigma * _norm.ppf(p_a) if p_a > 0.5 else 0.0


class RandomPadCropDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, pad_size: int = 4, device: str = "cuda"):
        spec = DefenseSpec(
            name="random_pad_crop",
            domain="av_perception",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"pad_size": int(pad_size)},
            is_stochastic=True,
            eot_samples=30,
            claimed_robust_accuracy=0.92,
        )
        super().__init__(base_model, spec, device)
        self.pad = RandomPadding(max_pad=pad_size, deterministic=False)
        self.pad_bpda = RandomPadding(max_pad=pad_size, deterministic=True)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.pad(x)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.pad_bpda


class RandomNoiseDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, std: float = 0.05, device: str = "cuda"):
        spec = DefenseSpec(
            name="random_noise",
            domain="vision",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"std": float(std)},
            is_stochastic=True,
            eot_samples=20,
            claimed_robust_accuracy=0.86,
        )
        super().__init__(base_model, spec, device)
        self.noise = RandomNoise(std=std, deterministic=False)
        self.noise_bpda = RandomNoise(std=std, deterministic=True)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.noise(x)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.noise_bpda


class EnsembleDiversityDefense(DefenseWrapper):
    def __init__(self, models: List[nn.Module], aggregation: str = "average", device: str = "cuda"):
        spec = DefenseSpec(
            name="ensemble_diversity",
            domain="vision",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"n_models": len(models), "aggregation": aggregation},
            is_stochastic=(aggregation == "random"),
            eot_samples=max(1, len(models)),
            claimed_robust_accuracy=0.88,
        )
        super().__init__(models[0], spec, device)
        self.models = nn.ModuleList([m.to(device).eval() for m in models])
        self.aggregation = str(aggregation)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor, use_approximation: bool = False) -> torch.Tensor:
        if use_approximation:
            return self._average_logits(x)
        if self.aggregation == "random":
            idx = torch.randint(0, len(self.models), (1,), device=x.device).item()
            return self.models[idx](x)
        if self.aggregation == "average":
            return self._average_logits(x)
        if self.aggregation == "vote":
            logits = self._average_logits(x)
            preds = logits.argmax(dim=1)
            num_classes = logits.size(1)
            votes = torch.zeros(x.size(0), num_classes, device=x.device)
            votes.scatter_add_(1, preds.unsqueeze(1), torch.ones_like(preds.unsqueeze(1), dtype=votes.dtype))
            return votes
        raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _average_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits_acc = None
        for model in self.models:
            logits = model(x)
            logits_acc = logits if logits_acc is None else logits_acc + logits
        return logits_acc / float(len(self.models))

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda x: x


class DefensiveDistillationDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, temperature: float = 20.0, device: str = "cuda"):
        spec = DefenseSpec(
            name="defensive_distillation",
            domain="malware",
            obfuscation_types=[ObfuscationType.VANISHING],
            params={"temperature": float(temperature)},
            claimed_robust_accuracy=0.97,
        )
        super().__init__(base_model, spec, device)
        self.distill = DefensiveDistillation(base_model, temperature=temperature, inference_temperature=1.0)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor, use_approximation: bool = False) -> torch.Tensor:
        return self.distill(x, use_temp=True)


class GradientRegularizationDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, lambda_grad: float = 1.0, device: str = "cuda"):
        spec = DefenseSpec(
            name="gradient_regularization",
            domain="malware",
            obfuscation_types=[ObfuscationType.VANISHING],
            params={"lambda_grad": float(lambda_grad)},
            claimed_robust_accuracy=0.87,
        )
        super().__init__(base_model, spec, device)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SpatialSmoothingDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, kernel_size: int = 3, sigma: float = 1.0, device: str = "cuda"):
        if kernel_size % 2 == 0:
            kernel_size += 1
        spec = DefenseSpec(
            name="spatial_smoothing",
            domain="av_perception",
            obfuscation_types=[ObfuscationType.VANISHING],
            params={"kernel_size": int(kernel_size), "sigma": float(sigma)},
            claimed_robust_accuracy=0.90,
        )
        super().__init__(base_model, spec, device)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.kernel = self._build_kernel().to(device)

    def _build_kernel(self) -> torch.Tensor:
        coords = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        g = torch.exp(-(coords**2) / (2 * self.sigma**2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        return kernel.unsqueeze(0).unsqueeze(0)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _h, _w = x.shape
        kernel = self.kernel.expand(c, 1, -1, -1)
        padding = self.kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=c)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.transform


class CertifiedDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, sigma: float = 0.5, device: str = "cuda"):
        spec = DefenseSpec(
            name="certified_defense",
            domain="av_perception",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"sigma": float(sigma)},
            is_stochastic=True,
            eot_samples=100,
            claimed_robust_accuracy=0.85,
        )
        super().__init__(base_model, spec, device)
        self.sigma = float(sigma)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        return torch.clamp(x + noise, 0, 1)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.transform


class ATTransformDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, noise_std: float = 0.05, device: str = "cuda"):
        spec = DefenseSpec(
            name="at_transform",
            domain="malware",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"noise_std": float(noise_std)},
            is_stochastic=True,
            eot_samples=20,
            claimed_robust_accuracy=0.86,
        )
        super().__init__(base_model, spec, device)
        self.noise = RandomNoise(std=noise_std, deterministic=False)
        self.noise_bpda = RandomNoise(std=noise_std, deterministic=True)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.noise(x)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.noise_bpda


class TotalVariationDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, weight: float = 0.1, n_iter: int = 10, device: str = "cuda"):
        spec = DefenseSpec(
            name="total_variation",
            domain="vision",
            obfuscation_types=[ObfuscationType.SHATTERED],
            params={"weight": float(weight), "n_iter": int(n_iter)},
            requires_bpda=True,
            bpda_approximation="tv",
            claimed_robust_accuracy=0.82,
        )
        super().__init__(base_model, spec, device)
        self.tv = TotalVariationMinimization(weight=weight, n_iter=n_iter, differentiable=False)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.tv(x)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.tv.get_bpda_approximation()


__all__ = [
    "DefenseSpec",
    "DefenseWrapper",
    "JPEGCompressionDefense",
    "BitDepthReductionDefense",
    "ThermometerEncodingDefense",
    "FeatureSqueezingDefense",
    "RandomizedSmoothingDefense",
    "RandomPadCropDefense",
    "RandomNoiseDefense",
    "EnsembleDiversityDefense",
    "DefensiveDistillationDefense",
    "GradientRegularizationDefense",
    "SpatialSmoothingDefense",
    "CertifiedDefense",
    "ATTransformDefense",
    "TotalVariationDefense",
]
