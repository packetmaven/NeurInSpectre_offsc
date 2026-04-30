"""
Unified defense wrappers with obfuscation metadata and BPDA/EOT support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import logging
import math

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
        )
        self.levels = int(levels)
        self._use_channel_adapter = False
        super().__init__(base_model, spec, device)
        self._adapt_model_input()

    def _adapt_model_input(self) -> None:
        if isinstance(self.base_model, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)):
            self._use_channel_adapter = True
            logger.warning(
                "ThermometerEncodingDefense: TorchScript model detected; "
                "using channel-reduction fallback instead of mutating conv1."
            )
            return
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
            try:
                self.base_model.conv1 = new_conv1
            except Exception as exc:
                self._use_channel_adapter = True
                logger.warning(
                    "ThermometerEncodingDefense: unable to replace conv1 (%s); "
                    "using channel-reduction fallback.",
                    exc,
                )
        else:
            self._use_channel_adapter = True
            logger.warning(
                "ThermometerEncodingDefense: base model missing conv1; "
                "using channel-reduction fallback."
            )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        thresholds = torch.linspace(0, 1, self.levels + 1, device=x.device)[1:]
        thresholds = thresholds.view(1, 1, -1, 1, 1)
        x_expanded = x.unsqueeze(2)
        thermometer = (x_expanded > thresholds).float()
        encoded = thermometer.view(b, c * self.levels, h, w)
        if self._use_channel_adapter:
            return encoded.view(b, c, self.levels, h, w).mean(dim=2)
        return encoded

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

        approx = SoftThermometer(levels=levels).to(self.device)
        if not self._use_channel_adapter:
            return approx

        class _ChannelReduce(nn.Module):
            def __init__(self, inner: nn.Module, levels: int):
                super().__init__()
                self.inner = inner
                self.levels = int(levels)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, h, w = x.shape
                encoded = self.inner(x)
                return encoded.view(b, c, self.levels, h, w).mean(dim=2)

        return _ChannelReduce(approx, levels=levels).to(self.device)


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
        # Vectorize the Monte-Carlo estimate to avoid Python loops. This matters
        # for Table-style runs where `n_samples` is large.
        n = int(self.n_samples)
        if n <= 1:
            return self.base_model(self.transform(x))
        b = int(x.size(0))
        # [n, B, C, H, W]
        noise = torch.randn((n, *x.shape), device=x.device, dtype=x.dtype) * float(self.sigma)
        x_rep = x.unsqueeze(0).expand(n, *x.shape)
        x_noisy = torch.clamp(x_rep + noise, 0.0, 1.0).reshape(n * b, *x.shape[1:])

        # Chunk to avoid OOM on laptop GPUs / MPS.
        max_batch = 512
        logits_chunks = []
        for chunk in x_noisy.split(max_batch, dim=0):
            logits_chunks.append(self.base_model(chunk))
        logits = torch.cat(logits_chunks, dim=0).reshape(n, b, -1).mean(dim=0)
        return logits

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.transform

    def certified_radius(self, x: torch.Tensor, n_samples: int = 1000) -> float:
        """
        Estimate a randomized-smoothing certified radius for a single sample.

        Notes:
        - This is a lightweight estimator used for diagnostics and failure analysis.
        - `scipy.stats.norm.ppf(1.0)` is `inf`, so we clamp p_a away from 1.0.
        """
        try:
            from scipy.stats import norm as _norm
        except Exception as exc:
            raise ImportError("scipy is required for certified radius computation") from exc

        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if x.ndim == 0 or int(x.shape[0]) != 1:
            raise ValueError("certified_radius expects a single input (batch_size==1).")

        counts = None
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.base_model(self.transform(x))
                preds = logits.argmax(dim=1)
                if counts is None:
                    if logits.ndim != 2 or int(logits.size(0)) != 1:
                        raise ValueError(
                            "certified_radius expects logits with shape (1, C). "
                            f"Got shape={tuple(logits.shape)}"
                        )
                    counts = torch.zeros(int(logits.size(1)), device=x.device, dtype=torch.float32)
                # Use scatter_add_ so repeated indices accumulate correctly.
                counts.scatter_add_(0, preds.to(torch.int64), torch.ones_like(preds, dtype=counts.dtype))

        if counts is None or float(counts.sum().item()) <= 0.0:
            return 0.0

        p_a = float((counts.max() / counts.sum()).item())
        if p_a <= 0.5:
            return 0.0

        # Clamp away from {0,1} to keep ppf finite.
        eps = 1e-6
        p_a = max(0.5 + eps, min(p_a, 1.0 - eps))
        radius = float(self.sigma) * float(_norm.ppf(p_a))
        if (not math.isfinite(radius)) or radius < 0.0:
            return 0.0
        return float(radius)


class RandomPadCropDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, pad_size: int = 4, device: str = "cuda"):
        spec = DefenseSpec(
            name="random_pad_crop",
            domain="av_perception",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"pad_size": int(pad_size)},
            is_stochastic=True,
            eot_samples=30,
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
        )
        super().__init__(base_model, spec, device)
        self.noise = RandomNoise(std=std, deterministic=False)
        self.noise_bpda = RandomNoise(std=std, deterministic=True)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.noise(x)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.noise_bpda


class RLObfuscationDefense(DefenseWrapper):
    """
    Stateful, temporally correlated obfuscation defense (RL-style).

    This wrapper is intentionally lightweight: it is meant for *decisive*
    Volterra-memory experiments (Issue 6) and for exercising the
    `ObfuscationType.RL_TRAINED` characterization path on real datasets.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        bits: int = 6,
        std: float = 0.08,
        alpha: float = 0.60,
        n_samples: int = 32,
        device: str = "cuda",
    ):
        spec = DefenseSpec(
            name="rl_obfuscation",
            domain="vision",
            obfuscation_types=[ObfuscationType.SHATTERED, ObfuscationType.STOCHASTIC, ObfuscationType.RL_TRAINED],
            params={
                "bits": int(bits),
                "std": float(std),
                "alpha": float(alpha),
                "n_samples": int(n_samples),
            },
            requires_bpda=True,
            bpda_approximation="identity",
            is_stochastic=True,
            eot_samples=20,
        )
        super().__init__(base_model, spec, device)

        self.bits = int(bits)
        self.std = float(std)
        self.alpha = float(alpha)
        self.n_samples = int(n_samples)

        # Use a straight-through estimator so gradients remain measurable for
        # characterization (Volterra/autocorr). The attack still uses BPDA+EOT.
        self._quant = BitDepthReduction(bits=self.bits, differentiable=True)
        self._quant_bpda = self._quant.get_bpda_approximation()

        # Internal state to induce temporal correlation (Volterra signature).
        self._prev_noise: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self._prev_noise = None

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self._quant(x)
        noise = torch.randn_like(x_q) * self.std
        if self._prev_noise is not None and self._prev_noise.shape == noise.shape:
            noise = self.alpha * self._prev_noise + (1.0 - self.alpha) * noise
        self._prev_noise = noise.detach()
        return torch.clamp(x_q + noise, 0.0, 1.0)

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        # Deterministic approximation (no stochastic noise) for BPDA.
        return self._quant_bpda

    def forward(self, x: torch.Tensor, use_approximation: bool = False) -> torch.Tensor:
        # Similar to randomized smoothing: average logits at eval-time to preserve
        # clean accuracy under high noise, but keep a single-sample path when
        # EOT is explicitly enabled.
        if self._eot_enabled:
            return self.base_model(self.transform(x))
        logits_acc = None
        for _ in range(self.n_samples):
            logits = self.base_model(self.transform(x))
            logits_acc = logits if logits_acc is None else logits_acc + logits
        return logits_acc / float(max(1, self.n_samples))


class TentDefense(DefenseWrapper):
    """
    Test-Time Entropy Minimization (TENT) wrapper.

    Tier 2 motivation:
    - Provides a practical, non-synthetic *stateful* defense proxy for validating
      RL/Volterra detection logic without needing custom RL training pipelines.
    - Updates BatchNorm affine parameters at inference time by minimizing the
      entropy of predictions (i.e., adapts to the test distribution).

    Notes:
    - This wrapper changes model parameters at inference time. That is expected.
    - This is not "stochastic" in the EOT sense; it is stateful across calls.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-3,
        steps: int = 1,
        reset_each_forward: bool = False,
        device: str = "cuda",
    ):
        spec = DefenseSpec(
            name="tent",
            domain="vision",
            # Treat as "policy-like / stateful" for reporting; characterization
            # still drives attack routing.
            obfuscation_types=[ObfuscationType.RL_TRAINED],
            params={
                "lr": float(lr),
                "steps": int(steps),
                "reset_each_forward": bool(reset_each_forward),
            },
            requires_bpda=False,
            is_stochastic=False,
        )
        super().__init__(base_model, spec, device)
        self.lr = float(lr)
        self.steps = int(steps)
        self.reset_each_forward = bool(reset_each_forward)

        self._tent_params: List[nn.Parameter] = []
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._initial_state: Optional[Dict[str, torch.Tensor]] = None

        self._configure_tent()
        if self.reset_each_forward:
            # Only cache full state when requested; can be large for big models.
            self._initial_state = {k: v.detach().cpu().clone() for k, v in self.base_model.state_dict().items()}

    def _configure_tent(self) -> None:
        # Freeze all params; enable only BN affine parameters.
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        params: List[nn.Parameter] = []
        for m in self.base_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if getattr(m, "weight", None) is not None:
                    m.weight.requires_grad_(True)
                    params.append(m.weight)
                if getattr(m, "bias", None) is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)

        self._tent_params = params
        if not self._tent_params:
            logger.warning("TentDefense: no BatchNorm affine parameters found; no-op defense.")
            self._optimizer = None
            return

        self._optimizer = torch.optim.SGD(self._tent_params, lr=float(self.lr), momentum=0.0)

    def reset_state(self) -> None:
        if self._initial_state is None:
            return
        self.base_model.load_state_dict(self._initial_state, strict=False)
        if self._optimizer is not None:
            self._optimizer.state = {}

    @staticmethod
    def _disable_dropout(model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()

    @staticmethod
    def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
        # Binary-logit models: shape [B] or [B,1]
        if logits.ndim == 1:
            p = torch.sigmoid(logits)
            p = p.clamp(min=1e-12, max=1.0 - 1e-12)
            ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            return ent.mean()
        if logits.ndim == 2 and int(logits.size(1)) == 1:
            p = torch.sigmoid(logits.squeeze(1))
            p = p.clamp(min=1e-12, max=1.0 - 1e-12)
            ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            return ent.mean()

        # Multi-class logits
        p = F.softmax(logits, dim=1)
        p = p.clamp_min(1e-12)
        return (-(p * p.log()).sum(dim=1)).mean()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # TENT is a model adaptation defense, not an input-space transform.
        return x

    def forward(self, x: torch.Tensor, use_approximation: bool = False) -> torch.Tensor:
        x = x.to(self.device)
        if self.reset_each_forward:
            self.reset_state()

        if self.steps <= 0 or self._optimizer is None:
            return self.base_model(self.transform(x))

        # Adaptation phase: update BN affine parameters using entropy minimization.
        self.base_model.train()
        self._disable_dropout(self.base_model)
        for _ in range(int(self.steps)):
            logits = self.base_model(self.transform(x))
            loss = self._entropy_loss(logits)
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()

        # Inference phase: return logits from adapted model without tracking grads.
        self.base_model.eval()
        with torch.no_grad():
            return self.base_model(self.transform(x))


class EnsembleDiversityDefense(DefenseWrapper):
    def __init__(self, models: List[nn.Module], aggregation: str = "average", device: str = "cuda"):
        spec = DefenseSpec(
            name="ensemble_diversity",
            domain="vision",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"n_models": len(models), "aggregation": aggregation},
            is_stochastic=(aggregation == "random"),
            eot_samples=max(1, len(models)),
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
    def __init__(self, base_model: nn.Module, sigma: float = 0.5, n_samples: int = 100, device: str = "cuda"):
        spec = DefenseSpec(
            name="certified_defense",
            domain="av_perception",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"sigma": float(sigma), "n_samples": int(n_samples)},
            is_stochastic=True,
            eot_samples=100,
        )
        super().__init__(base_model, spec, device)
        self.sigma = float(sigma)
        self.n_samples = int(n_samples)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        return torch.clamp(x + noise, 0, 1)

    def forward(self, x: torch.Tensor, use_approximation: bool = False) -> torch.Tensor:
        # Same rationale as RandomizedSmoothingDefense:
        # - When EOT is enabled (attacks), we provide a single-sample stochastic forward.
        # - For evaluation/clean inference, average logits to avoid collapsing accuracy under noise.
        if self._eot_enabled:
            return self.base_model(self.transform(x))

        n = int(self.n_samples)
        if n <= 1:
            return self.base_model(self.transform(x))

        b = int(x.size(0))
        noise = torch.randn((n, *x.shape), device=x.device, dtype=x.dtype) * float(self.sigma)
        x_rep = x.unsqueeze(0).expand(n, *x.shape)
        x_noisy = torch.clamp(x_rep + noise, 0.0, 1.0).reshape(n * b, *x.shape[1:])

        max_batch = 256
        logits_chunks = []
        for chunk in x_noisy.split(max_batch, dim=0):
            logits_chunks.append(self.base_model(chunk))
        logits = torch.cat(logits_chunks, dim=0).reshape(n, b, -1).mean(dim=0)
        return logits

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.transform

    def certified_radius(self, x: torch.Tensor, n_samples: int = 1000) -> float:
        """
        Estimate a randomized-smoothing certified radius for a single sample.

        This is a diagnostic helper (not a full certification pipeline).
        """
        try:
            from scipy.stats import norm as _norm
        except Exception as exc:
            raise ImportError("scipy is required for certified radius computation") from exc

        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if x.ndim == 0 or int(x.shape[0]) != 1:
            raise ValueError("certified_radius expects a single input (batch_size==1).")

        counts = None
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.base_model(self.transform(x))
                preds = logits.argmax(dim=1)
                if counts is None:
                    if logits.ndim != 2 or int(logits.size(0)) != 1:
                        raise ValueError(
                            "certified_radius expects logits with shape (1, C). "
                            f"Got shape={tuple(logits.shape)}"
                        )
                    counts = torch.zeros(int(logits.size(1)), device=x.device, dtype=torch.float32)
                counts.scatter_add_(0, preds.to(torch.int64), torch.ones_like(preds, dtype=counts.dtype))

        if counts is None or float(counts.sum().item()) <= 0.0:
            return 0.0

        p_a = float((counts.max() / counts.sum()).item())
        if p_a <= 0.5:
            return 0.0

        # Clamp away from {0,1} to keep ppf finite.
        eps = 1e-6
        p_a = max(0.5 + eps, min(p_a, 1.0 - eps))
        radius = float(self.sigma) * float(_norm.ppf(p_a))
        if (not math.isfinite(radius)) or radius < 0.0:
            return 0.0
        return float(radius)


class ATTransformDefense(DefenseWrapper):
    def __init__(self, base_model: nn.Module, noise_std: float = 0.05, device: str = "cuda"):
        spec = DefenseSpec(
            name="at_transform",
            domain="malware",
            obfuscation_types=[ObfuscationType.STOCHASTIC],
            params={"noise_std": float(noise_std)},
            is_stochastic=True,
            eot_samples=20,
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
    "TentDefense",
    "GradientRegularizationDefense",
    "SpatialSmoothingDefense",
    "CertifiedDefense",
    "ATTransformDefense",
    "TotalVariationDefense",
]
