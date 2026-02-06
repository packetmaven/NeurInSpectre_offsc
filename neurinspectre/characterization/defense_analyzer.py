"""
Automated defense characterization and obfuscation detection.

This is Layer 1 of NeurInSpectre's pipeline. It collects gradient samples,
computes ETD scores, fits Volterra kernels, and recommends attack settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mathematical.krylov import analyze_krylov_projection
from ..mathematical.volterra import compute_volterra_correlation, fit_volterra_kernel


class ObfuscationType(Enum):
    """Defense obfuscation taxonomy."""

    SHATTERED = "shattered"
    STOCHASTIC = "stochastic"
    VANISHING = "vanishing"
    RL_TRAINED = "rl_trained"
    EXPLODING = "exploding"
    HYBRID = "hybrid"
    NONE = "none"


@dataclass
class DefenseCharacterization:
    """Complete defense characterization profile."""

    obfuscation_types: List[ObfuscationType]
    etd_score: float
    alpha_volterra: float
    gradient_variance: float
    jacobian_rank: float
    autocorr_timescale: float

    requires_bpda: bool
    requires_eot: bool
    requires_mapgd: bool

    recommended_eot_samples: int
    recommended_memory_length: int

    confidence: float
    metadata: Dict

    def __repr__(self) -> str:
        obf_str = ", ".join([o.value for o in self.obfuscation_types])
        return (
            "DefenseCharacterization(\n"
            f"  obfuscation=[{obf_str}],\n"
            f"  ETD={self.etd_score:.3f}, alpha={self.alpha_volterra:.3f},\n"
            f"  BPDA={self.requires_bpda}, EOT={self.requires_eot}, MA-PGD={self.requires_mapgd},\n"
            f"  confidence={self.confidence:.2f}\n"
            ")"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "obfuscation_types": [o.value for o in self.obfuscation_types],
            "etd_score": float(self.etd_score),
            "alpha_volterra": float(self.alpha_volterra),
            "gradient_variance": float(self.gradient_variance),
            "jacobian_rank": float(self.jacobian_rank),
            "autocorr_timescale": float(self.autocorr_timescale),
            "requires_bpda": bool(self.requires_bpda),
            "requires_eot": bool(self.requires_eot),
            "requires_mapgd": bool(self.requires_mapgd),
            "recommended_eot_samples": int(self.recommended_eot_samples),
            "recommended_memory_length": int(self.recommended_memory_length),
            "confidence": float(self.confidence),
            "metadata": self.metadata,
        }


class DefenseAnalyzer:
    """
    Automated defense characterization system.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        n_probe_images: int = 100,
        device: str = "cuda",
        verbose: bool = True,
        krylov_dim: int = 20,
    ):
        self.model = model.to(device)
        self.n_samples = int(n_samples)
        self.n_probe_images = int(n_probe_images)
        self.device = device
        self.verbose = verbose
        self.krylov_dim = int(krylov_dim)

        # Thresholds
        self.ETD_THRESHOLD_MODERATE = 0.3
        self.ETD_THRESHOLD_SEVERE = 0.6
        self.ALPHA_RL_THRESHOLD = 0.7
        self.ALPHA_SHATTERED_THRESHOLD = 0.3
        self.GRAD_NORM_SHATTERED_THRESHOLD = 1e-5
        self.VARIANCE_SHATTERED_THRESHOLD = 1e-6
        self.VARIANCE_STOCHASTIC_THRESHOLD = 1e-5
        self.RANK_VANISHING_THRESHOLD = 0.5
        self.AUTOCORR_RL_THRESHOLD = 0.5
        # Spectral/Krylov heuristics (Paper Section 3.1.2)
        self.KRYLOV_REL_ERROR_SHATTERED = 0.5
        self.KRYLOV_NORM_RATIO_VANISHING = 0.3
        self.KRYLOV_NORM_GROWTH_SHATTERED = 0.3

    def characterize(
        self,
        data_loader: torch.utils.data.DataLoader,
        eps: float = 8 / 255,
    ) -> DefenseCharacterization:
        if self.verbose:
            print("[DefenseAnalyzer] Starting characterization...")
            print(
                f"  Collecting {self.n_samples} gradient samples from {self.n_probe_images} images"
            )

        gradients, images, labels = self._collect_gradient_samples(data_loader, eps)

        if self.verbose:
            print("[DefenseAnalyzer] Computing ETD score...")
        etd_score = self._compute_etd_score(images, labels, eps)

        if self.verbose:
            print("[DefenseAnalyzer] Fitting Volterra kernel...")
        alpha_volterra, volterra_rmse = self._fit_volterra_kernel(gradients)

        if self.verbose:
            print("[DefenseAnalyzer] Analyzing gradient statistics...")
        grad_variance = self._compute_gradient_variance(images, labels)
        stochastic_score = self._compute_stochastic_score(images, labels)
        grad_norm_mean = self._compute_gradient_norm_mean(gradients)
        jacobian_rank = self._estimate_jacobian_rank(images, labels)
        autocorr, timescale = self._compute_autocorrelation(gradients)
        spectral_signals = self._compute_spectral_signals(gradients)

        if grad_norm_mean < self.GRAD_NORM_SHATTERED_THRESHOLD:
            alpha_volterra = min(alpha_volterra, 0.2)

        if self.verbose:
            print("[DefenseAnalyzer] Classifying obfuscation mechanisms...")
        obfuscation_types = self._classify_obfuscation(
            etd_score,
            alpha_volterra,
            grad_variance,
            grad_norm_mean,
            stochastic_score,
            jacobian_rank,
            timescale,
            spectral_signals,
        )

        requires_bpda, requires_eot, requires_mapgd = self._recommend_attacks(
            obfuscation_types, etd_score, alpha_volterra
        )

        eot_samples = self._recommend_eot_samples(grad_variance, requires_eot)
        memory_length = self._recommend_memory_length(alpha_volterra, requires_mapgd)

        confidence = self._compute_confidence(
            etd_score,
            alpha_volterra,
            volterra_rmse,
            grad_variance,
            grad_norm_mean,
            jacobian_rank,
        )

        characterization = DefenseCharacterization(
            obfuscation_types=obfuscation_types,
            etd_score=etd_score,
            alpha_volterra=alpha_volterra,
            gradient_variance=grad_variance,
            jacobian_rank=jacobian_rank,
            autocorr_timescale=timescale,
            requires_bpda=requires_bpda,
            requires_eot=requires_eot,
            requires_mapgd=requires_mapgd,
            recommended_eot_samples=eot_samples,
            recommended_memory_length=memory_length,
            confidence=confidence,
            metadata={
                "volterra_rmse": volterra_rmse,
                "autocorrelation": autocorr.tolist()
                if isinstance(autocorr, np.ndarray)
                else autocorr,
                "n_samples_analyzed": len(gradients),
                "grad_norm_mean": grad_norm_mean,
                "stochastic_score": stochastic_score,
                "krylov_rel_error_mean": spectral_signals.get("rel_error_mean"),
                "krylov_norm_ratio_mean": spectral_signals.get("norm_ratio_mean"),
                "krylov_norm_growth_fraction": spectral_signals.get("norm_growth_fraction"),
                "krylov_dissipation_anomaly_score": spectral_signals.get("dissipation_anomaly_score"),
                "krylov_steps": spectral_signals.get("steps"),
            },
        )

        if self.verbose:
            print("\n[DefenseAnalyzer] Characterization complete:")
            print(characterization)

        return characterization

    def _collect_gradient_samples(
        self,
        data_loader: torch.utils.data.DataLoader,
        eps: float,
    ) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
        all_images = []
        all_labels = []

        for x_batch, y_batch in data_loader:
            all_images.append(x_batch)
            all_labels.append(y_batch)
            if sum(img.size(0) for img in all_images) >= self.n_probe_images:
                break

        images = torch.cat(all_images)[: self.n_probe_images].to(self.device)
        labels = torch.cat(all_labels)[: self.n_probe_images].to(self.device)

        x_min = float(images.min().item())
        x_max = float(images.max().item())
        delta = torch.zeros_like(images).uniform_(-eps, eps)
        delta = torch.clamp(images + delta, x_min, x_max) - images
        delta.requires_grad = True

        gradients: List[np.ndarray] = []

        for _ in range(self.n_samples):
            logits = self.model(images + delta)
            loss = F.cross_entropy(logits, labels)

            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            if delta.grad is None:
                grad = torch.zeros_like(delta)
            else:
                grad = delta.grad.detach().clone()
            grad_flat = grad.view(grad.size(0), -1).cpu().numpy()
            gradients.append(grad_flat.mean(axis=0))

            with torch.no_grad():
                delta.data = delta + (2 / 255) * grad.sign()
                delta.data = torch.clamp(delta, -eps, eps)
                delta.data = torch.clamp(images + delta, x_min, x_max) - images

            delta.requires_grad = True

        return gradients, images, labels

    def collect_gradient_samples(
        self,
        data_loader: torch.utils.data.DataLoader,
        eps: float,
    ) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
        return self._collect_gradient_samples(data_loader, eps)

    def _compute_etd_score(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        eps: float,
    ) -> float:
        from ..attacks.pgd import PGD

        was_training = self.model.training
        pgd = PGD(self.model, eps=eps, steps=40, device=self.device)
        x_adv = pgd(images, labels)

        dist_adv = (images - x_adv).pow(2).view(images.size(0), -1).sum(dim=1)
        view_shape = (-1,) + (1,) * (images.ndim - 1)
        directions = (x_adv - images) / (dist_adv.sqrt().view(view_shape) + 1e-8)
        x_min = float(images.min().item())
        x_max = float(images.max().item())

        cf_distances = []
        for i in range(images.size(0)):
            x_i = images[i : i + 1]
            y_i = labels[i : i + 1]
            dir_i = directions[i : i + 1]

            low, high = 0.0, eps
            for _ in range(20):
                mid = (low + high) / 2
                x_test = x_i + mid * dir_i
                x_test = torch.clamp(x_test, x_min, x_max)
                with torch.no_grad():
                    pred = self.model(x_test).argmax(1)
                if pred != y_i:
                    high = mid
                else:
                    low = mid

            cf_distances.append(((high + low) / 2) ** 2)

        cf_distances = torch.tensor(cf_distances, device=self.device)
        etd_ratios = cf_distances / (dist_adv + 1e-8)
        etd_score = etd_ratios.mean().item()
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        return float(np.clip(etd_score, 0.0, 2.0))

    def _fit_volterra_kernel(self, gradients: List[np.ndarray]) -> Tuple[float, float]:
        grad_array = np.array(gradients)
        try:
            kernel, rmse, info = fit_volterra_kernel(
                grad_array,
                kernel_type="power_law",
                method="L-BFGS-B",
                verbose=False,
            )
            if not info.get("success", False):
                warnings.warn("Volterra fitting did not converge, using alpha=0.5")
                return 0.5, np.nan
            alpha = float(kernel.alpha)
        except Exception as exc:
            warnings.warn(f"Volterra fitting failed: {exc}, using alpha=0.5")
            return 0.5, np.nan

        return alpha, float(rmse)

    def _compute_gradient_variance(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        if images.size(0) == 0:
            return 0.0
        delta = torch.zeros_like(images, requires_grad=True)
        logits = self.model(images + delta)
        loss = F.cross_entropy(logits, labels)
        self.model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        if delta.grad is None:
            grad = torch.zeros_like(delta)
        else:
            grad = delta.grad.detach()
        grad_flat = grad.view(grad.size(0), -1)
        var = grad_flat.var(unbiased=False).item()
        scale = float(grad_flat.shape[1]) * 1_000_000.0
        return float(var * scale)

    def _compute_stochastic_score(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n_repeats: int = 5,
    ) -> float:
        if images.size(0) == 0:
            return 0.0
        logits_list = []
        for _ in range(int(n_repeats)):
            logits = self.model(images)
            logits_list.append(logits.detach())
        logit_stack = torch.stack(logits_list, dim=0)
        logit_variance = logit_stack.var(dim=0).mean().item()
        return float(logit_variance)

    def _compute_gradient_norm_mean(self, gradients: List[np.ndarray]) -> float:
        if len(gradients) == 0:
            return 0.0
        grad_array = np.array(gradients)
        norms = np.linalg.norm(grad_array, axis=1)
        return float(np.mean(norms))

    def _estimate_jacobian_rank(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n_samples: int = 10,
    ) -> float:
        n = int(min(n_samples, images.size(0)))
        if n <= 1:
            return 0.0
        indices = torch.randperm(images.size(0))[:n]
        x_sample = images[indices].requires_grad_(True)
        y_sample = labels[indices]

        logits = self.model(x_sample)
        loss = F.cross_entropy(logits, y_sample)
        self.model.zero_grad()
        loss.backward()

        if x_sample.grad is None:
            grad = torch.zeros_like(x_sample)
        else:
            grad = x_sample.grad.detach()
        grad_flat = grad.view(n, -1).cpu().numpy()
        if not np.any(grad_flat):
            return 0.0
        try:
            _, s, _ = np.linalg.svd(grad_flat, full_matrices=False)
            s_norm = s / (s.sum() + 1e-8)
            effective_rank = np.exp(-np.sum(s_norm * np.log(s_norm + 1e-8)))
            effective_rank_normalized = effective_rank / n
        except np.linalg.LinAlgError:
            effective_rank_normalized = 0.1

        return float(effective_rank_normalized)

    def _compute_autocorrelation(
        self,
        gradients: List[np.ndarray],
        max_lag: int = 20,
    ) -> Tuple[np.ndarray, float]:
        grad_array = np.array(gradients)
        autocorr, timescale = compute_volterra_correlation(grad_array, max_lag=max_lag)
        return autocorr, timescale

    def _compute_spectral_signals(self, gradients: List[np.ndarray]) -> Dict[str, float]:
        if len(gradients) < 2:
            return {
                "rel_error_mean": 0.0,
                "norm_ratio_mean": 1.0,
                "norm_growth_fraction": 0.0,
                "dissipation_anomaly_score": 0.0,
                "steps": 0,
            }
        grad_array = np.array(gradients, dtype=np.float64)
        try:
            summary, _steps, _eigvals = analyze_krylov_projection(
                grad_array,
                krylov_dim=self.krylov_dim,
                dt=1.0,
                damping=0.1,
                steps=min(25, max(2, grad_array.shape[0] - 1)),
                stride=1,
            )
        except Exception as exc:
            warnings.warn(f"Krylov analysis failed: {exc}")
            return {
                "rel_error_mean": 0.0,
                "norm_ratio_mean": 1.0,
                "norm_growth_fraction": 0.0,
                "dissipation_anomaly_score": 0.0,
                "steps": 0,
            }
        reconstruction = summary.get("reconstruction_error", {}) or {}
        dissipation = summary.get("dissipation", {}) or {}
        return {
            "rel_error_mean": float(reconstruction.get("mean", 0.0)),
            "norm_ratio_mean": float(dissipation.get("norm_ratio_mean", 1.0)),
            "norm_growth_fraction": float(dissipation.get("norm_growth_fraction", 0.0)),
            "dissipation_anomaly_score": float(dissipation.get("dissipation_anomaly_score", 0.0)),
            "steps": int(summary.get("steps_analyzed", 0)),
        }

    def _classify_obfuscation(
        self,
        etd_score: float,
        alpha_volterra: float,
        grad_variance: float,
        grad_norm_mean: float,
        stochastic_score: float,
        jacobian_rank: float,
        timescale: float,
        spectral_signals: Dict[str, float],
    ) -> List[ObfuscationType]:
        obfuscation_types: List[ObfuscationType] = []

        if grad_norm_mean < self.GRAD_NORM_SHATTERED_THRESHOLD:
            obfuscation_types.append(ObfuscationType.SHATTERED)
            if self.verbose:
                print(
                    "  [DETECTED] Shattered gradients "
                    f"(alpha={alpha_volterra:.3f}, grad_norm={grad_norm_mean:.6f})"
                )

        if stochastic_score > self.VARIANCE_STOCHASTIC_THRESHOLD:
            obfuscation_types.append(ObfuscationType.STOCHASTIC)
            if self.verbose:
                print(
                    "  [DETECTED] Stochastic obfuscation "
                    f"(score={stochastic_score:.6f} > {self.VARIANCE_STOCHASTIC_THRESHOLD:.6f})"
                )

        if jacobian_rank < self.RANK_VANISHING_THRESHOLD:
            obfuscation_types.append(ObfuscationType.VANISHING)
            if self.verbose:
                print(
                    f"  [DETECTED] Vanishing gradients (rank={jacobian_rank:.3f} < 0.5)"
                )

        rel_error_mean = float(spectral_signals.get("rel_error_mean", 0.0))
        norm_ratio_mean = float(spectral_signals.get("norm_ratio_mean", 1.0))
        norm_growth_fraction = float(spectral_signals.get("norm_growth_fraction", 0.0))

        if (
            rel_error_mean >= self.KRYLOV_REL_ERROR_SHATTERED
            or norm_growth_fraction >= self.KRYLOV_NORM_GROWTH_SHATTERED
        ):
            if ObfuscationType.SHATTERED not in obfuscation_types:
                obfuscation_types.append(ObfuscationType.SHATTERED)
                if self.verbose:
                    print(
                        "  [DETECTED] Spectral anomaly (Krylov rel_error="
                        f"{rel_error_mean:.3f}, growth={norm_growth_fraction:.3f})"
                    )

        if norm_ratio_mean <= self.KRYLOV_NORM_RATIO_VANISHING:
            if ObfuscationType.VANISHING not in obfuscation_types:
                obfuscation_types.append(ObfuscationType.VANISHING)
                if self.verbose:
                    print(
                        "  [DETECTED] Spectral vanishing (Krylov norm_ratio="
                        f"{norm_ratio_mean:.3f})"
                    )

        if (
            self.ALPHA_SHATTERED_THRESHOLD <= alpha_volterra < self.ALPHA_RL_THRESHOLD
            and timescale > self.AUTOCORR_RL_THRESHOLD
        ):
            obfuscation_types.append(ObfuscationType.RL_TRAINED)
            if self.verbose:
                print(
                    f"  [DETECTED] RL-trained obfuscation (alpha={alpha_volterra:.3f}, "
                    f"timescale={timescale:.2f})"
                )

        if len(obfuscation_types) == 0:
            obfuscation_types.append(ObfuscationType.NONE)
            if self.verbose:
                print("  [NO OBFUSCATION DETECTED]")

        return obfuscation_types

    def _recommend_attacks(
        self,
        obfuscation_types: List[ObfuscationType],
        etd_score: float,
        alpha_volterra: float,
    ) -> Tuple[bool, bool, bool]:
        if len(obfuscation_types) == 1 and obfuscation_types[0] == ObfuscationType.NONE:
            return False, False, False
        requires_bpda = (
            ObfuscationType.SHATTERED in obfuscation_types
            or etd_score >= self.ETD_THRESHOLD_SEVERE
        )
        requires_eot = ObfuscationType.STOCHASTIC in obfuscation_types
        requires_mapgd = (
            ObfuscationType.RL_TRAINED in obfuscation_types
            or alpha_volterra < self.ALPHA_RL_THRESHOLD
        )
        return requires_bpda, requires_eot, requires_mapgd

    def _recommend_eot_samples(self, grad_variance: float, requires_eot: bool) -> int:
        if not requires_eot:
            return 1
        n_samples = int(20 * grad_variance / self.VARIANCE_STOCHASTIC_THRESHOLD)
        return int(np.clip(n_samples, 10, 50))

    def _recommend_memory_length(self, alpha_volterra: float, requires_mapgd: bool) -> int:
        if not requires_mapgd:
            return 1
        from ..attacks.memory_gradient import memory_length_schedule

        return memory_length_schedule(alpha_volterra, max_length=50)

    def _compute_confidence(
        self,
        etd_score: float,
        alpha_volterra: float,
        volterra_rmse: float,
        grad_variance: float,
        grad_norm_mean: float,
        jacobian_rank: float,
    ) -> float:
        if np.isnan(volterra_rmse):
            volterra_conf = 0.5
        else:
            volterra_conf = np.exp(-volterra_rmse * 10)
        etd_conf = 1.0 - np.exp(-abs(etd_score - 0.5) * 3)
        grad_scale = max(self.GRAD_NORM_SHATTERED_THRESHOLD * 5.0, 1e-12)
        grad_conf = 1.0 - np.exp(-float(grad_norm_mean) / grad_scale)
        rank_scale = max(self.RANK_VANISHING_THRESHOLD, 1e-6)
        rank_conf = float(np.clip(jacobian_rank / rank_scale, 0.0, 1.0))
        confidence = (
            np.sqrt(volterra_conf * etd_conf)
            * max(grad_conf, 1e-3)
            * max(rank_conf, 1e-3)
        )
        return float(np.clip(confidence, 0.0, 1.0))


def quick_characterize(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> DefenseCharacterization:
    """Quick defense characterization with default settings."""
    analyzer = DefenseAnalyzer(model, device=device, verbose=True)
    return analyzer.characterize(data_loader)
