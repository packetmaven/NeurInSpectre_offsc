"""
DefenseFactory for creating defended models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .wrappers import (
    ATTransformDefense,
    BitDepthReductionDefense,
    CertifiedDefense,
    DefensiveDistillationDefense,
    EnsembleDiversityDefense,
    FeatureSqueezingDefense,
    GradientRegularizationDefense,
    JPEGCompressionDefense,
    RLObfuscationDefense,
    RandomizedSmoothingDefense,
    RandomNoiseDefense,
    RandomPadCropDefense,
    SpatialSmoothingDefense,
    TentDefense,
    ThermometerEncodingDefense,
    TotalVariationDefense,
)


class DefenseFactory:
    @staticmethod
    def create_defense(defense_name: str, base_model: nn.Module, params: Dict[str, Any]) -> nn.Module:
        key = str(defense_name).lower()
        device = _infer_device(base_model, params)

        if key == "jpeg_compression":
            return JPEGCompressionDefense(base_model, quality=int(params.get("quality", 75)), device=device)
        if key == "bit_depth_reduction":
            return BitDepthReductionDefense(base_model, bits=int(params.get("bits", 4)), device=device)
        if key == "thermometer_encoding":
            return ThermometerEncodingDefense(base_model, levels=int(params.get("levels", 16)), device=device)
        if key == "random_pad_crop":
            return RandomPadCropDefense(base_model, pad_size=int(params.get("max_pad", 4)), device=device)
        if key == "randomized_smoothing":
            sigma = float(params.get("sigma", 0.25))
            n_samples = int(params.get("n_samples", 100))
            return RandomizedSmoothingDefense(base_model, sigma=sigma, n_samples=n_samples, device=device)
        if key == "spatial_smoothing":
            return SpatialSmoothingDefense(
                base_model,
                kernel_size=int(params.get("kernel_size", 3)),
                sigma=float(params.get("sigma", 1.0)),
                device=device,
            )
        if key == "feature_squeezing":
            return FeatureSqueezingDefense(
                base_model,
                bit_depth=int(params.get("bit_depth", 5)),
                kernel_size=int(params.get("median_filter_size", params.get("filter_size", 3))),
                device=device,
            )
        if key == "defensive_distillation":
            temperature = float(params.get("temperature", 20.0))
            return DefensiveDistillationDefense(base_model, temperature=temperature, device=device)
        if key == "at_transform":
            return ATTransformDefense(base_model, noise_std=float(params.get("noise_std", 0.05)), device=device)
        if key == "ensemble_diversity":
            return _ensemble_diversity_defense(base_model, params, device=device)
        if key == "gradient_regularization":
            return GradientRegularizationDefense(
                base_model,
                lambda_grad=float(params.get("lambda_grad", 1.0)),
                device=device,
            )
        if key == "certified_defense":
            sigma = float(params.get("sigma", 0.5))
            n_samples = int(params.get("n_samples", params.get("num_samples", 100)))
            return CertifiedDefense(base_model, sigma=sigma, n_samples=n_samples, device=device)
        if key == "random_noise":
            return RandomNoiseDefense(base_model, std=float(params.get("std", 0.05)), device=device)
        if key == "rl_obfuscation":
            return RLObfuscationDefense(
                base_model,
                bits=int(params.get("bits", 6)),
                std=float(params.get("std", 0.08)),
                alpha=float(params.get("alpha", 0.60)),
                n_samples=int(params.get("n_samples", 32)),
                device=device,
            )
        if key == "tent":
            return TentDefense(
                base_model,
                lr=float(params.get("lr", 1e-3)),
                steps=int(params.get("steps", 1)),
                reset_each_forward=bool(params.get("reset_each_forward", False)),
                device=device,
            )
        if key == "total_variation":
            return TotalVariationDefense(
                base_model,
                weight=float(params.get("weight", 0.1)),
                n_iter=int(params.get("n_iter", 10)),
                device=device,
            )

        raise ValueError(f"Unknown defense: {defense_name}")


def _ensemble_diversity_defense(base_model: nn.Module, params: Dict[str, Any], device: str) -> nn.Module:
    members_cfg = params.get("members")
    if not members_cfg:
        raise ValueError("ensemble_diversity requires 'members' list in params")

    models: List[nn.Module] = []
    for member in members_cfg:
        path = member.get("path") or member.get("checkpoint_path") or member.get("weights_path")
        if path:
            model = _load_model_from_path(Path(path), device=member.get("device", device))
        else:
            from ..models.factory import ModelFactory

            member_kwargs = {
                k: v
                for k, v in member.items()
                if k not in {"domain", "model_name", "training_type", "device"}
            }
            model = ModelFactory.load_model(
                domain=member["domain"],
                model_name=member["model_name"],
                training_type=member.get("training_type", "standard"),
                device=member.get("device", device),
                **member_kwargs,
            )
        models.append(model)

    aggregation = params.get("aggregation", params.get("voting", "average"))
    # Config ergonomics: paper/spec configs often say "majority" voting.
    # The runnable defense uses "vote" (argmax majority vote over members).
    agg_norm = str(aggregation).lower().strip().replace("-", "_")
    if agg_norm in {"majority", "majority_vote", "plurality"}:
        aggregation = "vote"
    elif agg_norm in {"avg", "mean"}:
        aggregation = "average"
    return EnsembleDiversityDefense(models=models, aggregation=aggregation, device=device)


def _infer_device(model: nn.Module, params: Dict[str, Any]) -> str:
    if "device" in params:
        return str(params["device"])
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _load_model_from_path(path: Path, device: str) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()
        return model
    if isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        model = obj["model"].to(device)
        model.eval()
        return model

    raise ValueError(
        "Unsupported checkpoint format for ensemble member. "
        "Provide a TorchScript model or a serialized nn.Module."
    )
