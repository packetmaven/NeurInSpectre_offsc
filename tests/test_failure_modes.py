import math

import pytest
import torch
import torch.nn as nn

from neurinspectre.attacks.square import SquareAttack
from neurinspectre.characterization.defense_analyzer import DefenseAnalyzer, ObfuscationType
from neurinspectre.defenses.wrappers import RandomizedSmoothingDefense


def test_confidence_downweights_short_gradient_sequences():
    # Failure-mode: characterization was run with too few gradient steps.
    # We shouldn't crash, but we should downweight confidence.
    model = nn.Linear(4, 2)
    analyzer = DefenseAnalyzer(model, device="cpu", verbose=False)

    conf_short = analyzer._compute_confidence(
        etd_score=1.0,
        alpha_volterra=0.5,
        volterra_rmse=0.01,
        volterra_rmse_scaled=0.01,
        grad_variance=1e-3,
        grad_norm_mean=1e-2,
        jacobian_rank=1.0,
        n_grad=32,  # N < 64
    )
    conf_long = analyzer._compute_confidence(
        etd_score=1.0,
        alpha_volterra=0.5,
        volterra_rmse=0.01,
        volterra_rmse_scaled=0.01,
        grad_variance=1e-3,
        grad_norm_mean=1e-2,
        jacobian_rank=1.0,
        n_grad=128,
    )
    assert 0.0 <= conf_short <= 1.0
    assert 0.0 <= conf_long <= 1.0
    assert conf_short < conf_long


def test_volterra_fit_ok_blocks_rl_classification_on_degenerate_scale():
    # Failure-mode: gradient history has (near) zero variance, so scale-free RMSE
    # is undefined and Volterra-based classification should not be trusted.
    model = nn.Linear(4, 2)
    analyzer = DefenseAnalyzer(model, device="cpu", verbose=False)

    gradients = [torch.zeros(8).numpy() for _ in range(12)]
    alpha, rmse, rmse_scaled, info = analyzer._fit_volterra_kernel(gradients)
    assert isinstance(info, dict)
    assert math.isfinite(float(alpha))
    # rmse_scaled should be NaN when the sequence scale is degenerate.
    assert not math.isfinite(float(rmse_scaled)) or math.isnan(float(rmse_scaled))

    obf = analyzer._classify_obfuscation(
        etd_score=0.0,
        alpha_volterra=float(alpha),
        grad_variance=0.0,
        grad_norm_mean=1.0,
        stochastic_score=0.0,
        jacobian_rank=1.0,
        timescale=1.0,  # would normally help trigger RL_TRAINED
        spectral_signals={"rel_error_mean": 0.0, "norm_ratio_mean": 1.0, "norm_growth_fraction": 0.0},
        volterra_fit_ok=False,  # critical for this failure mode
    )
    assert ObfuscationType.RL_TRAINED not in obf


def test_volterra_high_rmse_scaled_downweights_confidence():
    model = nn.Linear(4, 2)
    analyzer = DefenseAnalyzer(model, device="cpu", verbose=False)

    conf = analyzer._compute_confidence(
        etd_score=1.0,
        alpha_volterra=0.5,
        volterra_rmse=1.0,
        volterra_rmse_scaled=100.0,  # intentionally huge
        grad_variance=1e-3,
        grad_norm_mean=1e-2,
        jacobian_rank=1.0,
        n_grad=128,
    )
    assert 0.0 <= conf <= 1.0
    assert conf < 1e-3


def test_square_attack_rejects_too_low_query_budget():
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match=r">=1000"):
        SquareAttack(model, eps=0.03, n_queries=999, device="cpu")

    # Boundary: allow exactly the minimum.
    SquareAttack(model, eps=0.03, n_queries=1000, device="cpu")


def test_randomized_smoothing_certified_radius_edge_cases():
    class ConstantLogits(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = int(x.shape[0])
            logits = torch.zeros((b, 2), device=x.device, dtype=torch.float32)
            logits[:, 0] = 10.0  # always class 0
            return logits

    defense = RandomizedSmoothingDefense(ConstantLogits(), sigma=0.25, n_samples=1, device="cpu")
    x = torch.zeros((1, 3, 8, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        defense.certified_radius(x, n_samples=0)

    with pytest.raises(ValueError, match="batch_size==1"):
        defense.certified_radius(torch.zeros((2, 3, 8, 8), dtype=torch.float32), n_samples=10)

    r = defense.certified_radius(x, n_samples=25)
    assert math.isfinite(float(r))
    assert r >= 0.0

