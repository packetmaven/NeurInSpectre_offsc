"""Layer 3 ETD/Krylov characterization wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..mathematical.krylov import analyze_krylov_projection


def compute_etd_features(
    seq: np.ndarray,
    *,
    krylov_dim: int = 30,
    dt: float = 1.0,
    damping: float = 0.1,
    steps: int = 25,
    stride: int = 1,
) -> dict[str, Any]:
    """
    Compute ETD/Krylov summary features for a sequence.
    """
    summary, step_results, _eigvals0 = analyze_krylov_projection(
        seq,
        krylov_dim=int(krylov_dim),
        dt=float(dt),
        damping=float(damping),
        steps=int(steps),
        stride=int(stride),
    )
    reconstruction = summary.get("reconstruction_error", {}) or {}
    dissipation = summary.get("dissipation", {}) or {}
    return {
        "krylov_rel_error_mean": float(reconstruction.get("mean", 0.0)),
        "krylov_rel_error_max": float(reconstruction.get("max", 0.0)),
        "krylov_norm_ratio_mean": float(dissipation.get("norm_ratio_mean", 0.0)),
        "krylov_norm_ratio_max": float(dissipation.get("norm_ratio_max", 0.0)),
        "krylov_norm_growth_fraction": float(dissipation.get("norm_growth_fraction", 0.0)),
        "krylov_dissipation_anomaly_score": float(
            dissipation.get("dissipation_anomaly_score", 0.0)
        ),
        "krylov_steps": int(summary.get("steps_analyzed", len(step_results))),
    }
