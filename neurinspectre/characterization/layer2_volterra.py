"""Layer 2 Volterra memory characterization wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..mathematical.volterra import fit_volterra_power_law


def fit_volterra_features(seq: np.ndarray, *, dt: float = 1.0, normalize: str = "by_y0") -> dict[str, Any]:
    """
    Fit a power-law Volterra kernel and return alpha/c/RMSE features.
    """
    arr = np.asarray(seq)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    result = fit_volterra_power_law(arr, dt=float(dt), normalize=normalize)
    return {
        "volterra_alpha": float(result.alpha),
        "volterra_c": float(result.c),
        "volterra_rmse": float(result.rmse),
        "volterra_rmse_scaled": float(result.rmse_scaled),
        "volterra_success": bool(result.success),
    }
