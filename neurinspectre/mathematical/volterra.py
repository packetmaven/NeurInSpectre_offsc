"""
Volterra memory analysis (Layer 2).

This module implements a proof-of-concept Volterra integral equation fit consistent with the
NeurInSpectre three-layer framework described in the project README and the Packetmaven blog.

Primary output features:
- volterra_alpha: fitted power-law memory exponent (0 < alpha < 1)
- volterra_c: fitted kernel strength (c > 0)
- volterra_rmse: fit error between observed and predicted series
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


VolterraNormalize = Literal["none", "by_y0", "by_mean", "by_median", "by_rms"]


@dataclass(frozen=True)
class VolterraFitResult:
    """Result of fitting a Volterra power-law kernel."""

    alpha: float
    c: float
    rmse: float
    rmse_scaled: float
    n: int
    dt: float
    normalize: VolterraNormalize
    scale: float
    success: bool
    message: str
    nit: int
    nfev: int


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return x


def _compute_scale(y: np.ndarray, normalize: VolterraNormalize) -> float:
    y = _finite_1d(y)
    eps = 1e-12
    if y.size == 0:
        return 1.0
    if normalize == "none":
        return 1.0
    if normalize == "by_y0":
        return float(max(abs(float(y[0])), eps))
    if normalize == "by_mean":
        return float(max(abs(float(np.mean(y))), eps))
    if normalize == "by_median":
        return float(max(abs(float(np.median(y))), eps))
    if normalize == "by_rms":
        return float(max(float(np.sqrt(np.mean(y * y))), eps))
    # Defensive default
    return 1.0


def _volterra_power_law_kernel_weights(n: int, *, alpha: float, dt: float) -> np.ndarray:
    """
    Precompute lag weights w[k-1] = (k*dt)^(alpha-1) / Gamma(alpha) for k=1..n-1.
    """
    from scipy.special import gamma

    n = int(n)
    if n <= 1:
        return np.zeros((0,), dtype=np.float64)
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0,1) for the power-law kernel.")
    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be > 0.")

    # lags: 1..n-1
    k = np.arange(1, n, dtype=np.float64)
    # Avoid any accidental zeros
    lag = np.maximum(k * dt, 1e-12)
    inv_gamma = 1.0 / float(gamma(a))
    w = (lag ** (a - 1.0)) * inv_gamma
    return w.astype(np.float64, copy=False)


def predict_volterra_power_law(
    y0: float,
    *,
    alpha: float,
    c: float,
    n: int,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Solve a Volterra equation of the second kind (PoC discretization):

        y(t) = y0 + ∫_0^t K(t,s) y(s) ds

    with power-law kernel:

        K(t,s) = c * (t-s)^(alpha-1) / Gamma(alpha),  0<alpha<1, c>0

    Discretization (trapezoid rule, causal convention K(t,s)=0 for s>=t):

        y_i = y0 + dt * Σ_{j=0}^{i-1} 0.5 * ( K(t_i, t_j) y_j + K(t_i, t_{j+1}) y_{j+1} )

    with the convention K(t_i, t_i)=0 so the last term (j=i-1, right endpoint) contributes 0.

    This is solved sequentially for y_pred[0..n-1].
    """
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)

    a = float(alpha)
    c = float(c)
    if not np.isfinite(a) or not np.isfinite(c):
        raise ValueError("alpha and c must be finite.")
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if c <= 0.0:
        raise ValueError("c must be > 0.")

    y0 = float(y0)
    dt = float(dt)

    y_pred = np.zeros((n,), dtype=np.float64)
    y_pred[0] = y0

    if n == 1:
        return y_pred

    w = _volterra_power_law_kernel_weights(n, alpha=a, dt=dt)  # w[k-1] corresponds to lag k*dt

    # Sequential solve: O(n^2). For CLI use we keep n modest (can downsample upstream).
    for i in range(1, n):
        # Trapezoid discretization can be expressed as two dot-products over known y:
        #   Σ_{j=0}^{i-1} K(i,j) y[j]    uses lags k=i-j in {i,...,1}
        #   Σ_{j=0}^{i-2} K(i,j+1) y[j+1] uses lags k'=i-(j+1) in {i-1,...,1}
        w1 = w[i - 1 :: -1]  # length i: lags i..1
        term1 = float(np.dot(w1, y_pred[:i]))

        if i >= 2:
            w2 = w[i - 2 :: -1]  # length i-1: lags i-1..1
            term2 = float(np.dot(w2, y_pred[1:i]))
        else:
            term2 = 0.0

        integral = 0.5 * dt * c * (term1 + term2)
        y_pred[i] = y0 + integral

        # Early bailout on divergence to keep optimization stable.
        if not np.isfinite(y_pred[i]) or abs(y_pred[i]) > 1e12:
            # Fill rest with inf to signal failure.
            y_pred[i:] = np.inf
            break

    return y_pred


def fit_volterra_power_law(
    y_obs: np.ndarray,
    *,
    dt: float = 1.0,
    normalize: VolterraNormalize = "by_y0",
    alpha_bounds: Tuple[float, float] = (0.05, 0.995),
    c_bounds: Tuple[float, float] = (1e-6, 10.0),
    maxiter: int = 250,
    seed: int = 42,
) -> VolterraFitResult:
    """
    Fit (alpha, c) for the power-law Volterra kernel by minimizing RMSE.

    Notes:
    - This is a proof-of-concept numerical fit meant to produce stable, comparable parameters.
    - Real deployments should calibrate normalization + thresholds on domain-specific baselines.
    """
    y_raw = np.asarray(y_obs, dtype=np.float64).reshape(-1)
    if y_raw.size < 3:
        raise ValueError("Need at least 3 points to fit a Volterra kernel.")
    if not np.isfinite(y_raw).all():
        # keep only finite prefix (common when users concatenate runs with padding)
        y_raw = _finite_1d(y_raw)
    if y_raw.size < 3:
        raise ValueError("Need at least 3 finite points to fit a Volterra kernel.")

    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be > 0.")

    scale = _compute_scale(y_raw, normalize)
    y = (y_raw / scale).astype(np.float64, copy=False)

    y0 = float(y[0])
    n = int(y.size)

    a_lo, a_hi = float(alpha_bounds[0]), float(alpha_bounds[1])
    c_lo, c_hi = float(c_bounds[0]), float(c_bounds[1])
    if not (0.0 < a_lo < a_hi < 1.0):
        raise ValueError("alpha_bounds must satisfy 0 < lo < hi < 1.")
    if not (0.0 < c_lo < c_hi):
        raise ValueError("c_bounds must satisfy 0 < lo < hi.")

    # Deterministic init for reproducibility
    rng = np.random.default_rng(int(seed))
    x0 = np.array(
        [
            float(np.clip(0.90 + 0.02 * rng.normal(), a_lo, a_hi)),
            float(np.clip(1.00 + 0.10 * rng.normal(), c_lo, c_hi)),
        ],
        dtype=np.float64,
    )

    def _rmse_for(x: np.ndarray) -> float:
        a = float(x[0])
        c = float(x[1])
        try:
            y_pred = predict_volterra_power_law(y0, alpha=a, c=c, n=n, dt=dt)
        except Exception:
            return 1e9
        if not np.isfinite(y_pred).all():
            return 1e9
        err = y - y_pred
        return float(np.sqrt(np.mean(err * err)))

    from scipy.optimize import minimize

    res = minimize(
        _rmse_for,
        x0,
        method="L-BFGS-B",
        bounds=[(a_lo, a_hi), (c_lo, c_hi)],
        options={"maxiter": int(maxiter)},
    )

    alpha_hat = float(res.x[0])
    c_hat = float(res.x[1])

    y_pred = predict_volterra_power_law(y0, alpha=alpha_hat, c=c_hat, n=n, dt=dt)
    rmse_scaled = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    y_pred_raw = y_pred * scale
    rmse_raw = float(np.sqrt(np.mean((y_raw - y_pred_raw) ** 2)))

    return VolterraFitResult(
        alpha=alpha_hat,
        c=c_hat,
        rmse=rmse_raw,
        rmse_scaled=rmse_scaled,
        n=n,
        dt=dt,
        normalize=normalize,
        scale=float(scale),
        success=bool(getattr(res, "success", False)),
        message=str(getattr(res, "message", "")),
        nit=int(getattr(res, "nit", 0) or 0),
        nfev=int(getattr(res, "nfev", 0) or 0),
    )


