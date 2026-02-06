"""
Volterra memory analysis (Layer 2).

This module implements Volterra integral equation tooling consistent with the
NeurInSpectre framework and the Packetmaven Volterra discussion.

Primary output features:
- volterra_alpha: fitted power-law memory exponent (0 < alpha < 1)
- volterra_c: fitted kernel strength (c > 0)
- volterra_rmse: fit error between observed and predicted series

Additional utilities:
- VolterraKernel classes (power_law, exponential, uniform)
- fit_volterra_kernel for kernel fitting on gradient history
- compute_volterra_correlation for temporal autocorrelation analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import warnings

import numpy as np
from scipy import special
from scipy.optimize import differential_evolution, minimize


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


class VolterraKernel:
    """
    Base class for Volterra memory kernels.

    A kernel K(t,s) defines how the state at time s influences time t.
    Different kernel types capture different memory patterns.
    """

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, t: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Evaluate kernel K(t,s) for given time points.

        Args:
            t: Current time points (n,)
            s: Past time points (m,)

        Returns:
            Kernel matrix (n, m) where entry (i,j) is K(t[i], s[j])
        """
        raise NotImplementedError

    def __call__(self, t: np.ndarray, s: np.ndarray) -> np.ndarray:
        return self.evaluate(t, s)


class PowerLawKernel(VolterraKernel):
    """
    Power-law kernel: K(t,s) = c * (t-s)^(alpha-1) / Gamma(alpha)

    This kernel exhibits long-range temporal correlations and arises
    naturally from fractional calculus.
    """

    def __init__(self, alpha: float = 0.5, c: float = 1.0):
        super().__init__("power_law")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if c <= 0:
            raise ValueError(f"amplitude c must be positive, got {c}")
        self.alpha = float(alpha)
        self.c = float(c)
        self.gamma_alpha = special.gamma(self.alpha)

    def evaluate(self, t: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Evaluate K(t,s) = c * (t-s)^(alpha-1) / Gamma(alpha) for t > s."""
        t_mat = t[:, np.newaxis]
        s_mat = s[np.newaxis, :]
        tau = t_mat - s_mat
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            kernel = np.where(
                tau > 0,
                self.c * np.power(tau, self.alpha - 1.0) / self.gamma_alpha,
                0.0,
            )
        return np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_weights(self, memory_length: int) -> np.ndarray:
        """
        Compute normalized weights for discrete gradient history.

        Weights: w_i = (i+1)^(alpha-1) / sum_j (j)^(alpha-1)
        Returned in descending order (most recent first).
        """
        if memory_length <= 0:
            raise ValueError(f"memory_length must be positive, got {memory_length}")
        indices = np.arange(1, memory_length + 1, dtype=np.float64)
        raw_weights = np.power(indices, self.alpha - 1.0)
        weights = raw_weights / np.sum(raw_weights)
        return weights.copy()

    def __repr__(self) -> str:
        return f"PowerLawKernel(alpha={self.alpha:.3f}, c={self.c:.3f})"


class ExponentialKernel(VolterraKernel):
    """
    Exponential decay kernel: K(t,s) = lambda * exp(-lambda * (t-s))
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__("exponential")
        if lambda_ <= 0:
            raise ValueError(f"lambda must be positive, got {lambda_}")
        self.lambda_ = float(lambda_)

    def evaluate(self, t: np.ndarray, s: np.ndarray) -> np.ndarray:
        t_mat = t[:, np.newaxis]
        s_mat = s[np.newaxis, :]
        tau = t_mat - s_mat
        return np.where(
            tau > 0,
            self.lambda_ * np.exp(-self.lambda_ * tau),
            0.0,
        )

    def compute_weights(self, memory_length: int) -> np.ndarray:
        indices = np.arange(memory_length, dtype=np.float64)
        weights = np.exp(-self.lambda_ * indices)
        weights = weights / np.sum(weights)
        return weights

    def __repr__(self) -> str:
        return f"ExponentialKernel(lambda={self.lambda_:.3f})"


class UniformKernel(VolterraKernel):
    """Uniform kernel: K(t,s) = 1 for t > s."""

    def __init__(self):
        super().__init__("uniform")

    def evaluate(self, t: np.ndarray, s: np.ndarray) -> np.ndarray:
        t_mat = t[:, np.newaxis]
        s_mat = s[np.newaxis, :]
        tau = t_mat - s_mat
        return np.where(tau > 0, 1.0, 0.0)

    def compute_weights(self, memory_length: int) -> np.ndarray:
        return np.ones(memory_length, dtype=np.float64) / float(memory_length)

    def __repr__(self) -> str:
        return "UniformKernel()"


def fit_volterra_kernel(
    gradient_history: np.ndarray,
    kernel_type: str = "power_law",
    method: str = "L-BFGS-B",
    verbose: bool = False,
) -> Tuple[VolterraKernel, float, dict]:
    """
    Fit Volterra kernel parameters to observed gradient sequence.

    Args:
        gradient_history: Gradient sequence (T, D) where D is dimension
        kernel_type: Type of kernel ('power_law', 'exponential', 'uniform')
        method: Optimization method ('L-BFGS-B' or 'differential_evolution')
        verbose: Print optimization progress

    Returns:
        (fitted_kernel, rmse, info)
    """
    grad = np.asarray(gradient_history, dtype=np.float64)
    if grad.ndim == 1:
        grad = grad[:, np.newaxis]
    if grad.ndim != 2:
        raise ValueError("gradient_history must be a 2D array (T, D).")
    t_steps, _ = grad.shape
    if t_steps < 10:
        raise ValueError(f"Need at least 10 gradient samples, got {t_steps}")

    kernel_type = str(kernel_type).lower()
    if kernel_type in ("exp", "exponential"):
        kernel_type = "exponential"
    elif kernel_type in ("power", "powerlaw", "power_law", "power-law"):
        kernel_type = "power_law"
    elif kernel_type in ("uniform",):
        kernel_type = "uniform"
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    times = np.arange(t_steps, dtype=np.float64)

    def objective(params: np.ndarray) -> float:
        if kernel_type == "power_law":
            alpha, c = float(params[0]), float(params[1])
            kernel = PowerLawKernel(alpha=alpha, c=c)
        elif kernel_type == "exponential":
            lambda_ = float(params[0])
            kernel = ExponentialKernel(lambda_=lambda_)
        elif kernel_type == "uniform":
            kernel = UniformKernel()
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        k_mat = kernel.evaluate(times, times)
        g_recon = k_mat @ grad
        error = grad - g_recon
        return float(np.sqrt((error ** 2).mean()))

    if kernel_type == "power_law":
        bounds = [(0.1, 0.99), (0.1, 10.0)]
        x0 = np.array([0.5, 1.0], dtype=np.float64)
    elif kernel_type == "exponential":
        bounds = [(0.01, 10.0)]
        x0 = np.array([1.0], dtype=np.float64)
    else:
        kernel = UniformKernel()
        rmse = objective(np.array([], dtype=np.float64))
        return kernel, rmse, {"success": True, "nit": 0, "message": "uniform"}

    if method == "L-BFGS-B":
        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "disp": verbose},
        )
        optimal_params = result.x
        rmse = float(result.fun)
        info = {
            "success": bool(result.success),
            "nit": int(result.nit),
            "message": str(result.message),
        }
    elif method == "differential_evolution":
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            disp=verbose,
            seed=42,
        )
        optimal_params = result.x
        rmse = float(result.fun)
        info = {
            "success": bool(result.success),
            "nit": int(result.nit),
            "message": str(result.message),
        }
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    if kernel_type == "power_law":
        alpha_opt, c_opt = float(optimal_params[0]), float(optimal_params[1])
        fitted_kernel = PowerLawKernel(alpha=alpha_opt, c=c_opt)
        if verbose:
            print(f"[Volterra] Fitted alpha={alpha_opt:.4f}, c={c_opt:.4f}, RMSE={rmse:.6f}")
            if alpha_opt < 0.7:
                print("[Volterra] alpha < 0.7 -> RL-obfuscation suspected")
            else:
                print("[Volterra] alpha >= 0.7 -> Likely genuine gradients")
    elif kernel_type == "exponential":
        lambda_opt = float(optimal_params[0])
        fitted_kernel = ExponentialKernel(lambda_=lambda_opt)
        if verbose:
            print(f"[Volterra] Fitted lambda={lambda_opt:.4f}, RMSE={rmse:.6f}")
    else:
        fitted_kernel = UniformKernel()

    return fitted_kernel, rmse, info


def compute_volterra_correlation(
    gradient_history: np.ndarray,
    max_lag: int = 50,
) -> Tuple[np.ndarray, float]:
    """
    Compute temporal autocorrelation of gradient sequence.

    Returns:
        (autocorr, integral_timescale)
    """
    grad = np.asarray(gradient_history, dtype=np.float64)
    if grad.ndim == 1:
        grad = grad[:, np.newaxis]
    if grad.ndim != 2:
        raise ValueError("gradient_history must be a 2D array (T, D).")
    t_steps, dims = grad.shape
    if t_steps <= 1:
        return np.zeros((0,), dtype=np.float64), 0.0

    if max_lag >= t_steps:
        max_lag = t_steps - 1

    g_flat = grad - grad.mean(axis=0, keepdims=True)
    autocorr = np.zeros(max_lag, dtype=np.float64)

    for lag in range(max_lag):
        if lag == 0:
            autocorr[lag] = 1.0
            continue
        g_current = g_flat[lag:]
        g_lagged = g_flat[:-lag]
        if g_current.shape[0] < 2:
            autocorr[lag] = 0.0
            continue
        dims_used = int(min(dims, 100))
        corr_values = []
        for d in range(dims_used):
            a = g_current[:, d]
            b = g_lagged[:, d]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(a, b)[0, 1]
            if np.isfinite(corr):
                corr_values.append(float(corr))
        autocorr[lag] = float(np.mean(corr_values)) if corr_values else 0.0

    integral_timescale = float(np.trapz(autocorr, dx=1.0))
    return autocorr, integral_timescale


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


