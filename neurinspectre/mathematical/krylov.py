"""
Krylov subspace utilities (Layer 3).

This module implements a lightweight Arnoldi/Krylov projection for matrix-exponential
actions e^{dt L} v without materializing the (N×N) operator L.

Intended CLI usage: `neurinspectre math krylov` and the alias `neurinspectre dna_krylov_projection`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


MatVec = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class KrylovStepResult:
    """Per-step Krylov projection diagnostics."""

    m_eff: int
    rel_reconstruction_error: float
    norm_ratio: float


def laplacian_1d_matvec(x: np.ndarray, *, damping: float = 0.1) -> np.ndarray:
    """
    Apply a 1D Laplacian-like operator with optional damping:

        (Lx)_i = x_{i-1} - 2 x_i + x_{i+1}  - damping * x_i

    Boundary: Neumann-like (zero-derivative) as described in the blog:
      first row [-1, 1], last row [1, -1] (before damping).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        return x.copy()
    if n == 1:
        return np.array([(-2.0 - float(damping)) * float(x[0])], dtype=np.float64)

    y = np.empty_like(x)
    # interior
    y[1:-1] = x[:-2] - 2.0 * x[1:-1] + x[2:]
    # Neumann boundaries
    y[0] = -1.0 * x[0] + x[1]
    y[-1] = x[-2] - 1.0 * x[-1]

    d = float(damping)
    if d != 0.0:
        y = y - d * x
    return y


def arnoldi_iteration(
    matvec: MatVec,
    v: np.ndarray,
    *,
    m: int,
    atol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Arnoldi iteration (modified Gram-Schmidt).

    Returns:
        V: (n, m_eff) orthonormal basis
        H: (m_eff, m_eff) Hessenberg projection of A onto Krylov subspace
        beta: ||v||
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = int(v.size)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64), 0.0

    beta = float(np.linalg.norm(v))
    if not np.isfinite(beta) or beta < 1e-14:
        return np.zeros((n, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64), float(beta)

    m = int(max(1, min(int(m), n)))
    V = np.zeros((n, m), dtype=np.float64)
    H_full = np.zeros((m + 1, m), dtype=np.float64)

    V[:, 0] = v / beta

    m_eff = m
    for j in range(m):
        w = np.asarray(matvec(V[:, j]), dtype=np.float64).reshape(-1)
        if w.size != n:
            raise ValueError("matvec returned wrong shape")

        for i in range(j + 1):
            hij = float(np.dot(V[:, i], w))
            H_full[i, j] = hij
            w = w - hij * V[:, i]

        if j < m - 1:
            hj1 = float(np.linalg.norm(w))
            H_full[j + 1, j] = hj1
            if not np.isfinite(hj1) or hj1 < float(atol):
                m_eff = j + 1
                break
            V[:, j + 1] = w / hj1

    V = V[:, :m_eff]
    H = H_full[:m_eff, :m_eff]
    return V, H, beta


def krylov_expm_action(
    matvec_L: MatVec,
    v: np.ndarray,
    *,
    dt: float,
    m: int,
    atol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Approximate exp(dt * L) @ v using a Krylov subspace projection.

    Returns:
        expLv: approximated action (n,)
        H: (m_eff, m_eff) projection of (dt*L)
        V: (n, m_eff) basis
        beta: ||v||
    """
    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be > 0")

    def matvec_A(x: np.ndarray) -> np.ndarray:
        return dt * np.asarray(matvec_L(x), dtype=np.float64)

    V, H_L, beta = arnoldi_iteration(matvec_A, v, m=m, atol=atol)
    if V.shape[1] == 0:
        return np.zeros_like(np.asarray(v, dtype=np.float64).reshape(-1)), H_L, V, beta

    # exp(H) is small (m_eff×m_eff)
    from scipy.linalg import expm

    expH = expm(H_L)
    e1 = np.zeros((expH.shape[0],), dtype=np.float64)
    e1[0] = 1.0
    y = expH @ e1
    expLv = (beta * (V @ y)).reshape(-1)
    return expLv, H_L, V, beta


def analyze_krylov_projection(
    seq: np.ndarray,
    *,
    krylov_dim: int = 30,
    dt: float = 1.0,
    damping: float = 0.1,
    steps: int = 25,
    stride: int = 1,
    atol: float = 1e-12,
) -> Tuple[dict, list[KrylovStepResult], np.ndarray]:
    """
    Analyze a gradient sequence u[t] using a dissipative Laplacian operator + Krylov expm projection.

    Returns:
        summary: JSON-serializable dict with overall metrics
        step_results: list of per-step results
        eigvals0: eigenvalues of the first-step Hessenberg matrix (complex64)
    """
    arr = np.asarray(seq)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)

    T, D = int(arr.shape[0]), int(arr.shape[1])
    if T < 2:
        raise ValueError("Need a sequence with at least 2 time steps for Krylov projection.")

    steps = int(max(1, steps))
    stride = int(max(1, stride))
    max_transitions = min(T - 1, steps * stride)

    def matvec_L(x: np.ndarray) -> np.ndarray:
        return laplacian_1d_matvec(x, damping=float(damping))

    eps = 1e-12
    step_out: list[KrylovStepResult] = []
    eigvals0: Optional[np.ndarray] = None
    m_eff_vals: list[int] = []
    rel_errs: list[float] = []
    norm_ratios: list[float] = []

    transitions = 0
    for t0 in range(0, max_transitions, stride):
        if t0 >= T - 1:
            break
        u0 = np.asarray(arr[t0], dtype=np.float64).reshape(-1)
        u1 = np.asarray(arr[t0 + 1], dtype=np.float64).reshape(-1)

        exp_u0, H, V, _beta = krylov_expm_action(matvec_L, u0, dt=float(dt), m=int(krylov_dim), atol=float(atol))
        if eigvals0 is None and H.size:
            eigvals0 = np.linalg.eigvals(H).astype(np.complex64)

        denom = float(np.linalg.norm(u1) + eps)
        rel_err = float(np.linalg.norm(u1 - exp_u0) / denom)
        n0 = float(np.linalg.norm(u0) + eps)
        n1 = float(np.linalg.norm(u1) + eps)
        ratio = float(n1 / n0)

        m_eff = int(V.shape[1])
        step_out.append(KrylovStepResult(m_eff=m_eff, rel_reconstruction_error=rel_err, norm_ratio=ratio))
        m_eff_vals.append(m_eff)
        rel_errs.append(rel_err)
        norm_ratios.append(ratio)
        transitions += 1

        if transitions >= steps:
            break

    if eigvals0 is None:
        eigvals0 = np.zeros((0,), dtype=np.complex64)

    rel_errs_arr = np.asarray(rel_errs, dtype=np.float64) if rel_errs else np.zeros((0,), dtype=np.float64)
    norm_ratios_arr = np.asarray(norm_ratios, dtype=np.float64) if norm_ratios else np.zeros((0,), dtype=np.float64)

    # Dissipation anomaly: fraction of steps where the norm grows (>1.0).
    growth = (norm_ratios_arr > 1.0).astype(np.float64) if norm_ratios_arr.size else np.zeros((0,), dtype=np.float64)
    dissipation_anomaly_score = float(np.mean(growth)) if growth.size else 0.0

    summary = {
        "krylov_basis_dimension": int(krylov_dim),
        "krylov_basis_dimension_eff_mean": float(np.mean(m_eff_vals)) if m_eff_vals else 0.0,
        "dt": float(dt),
        "damping": float(damping),
        "sequence_shape": [int(T), int(D)],
        "steps_analyzed": int(transitions),
        "reconstruction_error": {
            "mean": float(np.mean(rel_errs_arr)) if rel_errs_arr.size else 0.0,
            "max": float(np.max(rel_errs_arr)) if rel_errs_arr.size else 0.0,
        },
        "dissipation": {
            "norm_ratio_mean": float(np.mean(norm_ratios_arr)) if norm_ratios_arr.size else 0.0,
            "norm_ratio_max": float(np.max(norm_ratios_arr)) if norm_ratios_arr.size else 0.0,
            "norm_growth_fraction": float(np.mean(growth)) if growth.size else 0.0,
            "dissipation_anomaly_score": float(dissipation_anomaly_score),
        },
    }

    return summary, step_out, eigvals0


