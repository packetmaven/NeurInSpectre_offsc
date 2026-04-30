"""
Statistical evasion utilities (draft-parity).

Implements a lightweight "iterative evasion loop" that modifies a current
distribution to evade a per-dimension drift detector while keeping the API
simple and deterministic for artifact evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .drift_detection_enhanced import PerDimKSDADCvMFisherBHDriftDetector


@dataclass(frozen=True)
class EvasionStep:
    iteration: int
    drift_detected: bool
    p_value: float
    adjusted_dims: List[int]
    top_features: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "drift_detected": bool(self.drift_detected),
            "p_value": float(self.p_value),
            "adjusted_dims": [int(x) for x in self.adjusted_dims],
            "top_features": list(self.top_features),
        }


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    return x.reshape(int(x.shape[0]), -1)


def iterative_evasion_loop(
    reference_data: np.ndarray,
    current_data: np.ndarray,
    *,
    confidence_level: float = 0.95,
    max_iters: int = 10,
    top_k_dims: int = 10,
    step_fraction: float = 1.0,
    seed: int = 0,
    detector: Optional[PerDimKSDADCvMFisherBHDriftDetector] = None,
) -> Tuple[np.ndarray, List[EvasionStep]]:
    """
    Iteratively modify `current_data` to evade per-dimension drift detection.

    This is *not* a substitute for an end-to-end attack that preserves task
    semantics; it is the statistical core needed to match the draft's described
    evasion loop.

    Returns:
        (evasive_current, history)
    """
    ref = _as_2d_float(reference_data)
    cur0 = _as_2d_float(current_data)
    cur = np.array(cur0, dtype=np.float64, copy=True)

    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)

    p = int(min(ref.shape[1], cur.shape[1]))
    ref = ref[:, :p]
    cur = cur[:, :p]

    rng = np.random.default_rng(int(seed))
    det = detector or PerDimKSDADCvMFisherBHDriftDetector(confidence_level=float(confidence_level), report_top_k=max(25, int(top_k_dims)))

    tf = float(step_fraction)
    tf = float(np.clip(tf, 0.0, 1.0))

    history: List[EvasionStep] = []

    for it in range(int(max(0, max_iters))):
        res = det.detect_drift(ref, cur)
        sig = res.statistical_significance or {}
        per_dim = sig.get("per_dimension") or {}
        top_features = list(per_dim.get("top_features") or [])

        dims: List[int] = []
        for f in top_features[: int(max(0, top_k_dims))]:
            try:
                dims.append(int(f.get("feature_index")))
            except Exception:
                continue

        history.append(
            EvasionStep(
                iteration=int(it),
                drift_detected=bool(res.drift_detected),
                p_value=float(res.p_value),
                adjusted_dims=list(dims),
                top_features=top_features[: int(max(0, top_k_dims))],
            )
        )

        if not bool(res.drift_detected):
            break
        if not dims or tf <= 0.0:
            break

        for j in dims:
            if j < 0 or j >= p:
                continue

            # Deterministic "strong" evasion when shapes match and tf==1:
            # make the marginal distribution identical by copying the reference column.
            if tf >= 1.0 and int(ref.shape[0]) == int(cur.shape[0]):
                cur[:, j] = ref[:, j]
                continue

            ref_col = ref[:, j]
            idx = rng.integers(0, int(ref_col.shape[0]), size=int(cur.shape[0]))
            target = ref_col[idx]
            cur[:, j] = (1.0 - tf) * cur[:, j] + tf * target

    return cur, history

