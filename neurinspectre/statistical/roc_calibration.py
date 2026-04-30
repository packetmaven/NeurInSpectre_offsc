"""
ROC/AUC calibration utilities (Tier 2).

Goal: replace hand-tuned thresholds with calibrated operating points (e.g., 5% FPR).
This module is intentionally generic and does not ship any paper baseline numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class RocCalibration:
    metric: str
    auc: float
    threshold: float
    rule: str  # "score>=threshold" or "score<=threshold"
    target_fpr: Optional[float]
    target_tpr: Optional[float]
    achieved_fpr: float
    achieved_tpr: float
    n_pos: int
    n_neg: int
    curve: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "metric": str(self.metric),
            "auc": float(self.auc),
            "threshold": float(self.threshold),
            "rule": str(self.rule),
            "target_fpr": None if self.target_fpr is None else float(self.target_fpr),
            "target_tpr": None if self.target_tpr is None else float(self.target_tpr),
            "achieved_fpr": float(self.achieved_fpr),
            "achieved_tpr": float(self.achieved_tpr),
            "n_pos": int(self.n_pos),
            "n_neg": int(self.n_neg),
        }
        if self.curve is not None:
            out["curve"] = dict(self.curve)
        return out


def _pick_index_by_target_fpr(fpr: np.ndarray, tpr: np.ndarray, *, target_fpr: float) -> int:
    target_fpr = float(np.clip(float(target_fpr), 0.0, 1.0))
    feasible = np.nonzero(fpr <= target_fpr + 1e-12)[0]
    if feasible.size == 0:
        # No operating point achieves the requested FPR. Fall back to minimal FPR.
        return int(np.argmin(fpr))
    # Among feasible points, pick the highest TPR.
    idx_local = int(np.argmax(tpr[feasible]))
    return int(feasible[idx_local])


def _pick_index_by_target_tpr(fpr: np.ndarray, tpr: np.ndarray, *, target_tpr: float) -> int:
    target_tpr = float(np.clip(float(target_tpr), 0.0, 1.0))
    feasible = np.nonzero(tpr >= target_tpr - 1e-12)[0]
    if feasible.size == 0:
        # No operating point achieves the requested TPR. Fall back to maximal TPR.
        return int(np.argmax(tpr))
    # Among feasible points, pick the lowest FPR.
    idx_local = int(np.argmin(fpr[feasible]))
    return int(feasible[idx_local])


def calibrate_threshold(
    *,
    metric: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    greater_is_positive: bool = True,
    target_fpr: Optional[float] = 0.05,
    target_tpr: Optional[float] = None,
    include_curve: bool = False,
) -> RocCalibration:
    """
    Calibrate a scalar threshold for a detection score using ROC analysis.

    Args:
        metric: Display name for the score.
        y_true: Binary labels (0/1) where 1 is the positive class.
        scores: Real-valued scores aligned with y_true.
        greater_is_positive: If False, smaller scores indicate the positive class.
        target_fpr: Select threshold to satisfy FPR<=target_fpr with maximal TPR.
        target_tpr: Alternatively, satisfy TPR>=target_tpr with minimal FPR.
        include_curve: Include full ROC arrays in the output dict.
    """
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have the same length.")

    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least 1 positive and 1 negative sample for ROC calibration.")

    # Unify the ROC direction so "higher score => more positive".
    s_eff = s if greater_is_positive else -s

    try:
        from sklearn.metrics import auc as _auc
        from sklearn.metrics import roc_curve
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-learn is required for ROC calibration") from exc

    fpr, tpr, thresholds_eff = roc_curve(y, s_eff)
    auc_val = float(_auc(fpr, tpr))

    if target_tpr is not None and target_fpr is not None:
        raise ValueError("Provide only one of target_fpr or target_tpr.")

    if target_tpr is not None:
        idx = _pick_index_by_target_tpr(fpr, tpr, target_tpr=float(target_tpr))
    else:
        idx = _pick_index_by_target_fpr(fpr, tpr, target_fpr=float(target_fpr) if target_fpr is not None else 0.05)

    th_eff = float(thresholds_eff[idx])
    th = float(th_eff if greater_is_positive else -th_eff)
    rule = "score>=threshold" if greater_is_positive else "score<=threshold"

    curve_out = None
    if include_curve:
        curve_out = {
            "fpr": [float(v) for v in fpr.tolist()],
            "tpr": [float(v) for v in tpr.tolist()],
            # Keep thresholds in the *original* score space.
            "thresholds": [float(v if greater_is_positive else -v) for v in thresholds_eff.tolist()],
        }

    return RocCalibration(
        metric=str(metric),
        auc=float(auc_val),
        threshold=float(th),
        rule=str(rule),
        target_fpr=None if target_fpr is None else float(target_fpr),
        target_tpr=None if target_tpr is None else float(target_tpr),
        achieved_fpr=float(fpr[idx]),
        achieved_tpr=float(tpr[idx]),
        n_pos=int(n_pos),
        n_neg=int(n_neg),
        curve=curve_out,
    )

