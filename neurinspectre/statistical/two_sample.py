"""
Two-sample tests for distribution shift / evasion evaluation.

Tier 3 motivation: avoid the common reviewer objection "p > 0.05 implies
indistinguishable". A classifier two-sample test (C2ST) provides an intuitive
metric: if a lightweight classifier cannot discriminate X_ref vs X_cur,
AUC ~= 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class C2STResult:
    auc: float
    n_ref: int
    n_cur: int
    seed: int
    test_fraction: float
    permutation_p_value: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "auc": float(self.auc),
            "n_ref": int(self.n_ref),
            "n_cur": int(self.n_cur),
            "seed": int(self.seed),
            "test_fraction": float(self.test_fraction),
            "permutation_p_value": (None if self.permutation_p_value is None else float(self.permutation_p_value)),
        }


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={x.shape}")
    # Replace non-finite values deterministically (rare, but happens in feature extraction).
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def c2st_auc(
    X_ref: np.ndarray,
    X_cur: np.ndarray,
    *,
    seed: int = 0,
    test_fraction: float = 0.33,
    standardize: bool = True,
    n_permutations: int = 0,
) -> C2STResult:
    """
    Classifier Two-Sample Test (C2ST) using logistic regression + ROC AUC.

    Returns:
        C2STResult(auc=..., permutation_p_value=...optional...)
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - optional dependency in some builds
        raise ImportError("scikit-learn is required for C2ST") from exc

    Xr = _as_2d_float(X_ref)
    Xc = _as_2d_float(X_cur)
    n_ref = int(Xr.shape[0])
    n_cur = int(Xc.shape[0])
    if n_ref < 4 or n_cur < 4:
        raise ValueError("C2ST requires at least 4 samples per split.")

    X = np.concatenate([Xr, Xc], axis=0)
    y = np.concatenate([np.zeros((n_ref,), dtype=np.int64), np.ones((n_cur,), dtype=np.int64)], axis=0)

    tf = float(test_fraction)
    tf = float(np.clip(tf, 0.05, 0.95))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=tf,
        random_state=int(seed),
        shuffle=True,
        stratify=y,
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=int(seed),
    )
    if standardize:
        model = make_pipeline(StandardScaler(with_mean=True, with_std=True), clf)
    else:
        model = clf

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    p_value: Optional[float] = None
    if int(n_permutations) > 0:
        rng = np.random.default_rng(int(seed))
        null_aucs = []
        for _ in range(int(n_permutations)):
            y_perm = rng.permutation(y)
            X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
                X,
                y_perm,
                test_size=tf,
                random_state=int(rng.integers(0, 2**31 - 1)),
                shuffle=True,
                stratify=y_perm,
            )
            model.fit(X_train_p, y_train_p)
            proba_p = model.predict_proba(X_test_p)[:, 1]
            null_aucs.append(float(roc_auc_score(y_test_p, proba_p)))
        # One-sided: how often does a random labeling produce >= observed AUC?
        ge = sum(1 for a in null_aucs if a >= auc)
        p_value = float((ge + 1) / (len(null_aucs) + 1))

    return C2STResult(
        auc=float(auc),
        n_ref=n_ref,
        n_cur=n_cur,
        seed=int(seed),
        test_fraction=float(tf),
        permutation_p_value=p_value,
    )

