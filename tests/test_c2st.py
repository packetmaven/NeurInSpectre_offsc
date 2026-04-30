import numpy as np

from neurinspectre.statistical.two_sample import c2st_auc


def test_c2st_auc_near_chance_on_same_distribution() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    res = c2st_auc(X, X.copy(), seed=0, test_fraction=0.33)
    # Finite-sample variance can push AUC away from 0.5; keep the check loose
    # to avoid flaky tests across sklearn/BLAS versions.
    assert 0.30 <= res.auc <= 0.70


def test_c2st_auc_detects_shift() -> None:
    rng = np.random.default_rng(0)
    X_ref = rng.normal(size=(200, 8))
    X_cur = rng.normal(loc=0.75, size=(200, 8))
    res = c2st_auc(X_ref, X_cur, seed=0, test_fraction=0.33)
    assert res.auc >= 0.70

