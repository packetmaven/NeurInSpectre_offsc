import numpy as np

from neurinspectre.cli.attention_security_analysis import (
    _attention_head_feature_matrix,
    _isolation_forest_scores,
)


def test_attention_head_feature_matrix_shapes():
    H = 4
    S = 10
    rng = np.random.default_rng(0)
    A = rng.random((H, S, S), dtype=np.float64)

    X, per, names = _attention_head_feature_matrix(A)
    assert X.shape == (H, len(names))
    assert X.shape[1] == 8
    assert names[0] == "entropy_mean"
    assert "col_sum_max" in per
    assert per["col_sum_max"].shape == (H,)


def test_isolation_forest_scores_small_n_single_column():
    # Regression test: small-n fallback used to assume X has >=2 columns.
    X = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)
    scores = _isolation_forest_scores(X, contamination="auto", n_estimators=32, seed=0)
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0 + 1e-9

