import numpy as np

from neurinspectre.cli.attention_security_analysis import _attention_feature_matrix


def test_attention_feature_matrix_feature_sets_shapes():
    S = 12
    rng = np.random.default_rng(0)
    A = rng.random((S, S), dtype=np.float64)
    toks = [f"t{i}" for i in range(S)]

    X1, _per1, names1 = _attention_feature_matrix(A, toks, feature_set="entropy_only")
    assert X1.shape == (S, 1)
    assert names1 == ["row_entropy"]

    X2, _per2, names2 = _attention_feature_matrix(A, toks, feature_set="entropy_inj")
    assert X2.shape == (S, 10)  # 1 entropy + 9 token-text features
    assert "row_entropy" in names2
    assert any(n.startswith("txt_") for n in names2)

    X3, _per3, names3 = _attention_feature_matrix(A, toks, feature_set="spectral_only")
    assert X3.shape == (S, 5)
    assert set(names3) == {"row_entropy", "row_max", "diag", "col_sum", "col_max"}

    X4, _per4, names4 = _attention_feature_matrix(A, toks, feature_set="all")
    assert X4.shape == (S, 14)  # 5 attention features + 9 token-text features
    assert "row_entropy" in names4
    assert any(n.startswith("txt_") for n in names4)

