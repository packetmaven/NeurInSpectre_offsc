from pathlib import Path

import torch

from neurinspectre.attacks.ednn_attack import EDNNAttack, EDNNConfig


def test_ednn_dim_selection_random_is_deterministic(tmp_path: Path) -> None:
    D = 64
    ref = torch.randn(32, D)
    cfg = EDNNConfig(device="cpu", output_dir=str(tmp_path))
    cfg.dim_selection = "random"
    cfg.dim_top_k = 10
    cfg.dim_selection_seed = 123
    ednn = EDNNAttack(reference_embeddings=ref, config=cfg)

    m1, info1 = ednn._get_dim_mask(D)
    m2, info2 = ednn._get_dim_mask(D)
    assert int(m1.sum().item()) == 10
    assert torch.allclose(m1, m2)
    assert info1["mode"] == "random"
    assert info2["mode"] == "random"


def test_ednn_dim_selection_spectral_selects_top_k(tmp_path: Path) -> None:
    D = 48
    # Reference embeddings with some structure.
    ref = torch.randn(64, D) * 0.1
    ref[:, :5] += torch.sin(torch.linspace(0, 20, 64)).unsqueeze(1)  # add oscillation to first dims

    cfg = EDNNConfig(device="cpu", output_dir=str(tmp_path))
    cfg.dim_selection = "spectral"
    cfg.dim_top_k = 12
    ednn = EDNNAttack(reference_embeddings=ref, config=cfg)

    mask, info = ednn._get_dim_mask(D)
    assert mask.shape == (D,)
    assert int(mask.sum().item()) == 12
    assert info["mode"] == "spectral"

