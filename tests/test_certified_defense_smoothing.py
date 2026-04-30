import torch
import torch.nn as nn

from neurinspectre.defenses.wrappers import CertifiedDefense


def test_certified_defense_averages_logits_when_not_eot():
    # With sigma=0, the stochastic transform is identity. The MC-averaged forward
    # should match the base model exactly (regression guard for vectorized averaging).
    base = nn.Sequential(nn.Flatten(), nn.Linear(16, 3))
    base.eval()

    defense = CertifiedDefense(base, sigma=0.0, n_samples=7, device="cpu")
    x = torch.rand(2, 1, 4, 4)
    with torch.no_grad():
        expected = base(x)
        got = defense(x)

    assert got.shape == expected.shape
    # Float ops (chunking/reshape/mean) can introduce tiny rounding diffs even when
    # sigma=0. Keep this strict but non-zero to avoid platform flakiness.
    assert torch.allclose(got, expected, atol=1e-7, rtol=0.0)

