import numpy as np
import torch
import torch.nn as nn

from neurinspectre.attacks import BPDA, EOT, MemoryAugmentedPGD
from neurinspectre.utils import select_attack_suite


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_attack_selection_memory_augmented():
    model = ToyModel()
    features = {"volterra_alpha": 0.5, "eps": 0.1, "steps": 5}
    attacks = select_attack_suite(model, features, device="cpu")
    assert any(isinstance(a, MemoryAugmentedPGD) for a in attacks)


def test_attack_selection_bpda_and_eot():
    model = ToyModel()

    def defense_fn(x):
        return x

    def transform_fn(x):
        return x + torch.randn_like(x) * 0.01

    features = {
        "volterra_alpha": 0.9,
        "nondiff_defense": True,
        "defense_fn": defense_fn,
        "stochastic_defense": True,
        "transform_fn": transform_fn,
    }
    attacks = select_attack_suite(model, features, device="cpu")
    assert any(isinstance(a, BPDA) for a in attacks)
    assert any(isinstance(a, EOT) for a in attacks)
