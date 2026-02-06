import torch
import torch.nn as nn

from neurinspectre.attacks import PGD
from neurinspectre.attacks.autotune import tune_attack_params


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_autotune_returns_params():
    torch.manual_seed(0)
    model = ToyModel()
    x = torch.rand(8, 1, 4, 4)
    y = torch.randint(0, 3, (8,))

    grid = {"eps": [0.05, 0.1], "steps": [10], "alpha": [0.01]}
    best_kwargs, best_asr = tune_attack_params(PGD, model, x, y, grid, max_trials=2)
    assert best_kwargs is not None
    assert isinstance(best_asr, float)
