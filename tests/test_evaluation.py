import torch
import torch.nn as nn

from neurinspectre.attacks import PGD
from neurinspectre.utils import AttackEvaluator


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_attack_evaluator_single_batch():
    model = ToyModel()
    x = torch.rand(6, 1, 4, 4)
    y = torch.randint(0, 3, (6,))
    attack = PGD(model, eps=0.1, steps=10, device="cpu")
    evaluator = AttackEvaluator(model, device="cpu")
    res = evaluator.evaluate_single_batch(attack, x, y)
    assert "attack_success_rate" in res
    assert "grad_norm_mean" in res
