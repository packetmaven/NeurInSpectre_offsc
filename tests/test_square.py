import torch
import torch.nn as nn
import torch.nn.functional as F

from neurinspectre.attacks.square import SquareAttack, SquareAttackL2


class SimpleConv(nn.Module):
    """Simple CNN for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_square_attack_linf():
    model = SimpleConv()
    attack = SquareAttack(model, eps=8 / 255, n_queries=1000, device="cpu")
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    x_adv, stats = attack(x, y)
    delta = (x_adv - x).abs()
    assert (delta <= 8 / 255 + 1e-6).all(), f"Max delta: {delta.max()}"
    assert (x_adv >= 0).all() and (x_adv <= 1).all()
    assert "queries_used" in stats
    assert "success" in stats
    assert "asr" in stats
    assert stats["queries_used"].shape == (2,)


def test_square_attack_l2():
    model = SimpleConv()
    attack = SquareAttackL2(model, eps=0.5, n_queries=1000, device="cpu")
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    x_adv, _stats = attack(x, y)
    delta = x_adv - x
    norms = delta.view(2, -1).norm(p=2, dim=1)
    assert (norms <= 0.5 + 1e-4).all(), f"Max L2 norm: {norms.max()}"
    assert (x_adv >= 0).all() and (x_adv <= 1).all()


def test_square_attack_query_efficiency():
    model = SimpleConv()
    attack = SquareAttack(model, eps=8 / 255, n_queries=1000, device="cpu")
    x = torch.rand(3, 3, 32, 32)
    y = torch.randint(0, 10, (3,))

    _x_adv, stats = attack(x, y)
    if stats["asr"] > 0:
        avg_queries = stats["queries_used"][stats["success"]].mean()
        assert avg_queries <= attack.n_queries


def test_square_attack_targeted():
    model = SimpleConv()
    attack = SquareAttack(model, eps=8 / 255, n_queries=1000, device="cpu")
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([5, 7])

    x_adv, _stats = attack(x, y, targeted=True)
    assert (x_adv - x).abs().max() <= 8 / 255 + 1e-6


def test_square_size_schedule():
    model = SimpleConv()
    attack = SquareAttack(model, eps=8 / 255, n_queries=1000, p_init=0.8, device="cpu")

    p_0 = attack._square_size_schedule(0)
    p_mid = attack._square_size_schedule(500)
    p_end = attack._square_size_schedule(999)

    assert p_0 == 0.8
    assert p_mid < p_0
    assert p_end < p_mid
    assert p_end > 0
