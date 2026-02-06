import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurinspectre.attacks.ma_pgd import MAPGD, MAPGDEnsemble
from neurinspectre.attacks.memory_gradient import memory_length_schedule
from neurinspectre.attacks.pgd import PGD


def _seed_all():
    torch.manual_seed(0)
    np.random.seed(0)


class RLObfuscationDefense(nn.Module):
    """Simulated RL-trained obfuscation defense for testing."""

    def __init__(self, model: nn.Module, alpha_correlation: float = 0.5):
        super().__init__()
        self.model = model
        self.alpha = alpha_correlation
        self.prev_grad_noise = None

    def forward(self, x):
        if self.training:
            x = x + self._generate_correlated_noise(x) * 0.01
        return self.model(x)

    def _generate_correlated_noise(self, x):
        noise = torch.randn_like(x)
        if self.prev_grad_noise is not None:
            noise = self.alpha * self.prev_grad_noise + (1 - self.alpha) * noise
        self.prev_grad_noise = noise.detach()
        return noise


class SimpleConvNet(nn.Module):
    """Simple CNN for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_mapgd_basic():
    _seed_all()
    model = SimpleConvNet()
    model.eval()

    attack = MAPGD(
        model,
        eps=8 / 255,
        steps=20,
        alpha_volterra=0.5,
        memory_length=10,
        auto_detect_alpha=False,
        device="cpu",
    )

    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    x_adv = attack(x, y)

    assert x_adv.shape == x.shape
    assert (x_adv - x).abs().max() <= 8 / 255 + 1e-6
    assert (x_adv >= 0).all() and (x_adv <= 1).all()


def test_mapgd_auto_detect_alpha():
    _seed_all()
    model = SimpleConvNet()
    model.eval()

    attack = MAPGD(
        model,
        eps=8 / 255,
        steps=50,
        auto_detect_alpha=True,
        n_detection_steps=20,
        device="cpu",
    )

    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0, 1, 2])

    x_adv, stats = attack(x, y, return_stats=True)

    assert stats["alpha_detected"] is not None
    assert 0.1 <= stats["alpha_detected"] <= 0.99
    assert stats["memory_length_used"] is not None
    assert 10 <= stats["memory_length_used"] <= 50

    assert x_adv.shape == x.shape


def test_mapgd_vs_pgd_improvement():
    _seed_all()
    base_model = SimpleConvNet()
    rl_defense = RLObfuscationDefense(base_model, alpha_correlation=0.6)
    rl_defense.eval()

    x = torch.rand(20, 3, 32, 32)
    y = torch.randint(0, 10, (20,))

    pgd = PGD(rl_defense, eps=8 / 255, steps=40, device="cpu")
    x_adv_pgd = pgd(x, y)

    with torch.no_grad():
        preds_pgd = rl_defense(x_adv_pgd).argmax(1)
        asr_pgd = (preds_pgd != y).float().mean()

    mapgd = MAPGD(
        rl_defense,
        eps=8 / 255,
        steps=40,
        alpha_volterra=0.4,
        memory_length=20,
        device="cpu",
    )
    x_adv_mapgd = mapgd(x, y)

    with torch.no_grad():
        preds_mapgd = rl_defense(x_adv_mapgd).argmax(1)
        asr_mapgd = (preds_mapgd != y).float().mean()

    improvement = asr_mapgd - asr_pgd
    assert asr_mapgd >= asr_pgd
    assert x_adv_mapgd.shape == x.shape
    assert x_adv_pgd.shape == x.shape

    _ = improvement


def test_mapgd_memory_weights():
    _seed_all()
    model = SimpleConvNet()
    attack = MAPGD(model, eps=8 / 255, alpha_volterra=0.5, memory_length=10, device="cpu")

    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    attack.forward(x, y)

    assert attack.memory_grad is not None
    weights = attack.memory_grad.weights
    assert torch.abs(weights.sum() - 1.0) < 1e-5
    assert weights[0] > weights[-1]


def test_mapgd_transformed_gradients():
    _seed_all()
    model = SimpleConvNet()

    attack_tg = MAPGD(model, eps=8 / 255, steps=20, use_tg=True, device="cpu")
    attack_no_tg = MAPGD(model, eps=8 / 255, steps=20, use_tg=False, device="cpu")

    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0, 1, 2])

    x_adv_tg, stats_tg = attack_tg(x, y, return_stats=True)
    x_adv_no_tg, stats_no_tg = attack_no_tg(x, y, return_stats=True)

    assert (x_adv_tg - x).abs().max() <= 8 / 255 + 1e-6
    assert (x_adv_no_tg - x).abs().max() <= 8 / 255 + 1e-6
    assert "final_asr" in stats_tg
    assert "final_asr" in stats_no_tg


def test_mapgd_ensemble():
    _seed_all()
    model = SimpleConvNet()
    attack = MAPGDEnsemble(
        model,
        eps=8 / 255,
        alphas=[0.3, 0.5, 0.7],
        kernel_types=["power_law"],
        device="cpu",
    )

    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    x_adv = attack(x, y)
    assert x_adv.shape == x.shape
    assert (x_adv - x).abs().max() <= 8 / 255 + 1e-6


def test_mapgd_kernel_types():
    _seed_all()
    model = SimpleConvNet()
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    for kernel_type in ["power_law", "exponential", "uniform"]:
        attack = MAPGD(
            model,
            eps=8 / 255,
            steps=20,
            kernel_type=kernel_type,
            memory_length=10,
            device="cpu",
        )
        x_adv = attack(x, y)
        assert x_adv.shape == x.shape


def test_mapgd_alpha_sensitivity():
    _seed_all()
    model = SimpleConvNet()
    x = torch.rand(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))

    alpha_values = [0.2, 0.4, 0.6, 0.8]
    asrs = []

    for alpha in alpha_values:
        attack = MAPGD(
            model,
            eps=8 / 255,
            steps=40,
            alpha_volterra=alpha,
            memory_length=memory_length_schedule(alpha),
            device="cpu",
        )
        _, stats = attack(x, y, return_stats=True)
        asrs.append(stats["final_asr"])

    assert len(asrs) == len(alpha_values)
    assert all(0.0 <= v <= 1.0 for v in asrs)


@pytest.mark.slow
def test_mapgd_correlation_with_alpha():
    pytest.skip("Requires full RL defense implementation (Week 3)")


def test_memory_length_schedule():
    _seed_all()
    k_small = memory_length_schedule(alpha=0.2, max_length=50)
    assert k_small > 30

    k_large = memory_length_schedule(alpha=0.9, max_length=50)
    assert k_large < 15

    k_medium = memory_length_schedule(alpha=0.5, max_length=50)
    assert 15 <= k_medium <= 30
