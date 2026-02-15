import torch
import torch.nn as nn
import torch.nn.functional as F

from neurinspectre.attacks.bpda import BPDA, LearnedBPDA
from neurinspectre.attacks.bpda_registry import BPDA_REGISTRY, bpda_thermometer


class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 10)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ToyDefense(nn.Module):
    def forward(self, x):
        x_quantized = torch.round(x * 255) / 255
        return x_quantized.detach()


def test_bpda_identity_approx():
    model = SimpleConv()
    defense = ToyDefense()
    attack = BPDA(model, defense, approx_name="identity", eps=0.3, steps=10, device="cpu")

    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    x_adv = attack(x, y)

    assert (x_adv - x).abs().max() <= 0.3 + 1e-6
    assert not torch.allclose(x, x_adv)


def test_bpda_registry():
    required_approxs = ["identity", "jpeg", "thermometer", "quantization"]
    for name in required_approxs:
        assert name in BPDA_REGISTRY, f"Missing approximation: {name}"


def test_bpda_thermometer_approx():
    x = torch.rand(5, 3, 32, 32).requires_grad_(True)
    x_approx = bpda_thermometer(x, levels=16)

    assert x_approx.shape == x.shape
    unique_vals = torch.unique(x_approx.detach())
    assert len(unique_vals) <= 17
    assert float(x_approx.min().item()) >= 0.0
    assert float(x_approx.max().item()) <= 1.0
    assert x_approx.requires_grad


def test_bpda_thermometer_bits_expands_channels():
    x = torch.rand(2, 3, 8, 8).requires_grad_(True)
    x_bits = BPDA_REGISTRY["thermometer_bits"](x, levels=10)
    assert x_bits.shape == (2, 3 * 10, 8, 8)
    assert x_bits.requires_grad is True


def test_learned_bpda_training():
    model = SimpleConv()
    defense = ToyDefense()

    approx_net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 3, 3, padding=1),
    )

    attack = LearnedBPDA(model, defense, approx_network=approx_net, train_steps=10, device="cpu")

    train_data = torch.utils.data.TensorDataset(
        torch.rand(20, 3, 32, 32),
        torch.zeros(20, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5)

    attack.train_approximation(train_loader)

    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    x_adv = attack(x, y)
    assert (x_adv - x).abs().max() <= attack.eps + 1e-6


def test_bpda_custom_approx():
    model = SimpleConv()
    defense = ToyDefense()

    def custom_approx(inp):
        return F.avg_pool2d(inp, 3, stride=1, padding=1)

    attack = BPDA(model, defense, approx_fn=custom_approx, eps=0.3, steps=10, device="cpu")

    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    x_adv = attack(x, y)
    assert not torch.allclose(x, x_adv)
