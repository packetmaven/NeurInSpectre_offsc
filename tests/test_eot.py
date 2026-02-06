import torch
import torch.nn as nn
import torch.nn.functional as F

from neurinspectre.attacks.eot import EOT, AdaptiveEOT


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


def random_noise_transform(x, sigma=0.1):
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0, 1)


def test_eot_uniform():
    model = SimpleConv()

    def transform(x):
        return random_noise_transform(x, sigma=0.1)

    attack = EOT(model, transform, num_samples=10, importance_sampling=False, eps=0.3, steps=5, device="cpu")
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    x_adv = attack(x, y)
    assert (x_adv - x).abs().max() <= 0.3 + 1e-6
    assert not torch.allclose(x, x_adv)


def test_eot_importance_sampling():
    model = SimpleConv()

    def transform(x):
        return random_noise_transform(x, sigma=0.1)

    attack = EOT(
        model,
        transform,
        num_samples=10,
        importance_sampling=True,
        temperature=0.1,
        eps=0.3,
        steps=5,
        device="cpu",
    )
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    x_adv = attack(x, y)
    assert (x_adv - x).abs().max() <= 0.3 + 1e-6


def test_adaptive_eot():
    model = SimpleConv()

    def transform(x):
        return random_noise_transform(x, sigma=0.2)

    attack = AdaptiveEOT(
        model,
        transform,
        target_variance=0.01,
        min_samples=5,
        max_samples=50,
        eps=0.3,
        steps=3,
        device="cpu",
    )
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    x_adv = attack(x, y)
    assert (x_adv - x).abs().max() <= 0.3 + 1e-6


def test_eot_variance_reduction():
    model = SimpleConv()

    def transform(x):
        return random_noise_transform(x, sigma=0.15)

    x = torch.rand(5, 3, 32, 32, requires_grad=True)
    y = torch.randint(0, 10, (5,))
    delta = torch.zeros_like(x, requires_grad=True)

    attack_uniform = EOT(model, transform, num_samples=20, importance_sampling=False, device="cpu")
    grad_uniform, _ = attack_uniform._uniform_gradients(x, delta, y)

    attack_importance = EOT(model, transform, num_samples=20, importance_sampling=True, device="cpu")
    grad_importance, _ = attack_importance._importance_weighted_gradients(x, delta, y)

    assert grad_uniform.abs().sum() > 0
    assert grad_importance.abs().sum() > 0
