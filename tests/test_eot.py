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
    grad_uniform, n_uniform = attack_uniform._uniform_gradients(x, delta, y)

    attack_importance = EOT(model, transform, num_samples=20, importance_sampling=True, device="cpu")
    grad_importance, divisor = attack_importance._importance_weighted_gradients(x, delta, y)

    assert int(n_uniform) == 20
    # Importance-weighted gradients are already a mean (weights sum to 1),
    # so the forward path should not divide by an "effective N".
    assert int(divisor) == 1
    assert hasattr(attack_importance, "_last_effective_n")
    assert 1.0 <= float(getattr(attack_importance, "_last_effective_n")) <= 20.0

    assert grad_uniform.abs().sum() > 0
    assert grad_importance.abs().sum() > 0


def test_eot_importance_weights_favor_higher_loss():
    weights = EOT._compute_importance_weights(
        [0.1, 0.5, 2.0, 1.0],
        temperature=0.2,
        device="cpu",
    )
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    # Highest loss (2.0) should get highest weight.
    assert int(weights.argmax().item()) == 2


def test_eot_importance_effective_n_equals_num_samples_for_deterministic_transform():
    model = SimpleConv()

    def transform(x):
        return x

    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    delta = torch.zeros_like(x, requires_grad=True)

    attack = EOT(model, transform, num_samples=10, importance_sampling=True, device="cpu")
    _grad, divisor = attack._importance_weighted_gradients(x, delta, y)

    assert int(divisor) == 1
    assert hasattr(attack, "_last_effective_n")
    assert abs(float(getattr(attack, "_last_effective_n")) - 10.0) < 1e-6
