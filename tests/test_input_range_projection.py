"""
Input range invariance tests for attack projection.
"""

import torch

from neurinspectre.attacks import AttackConfig, PGDAttack


def _make_linear_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 10),
    )


def test_projection_normalized_cifar10():
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])

    x_raw = torch.rand(10, 3, 32, 32)
    x_normalized = (x_raw - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

    expected_min = ((0.0 - mean) / std).min().item()
    expected_max = ((1.0 - mean) / std).max().item()

    config = AttackConfig(
        epsilon=8 / 255,
        n_iterations=10,
        auto_detect_range=True,
    )
    attack = PGDAttack(config, device="cpu")

    model = _make_linear_model()
    y = torch.randint(0, 10, (10,))

    result = attack.run(model, x_normalized, y)

    assert float(result.x_adv.min().item()) >= expected_min - 0.1
    assert float(result.x_adv.max().item()) <= expected_max + 0.1
    assert float(result.x_adv.min().item()) < 0.0


def test_projection_standard_images():
    x = torch.rand(10, 3, 32, 32)

    config = AttackConfig(
        epsilon=8 / 255,
        n_iterations=10,
        auto_detect_range=True,
    )
    attack = PGDAttack(config, device="cpu")

    model = _make_linear_model()
    y = torch.randint(0, 10, (10,))

    result = attack.run(model, x, y)

    assert float(result.x_adv.min().item()) >= 0.0
    assert float(result.x_adv.max().item()) <= 1.0
