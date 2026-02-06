import torch
import torch.nn as nn
import pytest

from neurinspectre.defenses.non_differentiable import (
    JPEGCompression,
    BitDepthReduction,
    ThermometerEncoding,
    MedianFilter,
)
from neurinspectre.defenses.stochastic import (
    RandomResizing,
    RandomPadding,
    RandomNoise,
    RandomSmoothing,
)
from neurinspectre.defenses.adversarial_training import AdversarialTraining, TRADES, MART


def test_jpeg_compression():
    """Test JPEG defense."""
    defense = JPEGCompression(quality=75, differentiable=False)
    x = torch.rand(2, 3, 32, 32)
    x_defended = defense(x)

    assert x_defended.shape == x.shape
    assert (x_defended >= 0).all() and (x_defended <= 1).all()

    approx = defense.get_bpda_approximation()
    x_approx = approx(x)
    assert x_approx.requires_grad == x.requires_grad


def test_bit_depth_reduction():
    """Test bit-depth reduction."""
    defense = BitDepthReduction(bits=4)
    x = torch.rand(2, 3, 32, 32)
    x_defended = defense(x)

    unique_values = torch.unique(x_defended)
    assert len(unique_values) <= 2 ** 4


def test_thermometer_encoding():
    defense = ThermometerEncoding(levels=16)
    x = torch.rand(2, 3, 32, 32)
    x_defended = defense(x)
    assert x_defended.shape == x.shape


def test_median_filter():
    defense = MedianFilter(kernel_size=3)
    x = torch.rand(2, 3, 32, 32)
    x_defended = defense(x)
    assert x_defended.shape == x.shape


def test_random_resizing():
    """Test random resizing (stochastic)."""
    defense = RandomResizing(scale_range=(0.8, 1.2))
    x = torch.rand(2, 3, 32, 32)

    x1 = defense(x)
    x2 = defense(x)
    assert not torch.allclose(x1, x2), "Should be stochastic"


def test_random_padding():
    defense = RandomPadding(max_pad=4)
    x = torch.rand(2, 3, 32, 32)
    x1 = defense(x)
    x2 = defense(x)
    assert x1.shape == x.shape
    assert not torch.allclose(x1, x2), "Should be stochastic"


def test_random_noise():
    """Test random noise defense."""
    defense = RandomNoise(std=0.05)
    x = torch.rand(2, 3, 32, 32)
    x1 = defense(x)
    x2 = defense(x)
    assert not torch.allclose(x1, x2), "Should add random noise"


def test_random_smoothing():
    defense = RandomSmoothing(sigma=0.25, n_samples=5)
    x = torch.rand(2, 3, 32, 32)
    x_defended = defense(x)
    assert x_defended.shape == x.shape


@pytest.mark.slow
def test_adversarial_training():
    """Test adversarial training trainer."""
    from neurinspectre.attacks import PGD

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )

    attack = PGD(model, eps=8 / 255, steps=10, device="cpu")
    trainer = AdversarialTraining(model, attack, device="cpu")

    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss = trainer.train_step(x, y, optimizer)
    assert loss > 0
