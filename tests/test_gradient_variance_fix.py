"""
Test that gradient variance is now computed correctly.
"""

import torch
import torch.nn as nn
import pytest

from neurinspectre.characterization.defense_analyzer import DefenseAnalyzer
from neurinspectre.defenses.non_differentiable import JPEGCompression


def test_variance_clean_vs_shattered():
    """
    Test that variance is DIFFERENT for clean vs shattered models.

    This was the BUG: both had zero variance.
    """
    # Clean model
    clean_model = create_test_model()

    # Shattered model (JPEG defense)
    shattered_model = nn.Sequential(
        JPEGCompression(quality=75),
        create_test_model(),
    )

    # Test data
    test_loader = create_test_loader(n_samples=50, batch_size=10)

    # Analyze clean
    analyzer_clean = DefenseAnalyzer(clean_model, n_samples=20, device="cpu")
    char_clean = analyzer_clean.characterize(test_loader)

    # Analyze shattered
    analyzer_shattered = DefenseAnalyzer(shattered_model, n_samples=20, device="cpu")
    char_shattered = analyzer_shattered.characterize(test_loader)

    print(f"\nClean variance: {char_clean.gradient_variance:.6f}")
    print(f"Shattered variance: {char_shattered.gradient_variance:.6f}")

    print(f"\nClean alpha: {char_clean.alpha_volterra:.3f}")
    print(f"Shattered alpha: {char_shattered.alpha_volterra:.3f}")

    # Variance should be DIFFERENT
    assert abs(char_clean.gradient_variance - char_shattered.gradient_variance) > 0.01, (
        "Variances should differ between clean and shattered!"
    )

    # Clean should have higher variance (smooth gradients)
    assert char_clean.gradient_variance > char_shattered.gradient_variance, (
        "Clean model should have HIGHER variance than shattered!"
    )

    # Alpha should be DIFFERENT
    assert abs(char_clean.alpha_volterra - char_shattered.alpha_volterra) > 0.1, (
        "Alpha values should differ significantly!"
    )

    # Shattered should be detected
    from neurinspectre.characterization.defense_analyzer import ObfuscationType

    assert ObfuscationType.SHATTERED in char_shattered.obfuscation_types, (
        "Shattered obfuscation should be detected!"
    )

    print("\n✅ All assertions passed! Variance bug is FIXED.")


def create_test_model():
    """Helper: simple CNN."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def create_test_loader(n_samples=100, batch_size=10):
    """Helper: test data."""
    dataset = torch.utils.data.TensorDataset(
        torch.rand(n_samples, 3, 32, 32),
        torch.randint(0, 10, (n_samples,)),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
