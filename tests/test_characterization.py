import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurinspectre.attacks.attack_orchestrator import AttackOrchestrator, attack_with_characterization
from neurinspectre.characterization import (
    compute_etd_features,
    compute_spectral_features,
    fit_volterra_features,
)
from neurinspectre.characterization.defense_analyzer import (
    DefenseAnalyzer,
    DefenseCharacterization,
    ObfuscationType,
    quick_characterize,
)


def _seed_all():
    torch.manual_seed(0)
    np.random.seed(0)


class JPEGDefense(nn.Module):
    """Simulated JPEG compression defense (shattered gradients)."""

    def __init__(self, model, quality=75):
        super().__init__()
        self.model = model
        self.quality = quality

    def forward(self, x):
        if self.training:
            x_quantized = torch.round(x * 255) / 255
            return self.model(x_quantized)
        return self.model(x)


class RandomNoiseDefense(nn.Module):
    """Simulated random noise defense (stochastic gradients)."""

    def __init__(self, model, noise_std=0.1):
        super().__init__()
        self.model = model
        self.noise_std = noise_std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return self.model(x + noise)
        return self.model(x)


class RLObfuscationDefense(nn.Module):
    """Simulated RL-trained obfuscation."""

    def __init__(self, model, alpha=0.5):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.prev_noise = None

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * 0.01
            if self.prev_noise is not None and self.prev_noise.shape == noise.shape:
                noise = self.alpha * self.prev_noise + (1 - self.alpha) * noise
            self.prev_noise = noise.detach()
            return self.model(x + noise)
        return self.model(x)


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


def create_dummy_loader(n_batches=3, batch_size=20):
    dataset = torch.utils.data.TensorDataset(
        torch.rand(n_batches * batch_size, 3, 32, 32),
        torch.randint(0, 10, (n_batches * batch_size,)),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def test_characterize_jpeg_defense():
    _seed_all()
    base_model = SimpleConvNet()
    jpeg_defense = JPEGDefense(base_model, quality=75)
    jpeg_defense.train()

    test_loader = create_dummy_loader(n_batches=5, batch_size=20)
    analyzer = DefenseAnalyzer(jpeg_defense, n_samples=30, n_probe_images=50, device="cpu")
    char = analyzer.characterize(test_loader)

    assert ObfuscationType.SHATTERED in char.obfuscation_types
    assert char.alpha_volterra < 0.5
    assert char.requires_bpda


def test_characterize_random_noise_defense():
    _seed_all()
    base_model = SimpleConvNet()
    noise_defense = RandomNoiseDefense(base_model, noise_std=0.1)
    noise_defense.train()

    test_loader = create_dummy_loader(n_batches=5, batch_size=20)
    analyzer = DefenseAnalyzer(noise_defense, n_samples=30, device="cpu")
    char = analyzer.characterize(test_loader)

    assert ObfuscationType.STOCHASTIC in char.obfuscation_types
    assert char.gradient_variance > 1e-5
    assert char.requires_eot
    assert char.recommended_eot_samples >= 10


def test_characterize_rl_defense():
    _seed_all()
    base_model = SimpleConvNet()
    rl_defense = RLObfuscationDefense(base_model, alpha=0.6)
    rl_defense.train()

    test_loader = create_dummy_loader(n_batches=5, batch_size=20)
    analyzer = DefenseAnalyzer(rl_defense, n_samples=50, device="cpu")
    char = analyzer.characterize(test_loader)

    assert ObfuscationType.RL_TRAINED in char.obfuscation_types
    assert 0.2 < char.alpha_volterra < 0.8
    assert char.requires_mapgd
    assert 10 <= char.recommended_memory_length <= 50


def test_characterize_clean_model():
    _seed_all()
    model = SimpleConvNet()
    model.eval()

    test_loader = create_dummy_loader(n_batches=3, batch_size=20)
    analyzer = DefenseAnalyzer(model, n_samples=30, device="cpu")
    char = analyzer.characterize(test_loader)

    assert ObfuscationType.NONE in char.obfuscation_types or len(char.obfuscation_types) == 0
    assert not (char.requires_bpda and char.requires_eot and char.requires_mapgd)


def test_orchestrator_with_characterization():
    _seed_all()
    model = SimpleConvNet()
    model.eval()

    test_loader = create_dummy_loader(n_batches=2, batch_size=10)
    analyzer = DefenseAnalyzer(model, device="cpu")
    char = analyzer.characterize(test_loader)

    orchestrator = AttackOrchestrator(model, characterization=char, device="cpu")

    x = torch.rand(5, 3, 32, 32)
    y = torch.randint(0, 10, (5,))

    x_adv = orchestrator(x, y)
    assert x_adv.shape == x.shape
    assert (x_adv - x).abs().max() <= 8 / 255 + 1e-6


def test_orchestrator_auto_characterize():
    _seed_all()
    model = SimpleConvNet()
    model.eval()

    test_loader = create_dummy_loader(n_batches=3, batch_size=20)
    orchestrator = AttackOrchestrator(
        model,
        auto_characterize_data=test_loader,
        device="cpu",
        verbose=True,
    )

    x = torch.rand(5, 3, 32, 32)
    y = torch.randint(0, 10, (5,))

    x_adv = orchestrator(x, y)
    assert x_adv.shape == x.shape
    assert orchestrator.characterization is not None


def test_quick_characterize():
    _seed_all()
    model = SimpleConvNet()
    test_loader = create_dummy_loader(n_batches=2, batch_size=20)
    char = quick_characterize(model, test_loader, device="cpu")

    assert isinstance(char, DefenseCharacterization)
    assert char.obfuscation_types is not None
    assert 0 <= char.etd_score <= 2.0
    assert 0.1 <= char.alpha_volterra <= 0.99


def test_attack_with_characterization_convenience():
    _seed_all()
    model = SimpleConvNet()
    model.eval()

    char_data = create_dummy_loader(n_batches=2, batch_size=20)
    x = torch.rand(3, 3, 32, 32)
    y = torch.randint(0, 10, (3,))

    x_adv = attack_with_characterization(
        model,
        x,
        y,
        characterization_data=char_data,
        eps=8 / 255,
        device="cpu",
    )
    assert x_adv.shape == x.shape


def test_characterization_accuracy():
    _seed_all()
    defenses = {
        "jpeg": JPEGDefense(SimpleConvNet(), quality=75),
        "noise": RandomNoiseDefense(SimpleConvNet(), noise_std=0.1),
        "rl": RLObfuscationDefense(SimpleConvNet(), alpha=0.5),
        "clean": SimpleConvNet(),
    }

    expected_types = {
        "jpeg": ObfuscationType.SHATTERED,
        "noise": ObfuscationType.STOCHASTIC,
        "rl": ObfuscationType.RL_TRAINED,
        "clean": ObfuscationType.NONE,
    }

    test_loader = create_dummy_loader(n_batches=3, batch_size=20)
    correct = 0
    total = len(defenses)

    for name, defense_model in defenses.items():
        if name != "clean":
            defense_model.train()
        analyzer = DefenseAnalyzer(defense_model, n_samples=30, device="cpu", verbose=False)
        char = analyzer.characterize(test_loader)

        expected = expected_types[name]
        detected = expected in char.obfuscation_types or (
            expected == ObfuscationType.NONE and len(char.obfuscation_types) == 0
        )
        if detected:
            correct += 1

    accuracy = correct / total
    assert accuracy >= 0.75


def test_etd_score_computation():
    _seed_all()
    model = SimpleConvNet()
    model.eval()

    x = torch.rand(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))

    analyzer = DefenseAnalyzer(model, device="cpu")
    etd = analyzer._compute_etd_score(x, y, eps=8 / 255)
    assert 0 <= etd <= 2.0


def test_volterra_fitting():
    _seed_all()
    model = SimpleConvNet()
    x = torch.rand(5, 3, 32, 32).requires_grad_(True)
    y = torch.randint(0, 10, (5,))

    gradients = []
    for _ in range(30):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grad = x.grad.detach().clone()
        grad_flat = grad.view(grad.size(0), -1).cpu().numpy().mean(axis=0)
        gradients.append(grad_flat)
        x.grad.zero_()

    analyzer = DefenseAnalyzer(model, device="cpu")
    alpha, rmse, rmse_scaled, info = analyzer._fit_volterra_kernel(gradients)
    assert isinstance(info, dict)
    assert 0.1 <= alpha <= 0.99
    assert not np.isnan(rmse)
    assert np.isnan(rmse_scaled) or rmse_scaled >= 0.0


def test_spectral_features_keys():
    seq = np.sin(np.linspace(0, 2 * np.pi, 64))
    feats = compute_spectral_features(seq)
    assert "spectral_entropy" in feats
    assert "spectral_entropy_norm" in feats
    assert "high_freq_ratio" in feats


def test_volterra_features_keys():
    seq = np.sin(np.linspace(0, 2 * np.pi, 64))
    feats = fit_volterra_features(seq)
    assert "volterra_alpha" in feats
    assert "volterra_c" in feats
    assert "volterra_rmse" in feats


def test_etd_features_keys():
    seq = np.random.randn(32, 8)
    feats = compute_etd_features(seq, steps=5)
    assert "krylov_rel_error_mean" in feats
    assert "krylov_rel_error_max" in feats
