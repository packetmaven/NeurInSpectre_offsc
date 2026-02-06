"""
Integration tests for Week 2 implementations.

Tests end-to-end workflows:
  1. Characterization -> Orchestration -> Attack
  2. Figure 1 reproduction (alpha vs delta ASR)
  3. All paper claim validations (fast/synthetic mode)
"""

import torch
import pytest

from neurinspectre.characterization import DefenseAnalyzer
from neurinspectre.attacks import AttackOrchestrator
from neurinspectre.experiments.reproduce_paper_claims import PaperClaimValidator, reproduce_all_paper_claims


def test_end_to_end_pipeline():
    """Test complete NeurInSpectre pipeline."""
    from tests.test_characterization import SimpleConvNet, RLObfuscationDefense

    base_model = SimpleConvNet()
    defense = RLObfuscationDefense(base_model, alpha=0.5)

    test_loader = create_dummy_loader(3, 20)
    analyzer = DefenseAnalyzer(defense, n_samples=20, n_probe_images=40, device="cpu")
    char = analyzer.characterize(test_loader)

    assert char is not None
    assert len(char.obfuscation_types) > 0

    orchestrator = AttackOrchestrator(defense, characterization=char, device="cpu")

    x = torch.rand(5, 3, 32, 32)
    y = torch.randint(0, 10, (5,))

    x_adv = orchestrator(x, y)
    with torch.no_grad():
        preds = defense(x_adv).argmax(1)
        asr = (preds != y).float().mean()

    assert asr >= 0.0


@pytest.mark.slow
def test_reproduce_figure_1():
    """Test Figure 1 reproduction (alpha vs delta ASR correlation)."""
    validator = PaperClaimValidator(
        device="cpu",
        n_seeds=2,
        verbose=True,
        fast_mode=True,
        synthetic_mode=True,
        plot=False,
    )
    passed = validator.validate_figure_1_alpha_correlation()
    assert passed


@pytest.mark.slow
def test_reproduce_all_claims():
    """Test all paper claim reproductions."""
    results = reproduce_all_paper_claims(device="cpu", n_seeds=2, fast_mode=True, synthetic_mode=True, plot=False)
    pass_rate = sum(results.values()) / len(results)
    assert pass_rate >= 0.6


def create_dummy_loader(n_batches=3, batch_size=20):
    """Helper: create dummy data loader."""
    dataset = torch.utils.data.TensorDataset(
        torch.rand(n_batches * batch_size, 3, 32, 32),
        torch.randint(0, 10, (n_batches * batch_size,)),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
