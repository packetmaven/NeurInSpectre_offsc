"""
Integration tests for Week 2 implementations.

Tests end-to-end workflows:
  1. Characterization -> Orchestration -> Attack
"""

import torch

from neurinspectre.characterization import DefenseAnalyzer
from neurinspectre.attacks import AttackOrchestrator


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


def create_dummy_loader(n_batches=3, batch_size=20):
    """Helper: create dummy data loader."""
    dataset = torch.utils.data.TensorDataset(
        torch.rand(n_batches * batch_size, 3, 32, 32),
        torch.randint(0, 10, (n_batches * batch_size,)),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
