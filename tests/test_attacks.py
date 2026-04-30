import torch
import torch.nn as nn

from neurinspectre.attacks import (
    APGD,
    APGDEnsemble,
    BPDA,
    EOT,
    FAB,
    AutoAttack,
    AutoAttackEnsemble,
    MemoryAugmentedPGD,
    PGD,
    PGDWithRestarts,
    AttackConfig,
    PGDAttack,
    SquareAttack,
    TemporalMomentumPGD,
)


class ToyModel(nn.Module):
    def __init__(self, in_dim=16, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _make_batch(batch_size=8):
    x = torch.rand(batch_size, 1, 4, 4)
    y = torch.randint(0, 3, (batch_size,))
    return x, y


def test_pgd_bounds_cpu():
    model = ToyModel()
    x, y = _make_batch()
    attack = PGD(model, eps=0.1, alpha=0.02, steps=10, device="cpu")
    adv = attack(x, y)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 0.1001


def test_pgd_targeted_runs_cpu():
    model = ToyModel()
    x, _y = _make_batch()
    targets = torch.randint(0, 3, (x.size(0),))
    attack = PGD(model, eps=0.1, alpha=0.02, steps=10, device="cpu")
    adv = attack(x, targets, targeted=True)
    assert adv.shape == x.shape


def test_pgd_with_restarts_runs_cpu():
    model = ToyModel()
    x, y = _make_batch()
    attack = PGDWithRestarts(model, n_restarts=3, eps=0.1, alpha=0.02, steps=10, device="cpu")
    adv = attack(x, y)
    assert adv.shape == x.shape


def test_apgd_bounds_cpu():
    model = ToyModel()
    x, y = _make_batch()
    attack = APGD(model, eps=0.1, steps=10, loss="dlr", device="cpu")
    adv = attack(x, y)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 0.1001


def test_pgd_l2_random_init_within_ball():
    from neurinspectre.attacks import AttackConfig, PGDAttack

    model = ToyModel()
    x, y = _make_batch()
    cfg = AttackConfig(
        norm="l2",
        epsilon=0.5,
        n_iterations=1,
        step_size=0.0,
        random_init=True,
        auto_detect_range=True,
    )
    attack = PGDAttack(cfg, device="cpu")
    result = attack.run(model, x, y)
    delta = (result.x_adv - x).view(x.size(0), -1)
    norms = delta.norm(p=2, dim=1)
    assert float(norms.max().item()) <= 0.5 + 1e-6


def test_attack_respects_normalized_input_range():
    model = ToyModel()
    x = torch.full((4, 1, 4, 4), -0.5)
    y = torch.randint(0, 3, (4,))
    cfg = AttackConfig(
        epsilon=0.1,
        n_iterations=1,
        step_size=0.0,
        random_init=True,
        auto_detect_range=True,
    )
    attack = PGDAttack(cfg, device="cpu")
    result = attack.run(model, x, y)
    adv = result.x_adv
    assert float(adv.min().item()) < 0.0


def test_memory_augmented_pgd_bounds_cpu():
    model = ToyModel()
    x, y = _make_batch()
    attack = MemoryAugmentedPGD(model, alpha_volterra=0.6, eps=0.1, steps=10, device="cpu")
    adv = attack(x, y)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 0.1001


def test_bpda_bounds_cpu():
    model = ToyModel()
    x, y = _make_batch()

    def defense(inp):
        return torch.round(inp * 8) / 8

    attack = BPDA(model, defense=defense, eps=0.1, steps=10, device="cpu")
    adv = attack(x, y)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 0.1001


def test_eot_bounds_cpu():
    model = ToyModel()
    x, y = _make_batch()

    def transform(inp):
        return inp + torch.randn_like(inp) * 0.01

    attack = EOT(model, transform_fn=transform, eps=0.1, steps=10, num_samples=10, device="cpu")
    adv = attack(x, y)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 0.1001


def test_square_attack_bounds_cpu():
    model = ToyModel()
    x, y = _make_batch()
    attack = SquareAttack(model, eps=0.1, n_queries=1000, device="cpu")
    adv, _stats = attack(x, y)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 0.1001


def test_fab_bounds_cpu():
    model = ToyModel()
    x, y = _make_batch()
    attack = FAB(model, norm="l2", steps=10, n_restarts=1, device="cpu")
    adv = attack(x, y)
    assert adv.shape == x.shape
    assert (adv >= x.min() - 1e-6).all()
    assert (adv <= x.max() + 1e-6).all()


def test_temporal_momentum_bounds_cpu():
    model = ToyModel()
    x, y = _make_batch()
    attack = TemporalMomentumPGD(model, eps=0.1, steps=10, device="cpu")
    adv = attack(x, y)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 0.1001


def test_autoattack_ensemble_runs_cpu():
    model = ToyModel()
    x, y = _make_batch()
    attacks = [
        APGD(model, eps=0.1, steps=10, loss="ce", device="cpu"),
        FAB(model, norm="l2", steps=10, n_restarts=1, device="cpu"),
    ]
    ensemble = AutoAttackEnsemble(model, attacks, device="cpu")
    adv = ensemble(x, y)
    assert adv.shape == x.shape


def test_autoattack_bounds_cpu():
    model = ToyModel()
    x, y = _make_batch()
    attack = AutoAttack(model, norm="linf", eps=0.1, version="standard", device="cpu")
    adv, _metrics = attack.run(x, y, verbose=False)
    assert adv.shape == x.shape
    assert float((adv - x).abs().max()) <= 0.1001
