import torch
import torch.nn as nn

from neurinspectre.attacks import APGD, APGDEnsemble


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_apgd_ce():
    model = ToyModel()
    attack = APGD(model, eps=0.3, steps=10, loss="ce", device="cpu")
    x = torch.rand(5, 1, 28, 28)
    y = torch.randint(0, 10, (5,))
    x_adv = attack(x, y)
    assert (x_adv - x).abs().max() <= 0.3 + 1e-6


def test_apgd_dlr():
    model = ToyModel()
    attack = APGD(model, eps=0.3, steps=10, loss="dlr", device="cpu")
    x = torch.rand(5, 1, 28, 28)
    y = torch.randint(0, 10, (5,))
    x_adv = attack(x, y)
    assert not torch.allclose(x, x_adv)


def test_apgd_ensemble():
    model = ToyModel()
    attack = APGDEnsemble(model, eps=0.3, steps=10, losses=["ce", "dlr"], device="cpu")
    x = torch.rand(5, 1, 28, 28)
    y = torch.randint(0, 10, (5,))
    x_adv = attack(x, y)
    assert (x_adv - x).abs().max() <= 0.3 + 1e-6


def test_apgd_default_no_transformed_gradients():
    from neurinspectre.attacks import AttackConfig, APGDAttack

    cfg = AttackConfig()
    attack = APGDAttack(cfg, device="cpu")
    assert cfg.use_tg is False
    assert attack.use_tg is False


def test_tg_disabled_by_default():
    from neurinspectre.attacks import AttackConfig

    config = AttackConfig()
    assert config.use_tg is False, "TG must be disabled by default"
    print("✓ TG disabled by default")


def test_apgd_loss_eot_uses_multiple_forwards():
    import torch
    from neurinspectre.attacks import AttackConfig, APGDAttack

    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x):
            self.calls += 1
            return torch.randn(x.size(0), 10)

    model = CountingModel()
    x = torch.rand(4, 3, 4, 4)
    y = torch.randint(0, 10, (4,))

    cfg = AttackConfig(loss="ce", use_eot=True, eot_samples=3)
    attack = APGDAttack(cfg, device="cpu")
    loss = attack._compute_loss(model, x, y, False, None)

    assert loss.shape[0] == x.shape[0]
    assert model.calls == 3


def test_apgd_t_topk_target_selection():
    import torch
    from neurinspectre.attacks import AttackConfig, APGDTargeted

    logits = torch.tensor(
        [
            [1.0, 5.0, 3.0, 2.0],
            [4.0, 1.0, 0.5, 3.5],
        ]
    )
    y = torch.tensor([1, 0])

    attack = APGDTargeted(AttackConfig(), device="cpu", n_target_classes=3)
    targets = attack._select_target_labels_topk(logits, y, k=3)

    expected = [
        torch.tensor([2, 3]),
        torch.tensor([3, 1]),
        torch.tensor([0, 2]),
    ]

    assert len(targets) == 3
    for t, e in zip(targets, expected):
        assert torch.equal(t, e)
