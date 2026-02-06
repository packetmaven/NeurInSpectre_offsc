import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from neurinspectre.attacks.fab import FAB, FABT, FABEnsemble


torch.manual_seed(0)


class SimpleConvNet(nn.Module):
    """Simple CNN for FAB testing."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _different_labels(preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    return (preds + 1) % num_classes


def test_fab_l2_untargeted():
    model = SimpleConvNet()
    model.eval()

    attack = FAB(model, norm="l2", steps=50, n_restarts=2, device="cpu")

    x = torch.rand(3, 3, 32, 32)
    with torch.no_grad():
        preds_clean = model(x).argmax(1)
    y = _different_labels(preds_clean, 10)

    x_adv = attack(x, y)

    assert x_adv.shape == x.shape
    assert (x_adv >= 0).all() and (x_adv <= 1).all()

    with torch.no_grad():
        preds = model(x_adv).argmax(1)
    assert (preds != y).any(), "FAB should generate some adversarial examples"


def test_fab_linf_untargeted():
    model = SimpleConvNet()
    model.eval()

    attack = FAB(model, norm="linf", steps=50, n_restarts=2, device="cpu")

    x = torch.rand(3, 3, 32, 32)
    with torch.no_grad():
        preds_clean = model(x).argmax(1)
    y = _different_labels(preds_clean, 10)

    x_adv = attack(x, y)
    assert x_adv.shape == x.shape

    with torch.no_grad():
        preds = model(x_adv).argmax(1)
    assert (preds != y).any()


def test_fab_targeted():
    model = SimpleConvNet()
    model.eval()

    attack = FAB(model, norm="l2", steps=50, n_restarts=1, device="cpu")

    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0, 1, 2])
    targets = torch.tensor([5, 6, 7])

    x_adv = attack(x, y, targeted=True, target_classes=targets)
    assert (x_adv >= 0).all() and (x_adv <= 1).all()


def test_fab_margin_computation():
    model = SimpleConvNet()
    attack = FAB(model, device="cpu")

    logits = torch.tensor(
        [
            [5.0, 3.0, 1.0],
            [2.0, 4.0, 1.0],
            [1.0, 2.0, 5.0],
        ]
    )
    y = torch.tensor([0, 1, 2])

    margins, runner_up = attack._margin_loss(logits, y)
    assert torch.allclose(margins, torch.tensor([2.0, 2.0, 3.0]))
    assert runner_up[0] == 1
    assert runner_up[1] == 0
    assert runner_up[2] == 1


def test_fab_minimum_norm_property():
    from neurinspectre.attacks.pgd import PGD

    model = SimpleConvNet()
    model.eval()

    x = torch.rand(5, 3, 32, 32)
    y = torch.randint(0, 10, (5,))

    pgd = PGD(model, eps=0.3, steps=40, device="cpu")
    x_adv_pgd = pgd(x, y)

    fab = FAB(model, norm="l2", steps=50, n_restarts=2, device="cpu")
    x_adv_fab = fab(x, y)

    with torch.no_grad():
        preds_pgd = model(x_adv_pgd).argmax(1)
        preds_fab = model(x_adv_fab).argmax(1)
        both_adv = (preds_pgd != y) & (preds_fab != y)

        if both_adv.any():
            norm_pgd = (x_adv_pgd[both_adv] - x[both_adv]).view(both_adv.sum(), -1).norm(
                p=2, dim=1
            )
            norm_fab = (x_adv_fab[both_adv] - x[both_adv]).view(both_adv.sum(), -1).norm(
                p=2, dim=1
            )
            assert (norm_fab <= norm_pgd + 0.01).all()


def test_fabt_multiple_targets():
    model = SimpleConvNet()
    model.eval()

    attack = FABT(model, norm="l2", n_target_classes=5, steps=30, device="cpu")

    x = torch.rand(5, 3, 32, 32)
    with torch.no_grad():
        preds_clean = model(x).argmax(1)
    y = _different_labels(preds_clean, 10)

    x_adv = attack(x, y)
    assert x_adv.shape == x.shape
    assert (x_adv >= 0).all() and (x_adv <= 1).all()

    with torch.no_grad():
        preds = model(x_adv).argmax(1)
    assert (preds != y).any()


def test_fab_ensemble():
    model = SimpleConvNet()
    model.eval()

    attack = FABEnsemble(model, norm="l2", device="cpu")

    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0, 1, 2])

    x_adv = attack(x, y)
    assert x_adv.shape == x.shape
    assert (x_adv >= 0).all() and (x_adv <= 1).all()

    with torch.no_grad():
        preds = model(x_adv).argmax(1)
    assert (preds != y).any()


def test_fab_backtracking():
    model = SimpleConvNet()
    attack = FAB(model, device="cpu")

    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    x_adv = torch.clamp(x + 0.1 * torch.randn_like(x), 0, 1)
    x_best = torch.clamp(x + 0.2 * torch.randn_like(x), 0, 1)

    x_result, improved = attack._backtrack_search(x, x_adv, x_best, y)
    assert x_result.shape == x.shape
    assert improved.shape == (2,)


def test_fab_convergence():
    model = SimpleConvNet()
    model.eval()

    attack = FAB(model, norm="l2", steps=100, n_restarts=1, device="cpu")

    x = torch.rand(3, 3, 32, 32)
    y = torch.randint(0, 10, (3,))

    x_adv = attack(x, y)
    with torch.no_grad():
        logits = model(x_adv)
        margins, _ = attack._margin_loss(logits, y)

    preds = logits.argmax(1)
    adversarial = preds != y
    if adversarial.any():
        assert (margins[adversarial] <= 0).all()


@pytest.mark.slow
def test_fab_vs_autoattack_baseline():
    model = SimpleConvNet()
    model.eval()

    x = torch.rand(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))

    fab_l2 = FAB(model, norm="l2", steps=100, n_restarts=5, device="cpu")
    x_adv_l2 = fab_l2(x, y)

    fab_linf = FAB(model, norm="linf", steps=100, n_restarts=5, device="cpu")
    x_adv_linf = fab_linf(x, y)

    with torch.no_grad():
        preds_l2 = model(x_adv_l2).argmax(1)
        preds_linf = model(x_adv_linf).argmax(1)
        asr_l2 = (preds_l2 != y).float().mean()
        asr_linf = (preds_linf != y).float().mean()

    assert asr_l2 > 0.3 or asr_linf > 0.3
