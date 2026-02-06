"""
Comprehensive test suite for all fixes.
"""

import pytest
import torch

from neurinspectre.attacks import APGDAttack, APGDTargeted, AttackConfig, PGDAttack


class TestInputRangeInvariance:
    def _run_attack(self, x, y):
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(x.numel() // x.size(0), 10),
        )
        cfg = AttackConfig(epsilon=8 / 255, n_iterations=5, auto_detect_range=True)
        attack = PGDAttack(cfg, device="cpu")
        result = attack.run(model, x, y)
        return result.x_adv

    def test_normalized_cifar10(self):
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])
        x_raw = torch.rand(8, 3, 32, 32)
        x_norm = (x_raw - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        expected_min = ((0.0 - mean) / std).min().item()
        expected_max = ((1.0 - mean) / std).max().item()
        y = torch.randint(0, 10, (8,))

        x_adv = self._run_attack(x_norm, y)
        assert float(x_adv.min().item()) >= expected_min - 0.1
        assert float(x_adv.max().item()) <= expected_max + 0.1
        assert float(x_adv.min().item()) < 0.0

    def test_normalized_imagenet(self):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        x_raw = torch.rand(8, 3, 32, 32)
        x_norm = (x_raw - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        expected_min = ((0.0 - mean) / std).min().item()
        expected_max = ((1.0 - mean) / std).max().item()
        y = torch.randint(0, 10, (8,))

        x_adv = self._run_attack(x_norm, y)
        assert float(x_adv.min().item()) >= expected_min - 0.1
        assert float(x_adv.max().item()) <= expected_max + 0.1
        assert float(x_adv.min().item()) < 0.0

    def test_standard_images(self):
        x = torch.rand(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))

        x_adv = self._run_attack(x, y)
        assert float(x_adv.min().item()) >= 0.0
        assert float(x_adv.max().item()) <= 1.0


class TestAutoAttackParity:
    def test_apgd_deterministic(self):
        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 8 * 8, 10),
        )
        x = torch.rand(4, 3, 8, 8)
        y = torch.randint(0, 10, (4,))

        cfg = AttackConfig(epsilon=8 / 255, n_iterations=5, loss="ce", random_init=True)
        rng_state = torch.random.get_rng_state()
        attack = APGDAttack(cfg, device="cpu")
        res1 = attack.run(model, x, y).x_adv

        torch.random.set_rng_state(rng_state)
        attack2 = APGDAttack(cfg, device="cpu")
        res2 = attack2.run(model, x, y).x_adv

        assert torch.allclose(res1, res2)

    def test_apgd_rollback(self):
        class ConstModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.zeros(1, 10)

            def forward(self, x):
                return self.logits.expand(x.size(0), -1)

        x = torch.rand(8, 3, 8, 8)
        y = torch.randint(0, 10, (8,))
        cfg = AttackConfig(epsilon=8 / 255, n_iterations=12, step_size=2 / 255, loss="ce")
        attack = APGDAttack(cfg, device="cpu", verbose=False)
        attack.run(ConstModel(), x, y)
        assert attack._rollback_count > 0

    def test_apgd_oscillation_detection(self):
        class OscillatingAPGD(APGDAttack):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._loss_calls = 0

            def _compute_loss(self, model, x, y, targeted, target_labels):
                pattern = [0.75, 1.0, 0.5, 0.8, 0.6]
                value = pattern[self._loss_calls % len(pattern)]
                self._loss_calls += 1
                return torch.full((x.size(0),), value)

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 4 * 4, 10),
        )
        x = torch.rand(8, 3, 4, 4)
        y = torch.randint(0, 10, (8,))
        cfg = AttackConfig(epsilon=8 / 255, n_iterations=20, step_size=2 / 255, loss="ce")
        attack = OscillatingAPGD(cfg, device="cpu")
        attack.run(model, x, y)
        assert attack._oscillation_count > 0


class TestTargetedAttacks:
    def test_target_selection_topk(self):
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
        for t, e in zip(targets, expected):
            assert torch.equal(t, e)


class TestL2Initialization:
    def test_l2_init_distribution(self):
        x = torch.full((256, 1, 4, 4), 0.5)
        y = torch.randint(0, 3, (256,))
        cfg = AttackConfig(
            norm="l2",
            epsilon=0.5,
            n_iterations=1,
            step_size=0.0,
            random_init=True,
            auto_detect_range=False,
            input_range=(-10.0, 10.0),
        )
        attack = PGDAttack(cfg, device="cpu")
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))
        res = attack.run(model, x, y)
        delta = (res.x_adv - x).view(x.size(0), -1)
        norms = delta.norm(p=2, dim=1)
        mean_norm = float(norms.mean().item())
        assert float(norms.max().item()) <= 0.5 + 1e-6
        assert mean_norm >= 0.85 * 0.5


class TestEOTConsistency:
    def test_eot_loss_matches_gradient(self):
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


class TestConfigDefaults:
    def test_tg_disabled_by_default(self):
        config = AttackConfig()
        assert config.use_tg is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
