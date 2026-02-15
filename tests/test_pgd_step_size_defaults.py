from __future__ import annotations

import torch
import torch.nn as nn

from neurinspectre.attacks import AttackFactory


def test_pgd_default_step_size_for_l2_is_not_forced_to_2_over_255() -> None:
    # Regression test: PGD runner must not hardcode 2/255 when step_size is unset.
    # For L2 threats, the PGDAttack computes a scale based on eps/iterations.
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2)).eval()
    x = torch.rand(8, 4)
    y = torch.zeros(8, dtype=torch.long)

    runner = AttackFactory.create_attack(
        "pgd",
        model,
        config={"norm": "l2", "epsilon": 1.0, "n_iterations": 5, "random_start": False},
        device="cpu",
    )
    _result = runner.run(x, y)

    step = float(getattr(runner.attack.config, "step_size", 0.0) or 0.0)
    assert step > 0.1

