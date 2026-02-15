from neurinspectre.evaluation.budgets import (
    apply_dataset_budget,
    resolve_attack_config,
    resolve_dataset_budget,
)


def test_resolve_dataset_budget_with_alias():
    cfg = {
        "attack_budgets": {
            "imagenet100": {"eps": 0.031, "norm": "linf"},
        }
    }
    budget, key = resolve_dataset_budget(cfg, "imagenet", aliases=["imagenet100"])
    assert key == "imagenet100"
    assert budget["eps"] == 0.031


def test_apply_dataset_budget_non_strict_only_fills_missing():
    attack_cfg = {"epsilon": 0.5, "norm": "l2"}
    budget_cfg = {"eps": 0.031, "norm": "linf", "alpha": 0.01}
    merged = apply_dataset_budget(attack_cfg, budget_cfg, strict_budget=False)
    assert merged["epsilon"] == 0.5
    assert merged["norm"] == "l2"
    assert merged["alpha"] == 0.01


def test_apply_dataset_budget_strict_overrides_existing():
    attack_cfg = {"epsilon": 0.5, "norm": "l2", "alpha": 0.2}
    budget_cfg = {"eps": 0.031, "norm": "linf", "alpha": 0.01}
    merged = apply_dataset_budget(attack_cfg, budget_cfg, strict_budget=True)
    assert merged["epsilon"] == 0.031
    assert merged["norm"] == "linf"
    assert merged["alpha"] == 0.01


def test_resolve_attack_config_tracks_budget_source():
    cfg = {"attack_budgets": {"ember": {"eps": 1.0, "norm": "l2"}}}
    merged = resolve_attack_config(
        cfg,
        attack_cfg={"n_iterations": 10},
        dataset_name="ember",
        strict_budget=False,
    )
    assert merged["epsilon"] == 1.0
    assert merged["norm"] == "l2"
    assert merged["dataset_budget_source"] == "ember"
