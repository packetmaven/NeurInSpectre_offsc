from pathlib import Path

import pytest

from neurinspectre.cli.table2_cmd import (
    _normalize_table2_config,
    _validate_strict_real_data,
)


def test_normalize_table2_config_mapping_style():
    raw = {
        "seed": 42,
        "attack_batch_size": 64,
        "datasets": {"cifar10": {"root": "./data/cifar10"}},
        "attacks": {"pgd": {"enabled": True, "steps": 10}},
        "defenses": {
            "jpeg_compression": {
                "enabled": True,
                "dataset": "cifar10",
                "params": {"quality": 75},
                "model_name": "resnet18",
            }
        },
        "attack_budgets": {"cifar10": {"norm": "linf", "eps": 8 / 255}},
    }
    cfg = _normalize_table2_config(raw, strict_dataset_budgets=True)
    assert cfg["seed"] == 42
    assert cfg["strict_dataset_budgets"] is True
    assert isinstance(cfg["attacks"], list) and cfg["attacks"][0]["name"] == "pgd"
    assert isinstance(cfg["defenses"], list) and cfg["defenses"][0]["type"] == "jpeg_compression"
    assert cfg["defenses"][0]["model"]["model_name"] == "resnet18"


def test_validate_strict_real_data_requires_budget_entries(tmp_path: Path):
    root = tmp_path / "data" / "cifar10"
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "datasets": {"cifar10": {"path": str(root)}},
        "defenses": [{"name": "jpeg_compression", "type": "jpeg_compression", "dataset": "cifar10"}],
        # missing attack_budgets on purpose
    }
    with pytest.raises(Exception):
        _validate_strict_real_data(cfg, allow_missing=False, base_dir=tmp_path)
