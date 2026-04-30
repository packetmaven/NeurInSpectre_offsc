from neurinspectre.cli.table2_cmd import _normalize_table2_spec


def test_table2_spec_normalization_sets_per_defense_model_path_fallback():
    raw = {
        "table_id": 2,
        "defaults": {"evaluation": {"batch_size": 8}},
        "datasets": {
            "content_moderation": {"backing_dataset": "cifar10", "split": "test"},
        },
        "attacks": {"pgd": {"enabled": True}},
        "models": {"cifar10": "models/cifar10_cnn_ts.pt"},
        "defenses": [
            {
                "id": "cm_jpeg_compression",
                "enabled": True,
                "domain": "content_moderation",
                "dataset": "content_moderation",
                "model": {"checkpoint_tag": "does_not_exist_anywhere"},
                "defense": {"type": "jpeg_compression", "params": {"quality": 75}},
            }
        ],
    }

    resolved = _normalize_table2_spec(raw, strict_dataset_budgets=True)
    d0 = resolved["defenses"][0]
    assert d0["dataset"] == "cifar10"
    assert d0["type"] == "jpeg_compression"
    assert d0["model_path"] == "models/cifar10_cnn_ts.pt"
    assert isinstance(d0.get("model_spec"), dict)
    assert d0.get("model_provenance", {}).get("fallback") == "dataset_default"


def test_table2_spec_normalization_uses_explicit_model_path_and_sets_ensemble_members():
    raw = {
        "table_id": 2,
        "defaults": {"evaluation": {"batch_size": 8}},
        "datasets": {
            "content_moderation": {"backing_dataset": "cifar10", "split": "test"},
        },
        "attacks": {"pgd": {"enabled": True}},
        "models": {"cifar10": "models/cifar10_cnn_ts.pt"},
        "defenses": [
            {
                "id": "cm_ensemble_diversity",
                "enabled": True,
                "domain": "content_moderation",
                "dataset": "content_moderation",
                "model": {"path": "models/cifar10_resnet20_norm_ts.pt"},
                "defense": {"type": "ensemble_diversity", "params": {"ensemble_size": 3, "voting": "majority"}},
            }
        ],
    }

    resolved = _normalize_table2_spec(raw, strict_dataset_budgets=True)
    d0 = resolved["defenses"][0]
    assert d0["model_path"] == "models/cifar10_resnet20_norm_ts.pt"
    assert "model_provenance" not in d0

    members = d0.get("params", {}).get("members")
    assert isinstance(members, list)
    assert len(members) == 3
    assert all(m.get("path") == "models/cifar10_resnet20_norm_ts.pt" for m in members)

