from types import SimpleNamespace

from neurinspectre.evaluation.table1_evaluator import _apply_overrides


def test_apply_overrides_updates_dataset_and_attack_batch_size():
    config = {
        "datasets": {
            "cifar10": {"n_samples": 10, "batch_size": 5},
            "ember": {"n_samples": 20, "batch_size": 8},
        },
        "defenses": {"jpeg_compression": {}, "randomized_smoothing": {}},
        "cache": {"enable_dataset_cache": True, "enable_attack_checkpoint": True},
        "attack_batch_size": 50,
    }
    args = SimpleNamespace(
        n_samples=100,
        batch_size=16,
        defenses=None,
        no_cache=False,
    )

    updated = _apply_overrides(config, args)

    assert updated["datasets"]["cifar10"]["n_samples"] == 100
    assert updated["datasets"]["ember"]["n_samples"] == 100
    assert updated["datasets"]["cifar10"]["batch_size"] == 16
    assert updated["datasets"]["ember"]["batch_size"] == 16
    assert updated["attack_batch_size"] == 16


def test_apply_overrides_filters_defenses_and_disables_cache():
    config = {
        "datasets": {"cifar10": {"n_samples": 10, "batch_size": 5}},
        "defenses": {"jpeg_compression": {}, "randomized_smoothing": {}},
        "cache": {"enable_dataset_cache": True, "enable_attack_checkpoint": True},
    }
    args = SimpleNamespace(
        n_samples=None,
        batch_size=None,
        defenses=["jpeg_compression"],
        no_cache=True,
    )

    updated = _apply_overrides(config, args)

    assert list(updated["defenses"].keys()) == ["jpeg_compression"]
    assert updated["cache"]["enable_dataset_cache"] is False
    assert updated["cache"]["enable_attack_checkpoint"] is False
