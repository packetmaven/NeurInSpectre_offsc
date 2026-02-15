from __future__ import annotations

from pathlib import Path

import click
import pytest

from neurinspectre.cli.table2_cmd import _validate_strict_real_data


def test_strict_real_data_refuses_stub_model_by_filename(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "cifar10"
    data_root.mkdir(parents=True)

    model_path = tmp_path / "models" / "cifar10_stub.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"stub")

    cfg = {
        "datasets": {"cifar10": {"path": str(data_root)}},
        "models": {"cifar10": str(model_path)},
        "defenses": [{"name": "d", "type": "jpeg_compression", "dataset": "cifar10"}],
        "attack_budgets": {"cifar10": {"norm": "linf", "eps": 8 / 255}},
    }

    with pytest.raises(click.ClickException) as exc:
        _validate_strict_real_data(cfg, allow_missing=False, base_dir=tmp_path)
    assert "refused placeholder model" in str(exc.value).lower()


def test_strict_real_data_requires_nuscenes_meta_json(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "nuscenes"
    (data_root / "v1.0-mini").mkdir(parents=True)
    labels_path = data_root / "label_map.json"
    labels_path.write_text("{}", encoding="utf-8")

    model_path = tmp_path / "models" / "nuscenes_resnet18_trained.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"artifact")

    cfg = {
        "datasets": {"nuscenes": {"path": str(data_root), "labels_path": str(labels_path)}},
        "models": {"nuscenes": str(model_path)},
        "defenses": [{"name": "d", "type": "spatial_smoothing", "dataset": "nuscenes"}],
        "attack_budgets": {"nuscenes": {"norm": "l2", "eps": 3.0}},
    }

    with pytest.raises(click.ClickException) as exc:
        _validate_strict_real_data(cfg, allow_missing=False, base_dir=tmp_path)
    msg = str(exc.value).lower()
    assert (
        "nuscenes-devkit" in msg
        or "missing_meta_json_for_nuscenes" in msg
        or "missing_labels_sha256_in_meta" in msg
    )

