from __future__ import annotations

import json
from pathlib import Path

from neurinspectre.evaluation.artifact_integrity import nuscenes_label_map_hash_gate, sha256_file


def test_nuscenes_label_map_hash_gate_passes_on_match(tmp_path: Path) -> None:
    labels_path = tmp_path / "label_map.json"
    labels_path.write_text(json.dumps({"sample_token_a": 1}), encoding="utf-8")

    model_path = tmp_path / "nuscenes_resnet18_trained.pt"
    model_path.write_bytes(b"")  # placeholder artifact

    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps({"labels_sha256": sha256_file(labels_path)}),
        encoding="utf-8",
    )

    report = nuscenes_label_map_hash_gate(model_path=model_path, labels_path=labels_path)
    assert report["enabled"] is True
    assert report["passed"] is True
    assert report["reasons"] == []


def test_nuscenes_label_map_hash_gate_fails_on_mismatch(tmp_path: Path) -> None:
    labels_path = tmp_path / "label_map.json"
    labels_path.write_text(json.dumps({"sample_token_a": 1}), encoding="utf-8")

    model_path = tmp_path / "nuscenes_resnet18_trained.pt"
    model_path.write_bytes(b"")  # placeholder artifact

    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps({"labels_sha256": sha256_file(labels_path)}),
        encoding="utf-8",
    )

    # Mutate the label map after "training"
    labels_path.write_text(json.dumps({"sample_token_a": 2}), encoding="utf-8")

    report = nuscenes_label_map_hash_gate(model_path=model_path, labels_path=labels_path)
    assert report["enabled"] is True
    assert report["passed"] is False
    assert "labels_sha256_mismatch" in report["reasons"]

