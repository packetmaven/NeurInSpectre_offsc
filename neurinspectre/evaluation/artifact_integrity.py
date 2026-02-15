"""Artifact integrity gates.

These checks are AE-friendly: they prevent silent mismatches between a model
artifact and dataset-side auxiliary assets (e.g., nuScenes label_map.json used to
define the classification proxy task).

This module intentionally does not contain any paper baseline numbers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_file(path: str | Path) -> str:
    """Compute SHA256 of a file (streamed)."""

    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_meta(model_path: str | Path) -> Optional[Dict[str, Any]]:
    """
    Load a model-side metadata JSON if present.

    Convention: `<model_path>.meta.json`, e.g. `model.pt.meta.json` as produced by
    `scripts/train_nuscenes_real.py`.
    """

    model_p = Path(model_path)
    meta_path = model_p.with_suffix(model_p.suffix + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def nuscenes_label_map_hash_gate(*, model_path: str | Path, labels_path: str | Path) -> Dict[str, Any]:
    """
    Verify that the nuScenes label-map hash matches the model metadata.

    The training script writes `labels_sha256` into `<model_path>.meta.json`.
    This gate compares that expected SHA256 to the observed SHA256 of the
    `labels_path` file used at evaluation time.
    """

    model_p = Path(model_path)
    labels_p = Path(labels_path)
    meta_path = model_p.with_suffix(model_p.suffix + ".meta.json")

    report: Dict[str, Any] = {
        "enabled": True,
        "passed": False,
        "model_path": str(model_p),
        "meta_path": str(meta_path),
        "labels_path": str(labels_p),
        "expected_labels_sha256": None,
        "observed_labels_sha256": None,
        "reasons": [],
    }

    if not labels_p.exists():
        report["reasons"].append("missing_labels_path")
        return report

    if not meta_path.exists():
        report["reasons"].append("missing_model_meta")
        return report

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        report["reasons"].append("invalid_model_meta_json")
        return report

    expected = meta.get("labels_sha256")
    if not expected:
        report["reasons"].append("missing_labels_sha256_in_meta")
        return report

    expected = str(expected)
    report["expected_labels_sha256"] = expected

    observed = sha256_file(labels_p)
    report["observed_labels_sha256"] = observed

    if observed != expected:
        report["reasons"].append("labels_sha256_mismatch")
        return report

    report["passed"] = True
    return report

