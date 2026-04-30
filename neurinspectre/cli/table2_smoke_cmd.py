"""One-command real-data Table2 smoke matrix.

This is intended for AE-friendly fast retry loops: it discovers runnable datasets
and models on disk, generates a small Table2-style config, and runs `table2`
with strict real-data validation enabled (including validity gates).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import click
import yaml

from ..evaluation.artifact_integrity import load_model_meta
from .table2_cmd import run_table2


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _looks_like_placeholder_model(path: Path, *, dataset: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    name = path.name.lower()
    if any(marker in name for marker in ("stub", "placeholder", "dummy")):
        reasons.append("filename_contains_stub_marker")

    meta = load_model_meta(path)
    if isinstance(meta, dict):
        if bool(meta.get("is_stub", False)) or bool(meta.get("stub", False)) or bool(meta.get("placeholder", False)):
            reasons.append("meta_marks_stub")

    if str(dataset).lower() == "nuscenes":
        # For nuScenes, require model-side metadata that pins the label-map semantics.
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.exists():
            reasons.append("missing_meta_json_for_nuscenes")
        elif not meta or not meta.get("labels_sha256"):
            reasons.append("missing_labels_sha256_in_meta")

    return (len(reasons) > 0), reasons


def _pick_first_runnable_model(candidates: List[Path], *, dataset: str) -> Path | None:
    for cand in candidates:
        if not cand.exists():
            continue
        is_placeholder, _reasons = _looks_like_placeholder_model(cand, dataset=dataset)
        if is_placeholder:
            continue
        return cand
    return None


def _ember_ready(root: Path) -> bool:
    # Vectorized features are required for a quick smoke run.
    data_dir = root / "ember_2018"
    return (data_dir / "X_test.dat").exists() and (data_dir / "y_test.dat").exists()


def _nuscenes_ready(root: Path, *, labels_path: Path) -> bool:
    return root.exists() and labels_path.exists() and (root / "v1.0-mini").exists()


def _imagenet100_ready(root: Path) -> bool:
    # ImageNet-style layout: root/{train,val}/<class>/*
    val_dir = root / "val"
    train_dir = root / "train"
    if val_dir.exists():
        # Require at least one class directory to avoid an "empty but exists" false positive.
        try:
            return any(p.is_dir() for p in val_dir.iterdir())
        except Exception:
            return False
    if train_dir.exists():
        try:
            return any(p.is_dir() for p in train_dir.iterdir())
        except Exception:
            return False
    return False


def _as_posix(path: Path) -> str:
    # Click/yaml configs are path-string based; keep deterministic formatting.
    return path.as_posix()


def run_table2_smoke(ctx: click.Context, **kwargs: Any) -> None:
    """
    Generate and run a strict real-data Table2 smoke matrix.

    This command intentionally avoids shipping any baseline numbers or paper
    reproduction scripts; it only checks that the pipeline is runnable on real
    assets and that validity/integrity gates pass.
    """

    output_dir = Path(str(kwargs.get("output_dir", "results/table2_smoke_real")))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(str(kwargs.get("data_root", "./data"))).resolve()
    models_root = Path(str(kwargs.get("models_root", "./models"))).resolve()

    # Candidate datasets. We only include datasets that look runnable *and* have
    # a non-placeholder model available.
    datasets: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, Any] = {}
    defenses: List[Dict[str, Any]] = []
    budgets: Dict[str, Dict[str, Any]] = {}

    # CIFAR-10 (downloadable; include by default if a model exists)
    cifar_root = data_root / "cifar10"
    cifar_root.mkdir(parents=True, exist_ok=True)
    cifar_model = _pick_first_runnable_model(
        [
            models_root / "cifar10_resnet20_norm_ts.pt",
            Path.cwd() / "_cli_runs" / "cifar10_resnet20_norm_ts.pt",
            models_root / "cifar10_cnn_ts.pt",
        ],
        dataset="cifar10",
    )
    if cifar_model is not None:
        datasets["cifar10"] = {
            "path": _as_posix(cifar_root),
            "split": "test",
            "num_samples": int(kwargs.get("cifar10_samples", 256)),
            "batch_size": int(kwargs.get("cifar10_batch_size", 64)),
            "num_workers": 0,
        }
        models["cifar10"] = _as_posix(cifar_model)
        budgets["cifar10"] = {"norm": "linf", "eps": float(8 / 255)}
        defenses.append(
            {
                "name": "smoke_cifar10_jpeg",
                "type": "jpeg_compression",
                "dataset": "cifar10",
                "params": {"quality": 75},
            }
        )

    # ImageNet-100 (optional; only if dataset layout exists)
    imagenet_root = data_root / "imagenet"
    imagenet_model = _pick_first_runnable_model(
        [
            models_root / "imagenet100_resnet50_trained.pt",
            models_root / "imagenet100_resnet50.pt",
            models_root / "imagenet100_resnet50_stub.pt",
        ],
        dataset="imagenet100",
    )
    imagenet_has_data = _imagenet100_ready(imagenet_root)
    if imagenet_model is not None and imagenet_has_data:
        datasets["imagenet100"] = {
            "path": _as_posix(imagenet_root),
            "split": "val",
            "num_samples": int(kwargs.get("imagenet100_samples", 64)),
            "batch_size": int(kwargs.get("imagenet100_batch_size", 8)),
            "num_workers": 0,
        }
        models["imagenet100"] = _as_posix(imagenet_model)
        budgets["imagenet100"] = {"norm": "linf", "eps": float(8 / 255)}
        defenses.append(
            {
                "name": "smoke_imagenet100_jpeg",
                "type": "jpeg_compression",
                "dataset": "imagenet100",
                "params": {"quality": 75},
            }
        )
    elif imagenet_has_data and imagenet_model is None:
        click.echo(
            "[table2-smoke] Skipping imagenet100: no non-placeholder TorchScript model found under models_root."
        )

    # EMBER (optional; require vectorized files for a fast smoke run)
    ember_root = data_root / "ember"
    ember_model = _pick_first_runnable_model([models_root / "ember_mlp_ts.pt"], dataset="ember")
    ember_has_data = _ember_ready(ember_root)
    if ember_model is not None and ember_has_data:
        datasets["ember"] = {
            "path": _as_posix(ember_root),
            "split": "test",
            "num_samples": int(kwargs.get("ember_samples", 512)),
            "batch_size": int(kwargs.get("ember_batch_size", 256)),
            "num_workers": 0,
        }
        models["ember"] = _as_posix(ember_model)
        budgets["ember"] = {"norm": "l2", "eps": 0.5}
        # Keep the malware smoke row scientifically meaningful: a defense that
        # destroys clean accuracy will immediately fail strict validity gates.
        # For fast AE retry loops, we default to an identity ("none") defense.
        defenses.append(
            {
                "name": "smoke_ember_none",
                "type": "none",
                "dataset": "ember",
            }
        )
    elif ember_model is not None and not ember_has_data:
        click.echo(
            "[table2-smoke] Skipping ember: vectorized features missing under data_root. "
            "Expected: data/ember/ember_2018/{X_test.dat,y_test.dat}. "
            f"Generate via: {sys.executable} scripts/vectorize_ember_safe.py"
        )

    # nuScenes (optional; require label_map.json + mini split)
    nuscenes_root = data_root / "nuscenes"
    labels_path = nuscenes_root / "label_map.json"
    nuscenes_model = _pick_first_runnable_model(
        [models_root / "nuscenes_resnet18_trained.pt", models_root / "nuscenes_resnet18.pt"],
        dataset="nuscenes",
    )
    nuscenes_has_data = _nuscenes_ready(nuscenes_root, labels_path=labels_path)
    nuscenes_has_dep = _module_available("nuscenes")
    if nuscenes_model is not None and nuscenes_has_data and nuscenes_has_dep:
        datasets["nuscenes"] = {
            "path": _as_posix(nuscenes_root),
            "split": "val",
            "version": "v1.0-mini",
            "labels_path": _as_posix(labels_path),
            "num_samples": int(kwargs.get("nuscenes_samples", 128)),
            "batch_size": int(kwargs.get("nuscenes_batch_size", 8)),
            "num_workers": 0,
        }
        models["nuscenes"] = _as_posix(nuscenes_model)
        budgets["nuscenes"] = {"norm": "l2", "eps": 3.0}
        defenses.append(
            {
                "name": "smoke_nuscenes_spatial_smoothing",
                "type": "spatial_smoothing",
                "dataset": "nuscenes",
                "params": {"kernel_size": 3, "sigma": 1.0},
            }
        )
    elif nuscenes_has_data and not nuscenes_has_dep:
        click.echo(
            "[table2-smoke] Skipping nuscenes: optional dependency missing. "
            f"Install with: {sys.executable} -m pip install nuscenes-devkit"
        )

    if not datasets or not defenses or not models:
        raise click.ClickException(
            "No runnable datasets found for table2-smoke. "
            "Expected at least one (dataset + model) pair under "
            f"data_root={data_root} models_root={models_root}."
        )

    cfg: Dict[str, Any] = {
        "seed": int(kwargs.get("seed", 42)),
        "attack_batch_size": int(kwargs.get("attack_batch_size", 64)),
        "attack_budgets": budgets,
        "datasets": datasets,
        "models": models,
        "defenses": defenses,
        # Keep the smoke run fast but exercise the full runner stack.
        "attacks": [
            {"name": "pgd", "steps": int(kwargs.get("pgd_steps", 10)), "random_start": True},
            {"name": "neurinspectre", "characterization_samples": 10, "steps": int(kwargs.get("neurinspectre_steps", 10))},
        ],
    }

    cfg_path = output_dir / "table2_smoke_real_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    click.echo(f"[table2-smoke] Wrote generated config: {cfg_path}")

    click.echo(
        "[table2-smoke] Included datasets: " + ", ".join(sorted(datasets.keys()))
    )

    run_table2(
        ctx,
        config=str(cfg_path),
        verbose=kwargs.get("verbose", 0),
        output_dir=str(output_dir),
        strict_real_data=True,
        strict_dataset_budgets=True,
        allow_missing=False,
        json_output=kwargs.get("json_output"),
        sarif_output=kwargs.get("sarif_output"),
        report_format=kwargs.get("report_format", "rich"),
        report=kwargs.get("report", True),
        brief=kwargs.get("brief", False),
        summary_only=kwargs.get("summary_only", False),
        color=kwargs.get("color", False),
        no_color=kwargs.get("no_color", False),
        no_progress=kwargs.get("no_progress", False),
        defenses=tuple(kwargs.get("defenses") or ()),
        attacks=tuple(kwargs.get("attacks") or ()),
        parallel=int(kwargs.get("parallel", 1)),
        resume=bool(kwargs.get("resume", False)),
        device=str(kwargs.get("device", "auto")),
    )

