"""Table 2 evaluation command with strict real-data validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

import click
import yaml

from ..evaluation.artifact_integrity import load_model_meta
from ..evaluation.budgets import get_attack_budgets
from .evaluate_cmd import run_evaluation
from .utils import load_yaml

REAL_DATASETS = {"cifar10", "imagenet100", "ember", "nuscenes"}


def _looks_like_table2_spec(raw: Dict[str, Any]) -> bool:
    """
    Detect the legacy/spec-style Table 2 YAML (``table2_config.yaml``).

    That format uses dataset *groups* with ``backing_dataset`` indirection and
    embeds non-runnable model metadata (factory_key/checkpoint_tag) that needs
    normalization before it can be executed via the Click CLI.
    """
    if int(raw.get("table_id", 0) or 0) != 2:
        return False
    if not isinstance(raw.get("defaults"), dict):
        return False
    if not isinstance(raw.get("datasets"), dict):
        return False
    if not isinstance(raw.get("attacks"), dict):
        return False
    if not isinstance(raw.get("defenses"), list):
        return False
    return any(
        isinstance(v, dict) and "backing_dataset" in v for v in (raw.get("datasets") or {}).values()
    )


def _default_dataset_path(dataset_name: str) -> str:
    key = str(dataset_name).lower()
    defaults = {
        "cifar10": "./data/cifar10",
        # ImageNet-100 loader expects train/val directly under root.
        "imagenet100": "./data/imagenet",
        "ember": "./data/ember",
        "nuscenes": "./data/nuscenes",
    }
    return defaults.get(key, f"./data/{key}")


def _default_model_path(dataset_name: str) -> str | None:
    key = str(dataset_name).lower()
    candidates = {
        "cifar10": [
            "models/cifar10_resnet20_norm_ts.pt",
            "_cli_runs/cifar10_resnet20_norm_ts.pt",
            "models/cifar10_cnn_ts.pt",
        ],
        "imagenet100": ["models/imagenet100_resnet50.pt"],
        "ember": ["models/ember_mlp_ts.pt"],
        # Prefer trained artifacts; fall back to older/stub paths if present.
        "nuscenes": ["models/nuscenes_resnet18_trained.pt", "models/nuscenes_resnet18.pt", "models/nuscenes_resnet18_stub.pt"],
    }.get(key, [])
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _normalize_table2_spec(
    raw: Dict[str, Any],
    *,
    strict_dataset_budgets: bool,
) -> Dict[str, Any]:
    defaults = dict(raw.get("defaults", {}) or {})
    eval_defaults = dict(defaults.get("evaluation", {}) or {})

    # Map dataset-group -> real dataset (e.g., "content_moderation" -> "cifar10")
    group_to_backing: Dict[str, str] = {}
    for group_name, ds_cfg in dict(raw.get("datasets", {}) or {}).items():
        if not isinstance(ds_cfg, dict):
            continue
        backing = ds_cfg.get("backing_dataset") or group_name
        group_to_backing[str(group_name)] = str(backing).lower()

    cfg: Dict[str, Any] = {}
    cfg["seed"] = int(raw.get("seed", 42))
    if isinstance(raw.get("seeds"), (list, tuple)) and raw.get("seeds"):
        cfg["seeds"] = [int(s) for s in list(raw.get("seeds") or [])]
    cfg["attack_batch_size"] = int(raw.get("attack_batch_size", eval_defaults.get("batch_size", 128)))
    cfg["cache"] = dict(raw.get("cache", {}) or {})
    cfg["strict_dataset_budgets"] = bool(strict_dataset_budgets)

    # Datasets: materialize concrete dataset configs keyed by backing dataset.
    datasets_out: Dict[str, Dict[str, Any]] = {}
    for group_name, ds_cfg in dict(raw.get("datasets", {}) or {}).items():
        if not isinstance(ds_cfg, dict):
            continue
        backing = str(ds_cfg.get("backing_dataset") or group_name).lower()
        out = dict(datasets_out.get(backing, {}) or {})

        root = ds_cfg.get("path") or ds_cfg.get("root") or out.get("path") or _default_dataset_path(backing)
        out["path"] = root

        if "split" in ds_cfg:
            out["split"] = ds_cfg.get("split")
        if backing == "nuscenes":
            out.setdefault("version", ds_cfg.get("version", "v1.0-mini"))
            out.setdefault("labels_path", ds_cfg.get("labels_path", "./data/nuscenes/label_map.json"))

        out.setdefault("batch_size", int(eval_defaults.get("batch_size", 128)))
        out.setdefault("num_workers", int(eval_defaults.get("num_workers", 0)))
        out.setdefault("shuffle_eval", bool(eval_defaults.get("shuffle_eval", False)))

        if bool(out.get("download", False)):
            out["download"] = False

        datasets_out[backing] = out
    cfg["datasets"] = _normalize_datasets(datasets_out)

    # Models: ignore spec-only model metadata; use per-dataset TorchScript artifacts if present.
    models_out: Dict[str, Any] = dict(raw.get("models", {}) or {})
    for ds_name in cfg["datasets"].keys():
        if ds_name in models_out:
            continue
        p = _default_model_path(ds_name)
        if p is not None:
            models_out[ds_name] = p
    if models_out:
        cfg["models"] = models_out

    # Attacks: resolve defaults via config_key indirection.
    attacks_out: List[Dict[str, Any]] = []
    for attack_name, attack_spec in dict(raw.get("attacks", {}) or {}).items():
        if not isinstance(attack_spec, dict):
            continue
        if not bool(attack_spec.get("enabled", True)):
            continue
        config_key = attack_spec.get("config_key")
        resolved = dict(defaults.get(str(config_key), {}) or {}) if config_key else {}
        resolved["name"] = str(attack_name)
        if str(attack_name).lower() == "neurinspectre":
            resolved.setdefault("characterization_samples", 50)
        attacks_out.append(resolved)
    cfg["attacks"] = _normalize_attacks(attacks_out)

    # Defenses: flatten nested params into top-level kwargs for DefenseFactory.
    defenses_out: List[Dict[str, Any]] = []
    for item in list(raw.get("defenses", []) or []):
        if not isinstance(item, dict):
            continue
        if not bool(item.get("enabled", True)):
            continue
        defense_block = item.get("defense", {}) or {}
        if not isinstance(defense_block, dict):
            defense_block = {}

        d_type_raw = defense_block.get("type") or item.get("type") or item.get("id", "defense")
        d_type = _normalize_defense_type(str(d_type_raw))
        if d_type == "certified_randomized_smoothing":
            d_type = "certified_defense"

        dataset_group = str(item.get("dataset", "") or "")
        dataset_name = group_to_backing.get(dataset_group, dataset_group).lower()

        entry: Dict[str, Any] = {
            "name": str(item.get("id") or item.get("name") or d_type),
            "type": str(d_type),
            "dataset": dataset_name,
        }
        if "domain" in item:
            entry["domain"] = item.get("domain")

        params = dict(defense_block.get("params", item.get("params", {})) or {})
        # Parameter aliases to match ``DefenseFactory`` expectations.
        if d_type == "randomized_smoothing" and "num_samples" in params and "n_samples" not in params:
            params["n_samples"] = params.pop("num_samples")
        if d_type == "gradient_regularization" and "lambda_reg" in params and "lambda_grad" not in params:
            params["lambda_grad"] = params.pop("lambda_reg")
        if d_type == "random_pad_crop" and "padding" in params and "max_pad" not in params:
            params["max_pad"] = params.pop("padding")
        if d_type == "ensemble_diversity":
            n_models = int(params.pop("ensemble_size", 3))
            model_ref = None
            if isinstance(cfg.get("models"), dict):
                model_ref = cfg["models"].get(dataset_name)
            if model_ref and "members" not in params:
                params["members"] = [{"path": model_ref} for _ in range(max(1, n_models))]

        if params:
            entry["params"] = params
        defenses_out.append(entry)

    cfg["defenses"] = _normalize_defenses(defenses_out)

    if "baseline_validation" in raw:
        cfg["baseline_validation"] = dict(raw.get("baseline_validation", {}) or {})
    if "validity_gates" in raw:
        cfg["validity_gates"] = dict(raw.get("validity_gates", {}) or {})

    budgets = get_attack_budgets(raw)
    if not budgets:
        budgets = dict(raw.get("attack_budgets", {}) or {})
    if budgets:
        cfg["attack_budgets"] = budgets

    return cfg


def run_table2(ctx: click.Context, **kwargs: Any) -> None:
    config_path = str(kwargs.get("config"))
    output_dir = Path(str(kwargs.get("output_dir", "results/table2")))
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_real_data = bool(kwargs.get("strict_real_data", True))
    strict_dataset_budgets = bool(kwargs.get("strict_dataset_budgets", True))
    allow_missing = bool(kwargs.get("allow_missing", False))

    raw = load_yaml(config_path)
    resolved = _normalize_table2_config(
        raw,
        strict_dataset_budgets=strict_dataset_budgets,
    )

    if strict_real_data:
        _validate_strict_real_data(
            resolved,
            allow_missing=allow_missing,
            base_dir=Path(config_path).resolve().parent,
        )

        # Scientific validity gate: in strict real-data mode, fail fast if a model
        # has ~0 clean accuracy (ASR becomes non-informative / undefined).
        gates = dict(resolved.get("validity_gates", {}) or {})
        gates.setdefault("enabled", True)
        gates.setdefault("strict", True)
        gates.setdefault("min_clean_accuracy", 0.05)
        gates.setdefault("min_correct_samples", 5)
        gates.setdefault("min_correct_fraction", 0.10)
        # Chance-aware gate: require clean accuracy meaningfully above uniform chance (1/K).
        gates.setdefault("min_clean_accuracy_over_chance", 0.05)
        # Artifact integrity: for nuScenes, enforce that label_map.json matches model meta.
        gates.setdefault("require_label_map_sha256_match", True)
        resolved["validity_gates"] = gates

    resolved_path = output_dir / "resolved_table2_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    click.echo(f"Resolved Table 2 config written to {resolved_path}")

    run_evaluation(
        ctx,
        config=str(resolved_path),
        verbose=kwargs.get("verbose", 0),
        output_dir=str(output_dir),
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
        seeds=tuple(kwargs.get("seeds") or ()),
        num_seeds=int(kwargs.get("num_seeds", 1)),
    )


def _normalize_table2_config(
    raw: Dict[str, Any],
    *,
    strict_dataset_budgets: bool,
) -> Dict[str, Any]:
    if _looks_like_table2_spec(raw):
        return _normalize_table2_spec(raw, strict_dataset_budgets=strict_dataset_budgets)

    cfg: Dict[str, Any] = {}
    evaluation_cfg = dict(raw.get("evaluation", {}) or {})

    cfg["seed"] = int(raw.get("seed", 42))
    if isinstance(raw.get("seeds"), (list, tuple)) and raw.get("seeds"):
        cfg["seeds"] = [int(s) for s in list(raw.get("seeds") or [])]
    cfg["attack_batch_size"] = int(raw.get("attack_batch_size", evaluation_cfg.get("batch_size", 128)))
    cfg["cache"] = dict(raw.get("cache", {}) or {})
    cfg["datasets"] = _normalize_datasets(dict(raw.get("datasets", {}) or {}))
    cfg["attacks"] = _normalize_attacks(raw.get("attacks", {}))
    cfg["defenses"] = _normalize_defenses(raw.get("defenses", {}))
    cfg["strict_dataset_budgets"] = bool(strict_dataset_budgets)

    # Preserve model references so ``evaluate_cmd`` can resolve them per dataset/defense.
    if "models" in raw and isinstance(raw.get("models"), dict):
        cfg["models"] = dict(raw.get("models", {}) or {})
    if "model" in raw:
        cfg["model"] = raw.get("model")
    if "model_name" in raw:
        cfg["model_name"] = raw.get("model_name")

    # Preserve optional baseline validation gate config if provided.
    if "baseline_validation" in raw:
        cfg["baseline_validation"] = dict(raw.get("baseline_validation", {}) or {})
    if "validity_gates" in raw:
        cfg["validity_gates"] = dict(raw.get("validity_gates", {}) or {})

    budgets = get_attack_budgets(raw)
    if not budgets:
        budgets = dict(raw.get("attack_budgets", {}) or {})
    if budgets:
        cfg["attack_budgets"] = budgets

    return cfg


def _normalize_datasets(datasets_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, raw in (datasets_cfg or {}).items():
        ds = dict(raw or {})
        if "root" in ds and "path" not in ds:
            ds["path"] = ds["root"]
        # strict mode should avoid implicit downloads
        if "download" in ds and bool(ds["download"]):
            ds["download"] = False
        out[str(name)] = ds
    return out


def _normalize_attacks(attacks_cfg: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if isinstance(attacks_cfg, dict):
        for name, raw in attacks_cfg.items():
            cfg = dict(raw or {})
            if not bool(cfg.get("enabled", True)):
                continue
            cfg.pop("enabled", None)
            cfg["name"] = str(name)
            entries.append(cfg)
        return entries
    if isinstance(attacks_cfg, list):
        for item in attacks_cfg:
            if isinstance(item, str):
                entries.append({"name": item})
            elif isinstance(item, dict):
                if not bool(item.get("enabled", True)):
                    continue
                cfg = dict(item)
                cfg.pop("enabled", None)
                cfg.setdefault("name", cfg.get("type", "attack"))
                entries.append(cfg)
        return entries
    raise ValueError("attacks must be a mapping or list.")


def _normalize_defense_type(defense_type: str) -> str:
    key = str(defense_type).lower()
    alias = {
        "adversarial_training": "at_transform",
        "adversarial_training_transform": "at_transform",
        "certified_randomized_smoothing": "certified_defense",
        "random_pad_and_crop": "random_pad_crop",
    }
    return alias.get(key, key)


def _normalize_defenses(defenses_cfg: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if isinstance(defenses_cfg, dict):
        for name, raw in defenses_cfg.items():
            cfg = dict(raw or {})
            if not bool(cfg.get("enabled", True)):
                continue
            cfg.pop("enabled", None)
            d_type = _normalize_defense_type(cfg.get("type", name))
            entry: Dict[str, Any] = {
                "name": str(name),
                "type": str(d_type),
            }
            for key in ("domain", "dataset", "params", "claimed_robust_accuracy"):
                if key in cfg:
                    entry[key] = cfg[key]

            model_cfg = cfg.get("model")
            if isinstance(model_cfg, dict):
                entry["model"] = dict(model_cfg)
            else:
                m: Dict[str, Any] = {}
                if "model_name" in cfg:
                    m["model_name"] = cfg["model_name"]
                if "checkpoint_path" in cfg:
                    m["checkpoint_path"] = cfg["checkpoint_path"]
                if "dataset" in cfg:
                    m["dataset"] = cfg["dataset"]
                if "domain" in cfg:
                    m["domain"] = cfg["domain"]
                if m:
                    entry["model"] = m

            # Preserve any additional defense kwargs (common in evaluate-style YAMLs),
            # by merging them into params so ``evaluate_cmd`` can flatten them.
            params = dict(entry.get("params") or {})
            reserved = {
                "name",
                "type",
                "domain",
                "dataset",
                "model",
                "model_path",
                "model_name",
                "model_factory",
                "checkpoint_path",
                "weights_path",
                "architecture",
                "params",
                "claimed_robust_accuracy",
            }
            for k, v in cfg.items():
                if k in reserved:
                    continue
                params.setdefault(k, v)
            if params:
                entry["params"] = params

            entries.append(entry)
        return entries

    if isinstance(defenses_cfg, list):
        for item in defenses_cfg:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("enabled", True)):
                continue
            defense_block = item.get("defense", {})
            if not isinstance(defense_block, dict):
                defense_block = {}

            d_type = _normalize_defense_type(
                defense_block.get("type") or item.get("type") or item.get("id", "defense")
            )
            entry: Dict[str, Any] = {
                "name": str(item.get("id") or item.get("name") or d_type),
                "type": str(d_type),
                "domain": item.get("domain", "unknown"),
                "dataset": item.get("dataset"),
                "claimed_robust_accuracy": float(item.get("claimed_robust_accuracy", 0.0)),
            }
            params: Dict[str, Any] = {}
            params.update(dict(item.get("params", {}) or {}))
            params.update(dict(defense_block.get("params", {}) or {}))
            reserved = {
                "id",
                "name",
                "type",
                "defense",
                "domain",
                "dataset",
                "model",
                "params",
                "claimed_robust_accuracy",
                "enabled",
            }
            for k, v in item.items():
                if k in reserved:
                    continue
                params.setdefault(k, v)
            if params:
                entry["params"] = params
            if "model" in item:
                entry["model"] = item["model"]
            entries.append(entry)
        return entries

    raise ValueError("defenses must be a mapping or list.")


def _validate_strict_real_data(
    config: Dict[str, Any],
    *,
    allow_missing: bool,
    base_dir: Path,
) -> None:
    def _resolve_existing_path(raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            return p
        cwd_candidate = Path.cwd() / p
        if cwd_candidate.exists():
            return cwd_candidate
        return base_dir / p

    dataset_cfg = dict(config.get("datasets", {}) or {})
    defenses = list(config.get("defenses", []) or [])
    used_datasets: Set[str] = set()

    for defense in defenses:
        ds = str(defense.get("dataset", "")).lower()
        if not ds:
            raise click.ClickException(
                f"Defense {defense.get('name', defense.get('type'))} missing dataset assignment."
            )
        used_datasets.add(ds)
        if ds not in REAL_DATASETS:
            raise click.ClickException(
                f"Strict real-data mode forbids non-real dataset '{ds}'."
            )

    for ds in sorted(used_datasets):
        cfg = dict(dataset_cfg.get(ds, {}) or {})
        root = cfg.get("path") or cfg.get("root")
        if root is None:
            raise click.ClickException(
                f"Strict real-data mode requires datasets.{ds}.root/path to be set."
            )
        # Fail early with actionable errors if optional dataset dependencies are missing.
        if ds == "ember":
            try:
                import ember  # noqa: F401
            except Exception as exc:
                raise click.ClickException(
                    "Strict real-data mode requires the 'ember' Python package. "
                    "Install with: python -m pip install git+https://github.com/elastic/ember.git"
                ) from exc
        if ds == "nuscenes":
            try:
                import nuscenes  # noqa: F401
            except Exception as exc:
                raise click.ClickException(
                    "Strict real-data mode requires the 'nuscenes-devkit' Python package. "
                    "Install with: python -m pip install nuscenes-devkit"
                ) from exc
        root_path = _resolve_existing_path(str(root))
        if not root_path.exists() and not allow_missing:
            raise click.ClickException(f"Dataset root missing for {ds}: {root_path}")

        if ds == "nuscenes":
            labels = cfg.get("labels_path")
            if not labels:
                raise click.ClickException("Strict real-data mode requires nuscenes labels_path.")
            labels_path = _resolve_existing_path(str(labels))
            if not labels_path.exists() and not allow_missing:
                raise click.ClickException(f"nuScenes labels_path missing: {labels_path}")

    # Stub/placeholder model hard-fail:
    # In strict real-data mode we refuse obvious placeholders deterministically
    # (rather than waiting for clean-accuracy symptoms on small smoke runs).
    models_cfg = config.get("models") or {}

    def _resolve_existing_model_path(ref: Any) -> Path | None:
        if isinstance(ref, (str, Path)):
            p = _resolve_existing_path(str(ref))
            return p if p.exists() else None
        if isinstance(ref, dict):
            path_val = ref.get("path") or ref.get("checkpoint_path") or ref.get("weights_path")
            if path_val:
                p = _resolve_existing_path(str(path_val))
                return p if p.exists() else None
        return None

    def _is_obvious_placeholder_model(path: Path, *, dataset: str) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        name = path.name.lower()
        if any(marker in name for marker in ("stub", "placeholder", "dummy")):
            reasons.append("filename_contains_stub_marker")

        meta = load_model_meta(path)
        if isinstance(meta, dict):
            if bool(meta.get("is_stub", False)) or bool(meta.get("stub", False)) or bool(meta.get("placeholder", False)):
                reasons.append("meta_marks_stub")

        if str(dataset).lower() == "nuscenes":
            # nuScenes evaluation depends on the label-map semantics; require the
            # training-side metadata that pins `labels_sha256`.
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            if not meta_path.exists():
                reasons.append("missing_meta_json_for_nuscenes")
            elif not meta or not meta.get("labels_sha256"):
                reasons.append("missing_labels_sha256_in_meta")

        return (len(reasons) > 0), reasons

    # Validate per-defense model refs (if present)
    for defense in defenses:
        ds = str(defense.get("dataset", "")).lower()
        if not ds:
            continue
        model_ref = None
        origin = None
        if "model" in defense:
            model_ref = defense.get("model")
            origin = f"defenses[{defense.get('name', defense.get('type'))}].model"
        elif "model_path" in defense:
            model_ref = defense.get("model_path")
            origin = f"defenses[{defense.get('name', defense.get('type'))}].model_path"
        if model_ref is not None:
            model_path = _resolve_existing_model_path(model_ref)
            if model_path is not None:
                is_stub, reasons = _is_obvious_placeholder_model(model_path, dataset=ds)
                if is_stub:
                    raise click.ClickException(
                        "Strict real-data mode refused placeholder model: "
                        f"{origin}={model_path} reasons={','.join(reasons)}"
                    )

    # Validate per-dataset model refs (common Table2 config style)
    if isinstance(models_cfg, dict):
        for ds in sorted(used_datasets):
            if ds not in models_cfg:
                continue
            model_path = _resolve_existing_model_path(models_cfg.get(ds))
            if model_path is None:
                continue  # model may be a ModelFactory name; cannot validate here
            is_stub, reasons = _is_obvious_placeholder_model(model_path, dataset=ds)
            if is_stub:
                raise click.ClickException(
                    "Strict real-data mode refused placeholder model: "
                    f"models.{ds}={model_path} reasons={','.join(reasons)}"
                )

    budgets = get_attack_budgets(config)
    for ds in sorted(used_datasets):
        if ds not in budgets:
            raise click.ClickException(
                f"Strict real-data mode requires per-dataset budget entry for '{ds}' "
                "(attack_budgets or defaults.attack_budgets)."
            )
        budget = budgets.get(ds, {})
        if "norm" not in budget or ("eps" not in budget and "epsilon" not in budget):
            raise click.ClickException(
                f"Dataset budget for '{ds}' must include norm and eps/epsilon."
            )
