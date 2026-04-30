"""Tier 2: Table 2 job orchestrator.

This extends the legacy Table 2 evaluation harness by introducing a top-level
`jobs:` list in the YAML config. Each job runs in its own output directory:

  {output-dir}/{job_id}/run_metadata.json
  {output-dir}/{job_id}/result.json

Additional artifacts may be written under the job directory (plots, tables,
raw per-defense evaluation outputs, etc.).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import yaml

from ..baselines.frameworks_compare import run_framework_head_to_head
from ..characterization.defense_analyzer import DefenseAnalyzer
from ..statistical.roc_calibration import calibrate_threshold
from .evaluate_cmd import run_evaluation
from .metadata import write_run_metadata
from .utils import (
    build_defense,
    load_dataset,
    load_model,
    load_threshold_overrides,
    resolve_device,
    save_json,
    set_seed,
)


SUPPORTED_JOB_KINDS = {
    "core_evasion",
    "baseline_framework",
    "detection_calibration",
    "stateful_defense_eval",
    "volterra_confound",
    "module_ablation",
    # Tier 3: query-limited / black-box evidence (Square Attack sweeps).
    "query_sweep",
}


def _sanitize_job_id(raw: str) -> str:
    s = str(raw).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._-") or "job"


def _job_id(job: Dict[str, Any], idx: int) -> str:
    for key in ("id", "job_id", "name"):
        if job.get(key):
            return _sanitize_job_id(str(job[key]))
    kind = _sanitize_job_id(str(job.get("kind", "job")))
    ds = job.get("dataset")
    defense = job.get("defense")
    parts = [f"{idx:03d}", kind]
    if ds:
        parts.append(_sanitize_job_id(str(ds)))
    if defense:
        parts.append(_sanitize_job_id(str(defense)))
    return "_".join([p for p in parts if p])


def _normalize_defense_type(defense_type: str) -> str:
    key = str(defense_type).lower()
    alias = {
        "adversarial_training": "at_transform",
        "adversarial_training_transform": "at_transform",
        "certified_randomized_smoothing": "certified_defense",
        "random_pad_and_crop": "random_pad_crop",
    }
    return alias.get(key, key)


def _resolve_config_path(base_dir: Path, raw_path: str | Path) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / p).resolve()


def _write_job_config(job_dir: Path, job: Dict[str, Any]) -> Path:
    job_cfg_path = job_dir / "job_config.yaml"
    with job_cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(job, handle, sort_keys=False)
    return job_cfg_path


def _resolve_dataset_cfg(raw: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    datasets = raw.get("datasets") or {}
    if not isinstance(datasets, dict):
        raise click.ClickException("Top-level 'datasets' must be a mapping when using jobs.")
    ds_cfg = datasets.get(dataset_name)
    if not isinstance(ds_cfg, dict):
        raise click.ClickException(f"Missing datasets[{dataset_name!r}] in Table2 jobs config.")
    out = dict(ds_cfg)
    # Accept either `root:` or `path:` like other commands.
    if "root" in out and "path" not in out:
        out["path"] = out["root"]
    return out


def _resolve_model_ref(raw: Dict[str, Any], dataset_name: str, job: Dict[str, Any]) -> str:
    # Job-level override
    if job.get("model"):
        return str(job["model"])
    if job.get("model_path"):
        return str(job["model_path"])

    models = raw.get("models") or {}
    if isinstance(models, dict) and models.get(dataset_name):
        return str(models[dataset_name])

    # Legacy fallback: allow a single top-level model reference.
    if raw.get("model"):
        return str(raw["model"])
    if raw.get("model_path"):
        return str(raw["model_path"])

    raise click.ClickException(
        f"Missing model reference for dataset={dataset_name!r}. "
        "Provide jobs[].model or top-level models: mapping."
    )


def _resolve_attack_entries(raw: Dict[str, Any], job: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Job can override attacks; otherwise fall back to top-level attacks.
    attacks_cfg = job.get("attacks", raw.get("attacks", []))
    entries: List[Dict[str, Any]] = []
    if isinstance(attacks_cfg, list):
        for item in attacks_cfg:
            if isinstance(item, str):
                entries.append({"name": item})
            elif isinstance(item, dict):
                entries.append(dict(item))
            else:
                raise click.ClickException("attacks entries must be strings or mappings.")
    elif isinstance(attacks_cfg, dict):
        for name, cfg in attacks_cfg.items():
            d = dict(cfg or {})
            d["name"] = str(name)
            entries.append(d)
    else:
        raise click.ClickException("attacks must be a list or mapping.")
    return entries


def _resolve_budget(
    *,
    raw: Dict[str, Any],
    dataset: str,
    job: Dict[str, Any],
    strict_dataset_budgets: bool,
) -> Tuple[float, str]:
    if job.get("epsilon") is not None:
        epsilon = float(job["epsilon"])
    else:
        # Fall back to attack_budgets[dataset].epsilon if present.
        budgets = raw.get("attack_budgets") or {}
        b = budgets.get(dataset, {}) if isinstance(budgets, dict) else {}
        if strict_dataset_budgets and "epsilon" not in (b or {}):
            raise click.ClickException(
                f"Missing epsilon for dataset={dataset!r}. Provide jobs[].epsilon or attack_budgets[{dataset!r}].epsilon."
            )
        epsilon = float((b or {}).get("epsilon", 0.03))

    if job.get("norm") is not None:
        norm = str(job["norm"])
    else:
        budgets = raw.get("attack_budgets") or {}
        b = budgets.get(dataset, {}) if isinstance(budgets, dict) else {}
        if strict_dataset_budgets and "norm" not in (b or {}):
            raise click.ClickException(
                f"Missing norm for dataset={dataset!r}. Provide jobs[].norm or attack_budgets[{dataset!r}].norm."
            )
        norm = str((b or {}).get("norm", "Linf"))

    return epsilon, norm


def _run_job_core_evasion(
    ctx: click.Context,
    *,
    raw: Dict[str, Any],
    job: Dict[str, Any],
    job_id: str,
    job_dir: Path,
    device: str,
    strict_dataset_budgets: bool,
    common_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    dataset = str(job.get("dataset") or "")
    if not dataset:
        raise click.ClickException(f"{job_id}: core_evasion job requires 'dataset'.")
    defense = str(job.get("defense") or "")
    if not defense:
        raise click.ClickException(f"{job_id}: core_evasion job requires 'defense'.")

    ds_cfg = _resolve_dataset_cfg(raw, dataset)
    model_ref = _resolve_model_ref(raw, dataset, job)
    epsilon, norm = _resolve_budget(raw=raw, dataset=dataset, job=job, strict_dataset_budgets=strict_dataset_budgets)

    # Allow job to override sample counts without mutating the global config.
    if job.get("num_samples") is not None:
        ds_cfg["num_samples"] = int(job["num_samples"])
    if job.get("batch_size") is not None:
        ds_cfg["batch_size"] = int(job["batch_size"])
    if job.get("split") is not None:
        ds_cfg["split"] = str(job["split"])

    seed = int(job.get("seed", raw.get("seed", 42)))
    iterations = int(job.get("iterations", raw.get("iterations", raw.get("attack_iterations", 100))))

    # Defense params can be provided explicitly via `defense_params`/`params`.
    defense_params = dict(job.get("defense_params") or job.get("params") or {})

    attacks = _resolve_attack_entries(raw, job)

    eval_cfg: Dict[str, Any] = {
        "seed": seed,
        "strict_dataset_budgets": bool(strict_dataset_budgets),
        "datasets": {dataset: ds_cfg},
        "models": {dataset: model_ref},
        "defenses": [
            {
                "name": defense,
                "type": _normalize_defense_type(defense),
                "dataset": dataset,
                "params": defense_params,
            }
        ],
        "attacks": attacks,
        "perturbation": {"epsilon": float(epsilon), "norm": str(norm)},
        "iterations": iterations,
    }

    # Preserve optional gates/settings from the parent config.
    for key in (
        "attack_budgets",
        "baseline_validation",
        "validity_gates",
        "cache",
        "characterization_thresholds",
        "characterization_thresholds_path",
        "characterization_thresholds_source",
        "threshold_overrides",
        "thresholds",
    ):
        if key in raw:
            eval_cfg[key] = raw.get(key)

    eval_cfg_path = job_dir / "eval_config.yaml"
    with eval_cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(eval_cfg, handle, sort_keys=False)

    # Run evaluation, but let Table2-jobs own the run metadata.
    run_evaluation(
        ctx,
        config=str(eval_cfg_path),
        output_dir=str(job_dir),
        _skip_run_metadata=True,
        **common_kwargs,
    )

    # Evaluation writes summary.json + per-defense JSONs. We provide a canonical `result.json`.
    summary_path = job_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        summary = {}

    result = {
        "kind": "core_evasion",
        "job_id": job_id,
        "dataset": dataset,
        "defense": defense,
        "epsilon": float(epsilon),
        "norm": str(norm),
        "seed": seed,
        "iterations": iterations,
        "outputs": {
            "eval_config": str(eval_cfg_path),
            "summary_json": str(summary_path),
        },
        "evaluation_summary": summary,
    }
    return result


def _run_job_query_sweep(
    ctx: click.Context,
    *,
    raw: Dict[str, Any],
    job: Dict[str, Any],
    job_id: str,
    job_dir: Path,
    device: str,
    strict_dataset_budgets: bool,
    common_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Tier 3: run a query-budget sweep (typically Square Attack) and emit
    per-budget summaries under subdirectories.

    This is intended to produce defensible evidence for "query-limited fallback"
    statements (ASR vs query budget curves + actual query accounting).
    """
    dataset = str(job.get("dataset") or "")
    if not dataset:
        raise click.ClickException(f"{job_id}: query_sweep job requires 'dataset'.")
    defense = str(job.get("defense") or "none")

    ds_cfg = _resolve_dataset_cfg(raw, dataset)
    model_ref = _resolve_model_ref(raw, dataset, job)
    epsilon, norm = _resolve_budget(raw=raw, dataset=dataset, job=job, strict_dataset_budgets=strict_dataset_budgets)

    seed = int(job.get("seed", raw.get("seed", 42)))
    set_seed(seed)

    defense_params = dict(job.get("defense_params") or job.get("params") or {})

    num_samples = int(job.get("num_samples", ds_cfg.get("num_samples", 200)))
    batch_size = int(job.get("batch_size", ds_cfg.get("batch_size", 128)))
    split = str(job.get("split", ds_cfg.get("split", "test")))
    num_workers = int(ds_cfg.get("num_workers", 0))

    # Query budgets to sweep (n_queries for Square Attack).
    query_budgets = job.get("query_budgets", job.get("budgets"))
    if not isinstance(query_budgets, list) or not query_budgets:
        raise click.ClickException(f"{job_id}: query_sweep job requires non-empty 'query_budgets: [...]'.")
    budgets: List[int] = []
    for q in query_budgets:
        try:
            budgets.append(int(q))
        except Exception:
            raise click.ClickException(f"{job_id}: query_budgets must be integers (got {q!r}).")
    budgets = sorted(set([q for q in budgets if q > 0]))
    if not budgets:
        raise click.ClickException(f"{job_id}: query_budgets must contain at least one positive integer.")

    # Attack parameters for Square Attack.
    attack_cfg = dict(job.get("attack") or {})
    p_init = float(attack_cfg.get("p_init", 0.8))
    loss_type = str(attack_cfg.get("loss_type", "margin"))
    attack_type = str(attack_cfg.get("type", "square"))
    if attack_type.lower() not in {"square"}:
        raise click.ClickException(f"{job_id}: query_sweep currently supports attack.type='square' only.")

    runs: List[Dict[str, Any]] = []
    outputs: Dict[str, Any] = {}

    for q in budgets:
        # Keep subdirectories stable for easy plotting/aggregation.
        sub_id = f"q{q}"
        sub_dir = job_dir / sub_id
        sub_dir.mkdir(parents=True, exist_ok=True)

        eval_cfg: Dict[str, Any] = {
            "seed": seed,
            "strict_dataset_budgets": bool(strict_dataset_budgets),
            "datasets": {
                dataset: {
                    **dict(ds_cfg),
                    "num_samples": int(num_samples),
                    "batch_size": int(batch_size),
                    "num_workers": int(num_workers),
                    "split": split,
                }
            },
            "models": {dataset: model_ref},
            "defenses": [
                {
                    "name": defense,
                    "type": _normalize_defense_type(defense),
                    "dataset": dataset,
                    "params": dict(defense_params),
                }
            ],
            "attacks": [
                {
                    "name": f"square_{sub_id}",
                    "type": "square",
                    "n_queries": int(q),
                    "p_init": float(p_init),
                    "loss_type": str(loss_type),
                }
            ],
            "perturbation": {"epsilon": float(epsilon), "norm": str(norm)},
            # iterations is unused by Square Attack, but keep schema consistent.
            "iterations": int(job.get("iterations", raw.get("iterations", 100))),
        }

        # Preserve optional gates/settings from the parent config.
        for key in (
            "attack_budgets",
            "baseline_validation",
            "validity_gates",
            "cache",
            "characterization_thresholds",
            "characterization_thresholds_path",
            "characterization_thresholds_source",
        ):
            if key in raw:
                eval_cfg[key] = raw.get(key)

        eval_cfg_path = sub_dir / "eval_config.yaml"
        with eval_cfg_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(eval_cfg, handle, sort_keys=False)

        run_evaluation(
            ctx,
            config=str(eval_cfg_path),
            output_dir=str(sub_dir),
            _skip_run_metadata=True,
            **common_kwargs,
        )

        summary_path = sub_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

        runs.append(
            {
                "query_budget": int(q),
                "outputs": {"eval_config": str(eval_cfg_path), "summary_json": str(summary_path)},
                "evaluation_summary": summary,
            }
        )
        outputs[sub_id] = {"eval_config": str(eval_cfg_path), "summary_json": str(summary_path)}

    return {
        "kind": "query_sweep",
        "job_id": job_id,
        "dataset": dataset,
        "defense": defense,
        "epsilon": float(epsilon),
        "norm": str(norm),
        "seed": seed,
        "num_samples": int(num_samples),
        "attack": {"type": "square", "p_init": float(p_init), "loss_type": str(loss_type)},
        "query_budgets": budgets,
        "outputs": outputs,
        "runs": runs,
    }


def _run_job_baseline_framework(
    ctx: click.Context,
    *,
    raw: Dict[str, Any],
    job: Dict[str, Any],
    job_id: str,
    job_dir: Path,
    device: str,
    strict_dataset_budgets: bool,
) -> Dict[str, Any]:
    dataset = str(job.get("dataset") or "")
    if not dataset:
        raise click.ClickException(f"{job_id}: baseline_framework job requires 'dataset'.")
    defense = str(job.get("defense") or "none")

    ds_cfg = _resolve_dataset_cfg(raw, dataset)
    model_ref = _resolve_model_ref(raw, dataset, job)
    epsilon, norm = _resolve_budget(raw=raw, dataset=dataset, job=job, strict_dataset_budgets=strict_dataset_budgets)

    seed = int(job.get("seed", raw.get("seed", 42)))
    set_seed(seed)

    steps = int(job.get("iterations", job.get("steps", raw.get("iterations", 100))))
    restarts = int(job.get("restarts", job.get("n_restarts", 1)))
    eps_step = job.get("step_size", job.get("eps_step", None))
    eps_step_f = float(eps_step) if eps_step is not None else None

    defense_params = dict(job.get("defense_params") or job.get("params") or {})

    num_samples = int(job.get("num_samples", ds_cfg.get("num_samples", 1000)))
    batch_size = int(job.get("batch_size", ds_cfg.get("batch_size", 128)))
    num_workers = int(ds_cfg.get("num_workers", 0))
    split = str(job.get("split", ds_cfg.get("split", "test")))

    # Resolve which frameworks to run.
    requested = job.get("frameworks") or ["neurinspectre", "art", "foolbox"]
    if isinstance(requested, list):
        names = []
        for item in requested:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
            else:
                raise click.ClickException(f"{job_id}: frameworks entries must be strings or mappings with name.")
    else:
        raise click.ClickException(f"{job_id}: frameworks must be a list.")

    fw_map = {
        "neurinspectre": "neurinspectre",
        "art": "art_apgd",
        "art_apgd": "art_apgd",
        "foolbox": "foolbox_pgd",
        "foolbox_pgd": "foolbox_pgd",
    }
    frameworks: Tuple[str, ...] = tuple([fw_map.get(str(n).lower().strip(), str(n)) for n in names])

    # Budget parity gate (strict mode).
    if bool(job.get("strict_budget_parity", strict_dataset_budgets)):
        missing = []
        if eps_step_f is None:
            missing.append("step_size/eps_step")
        if steps <= 0:
            missing.append("iterations/steps")
        if restarts <= 0:
            missing.append("restarts")
        if missing:
            raise click.ClickException(
                f"{job_id}: budget parity fields missing ({', '.join(missing)}). "
                "Provide steps, restarts, and step_size/eps_step explicitly."
            )

    # Load a single data loader (DataLoader is re-iterable).
    loader, _x, _y = load_dataset(
        dataset,
        data_path=ds_cfg.get("path") or ds_cfg.get("root"),
        labels_path=ds_cfg.get("labels_path"),
        nuscenes_version=ds_cfg.get("version") or ds_cfg.get("nuscenes_version"),
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        split=split,
        device=device,
    )

    # Each framework gets its own freshly constructed model/defense.
    def eval_model_factory(framework_key: str):
        base = load_model(model_ref, dataset=dataset, device=device)
        if defense.lower() in {"none", "clean", "identity"}:
            return base, None
        defended = build_defense(
            _normalize_defense_type(defense),
            base,
            params=dict(defense_params),
            device=device,
        )
        return (defended or base), defended

    start = time.time()
    fw_results = run_framework_head_to_head(
        eval_model_factory=eval_model_factory,
        data_loader=loader,
        eps=float(epsilon),
        norm=str(norm),
        steps=int(steps),
        restarts=int(restarts),
        eps_step=eps_step_f,
        batch_size=int(batch_size),
        num_samples=int(num_samples),
        device=str(device),
        frameworks=frameworks,
        neurinspectre_raw_config={
            # Keep the NeurInSpectre side budget-aligned.
            "n_iterations": int(steps),
            "n_restarts": int(restarts),
            "step_size": eps_step_f,
        },
    )
    elapsed = float(time.time() - start)

    # Convert into a stable JSON schema.
    results_out: List[Dict[str, Any]] = []
    for key, item in fw_results.items():
        results_out.append(
            {
                "framework": str(item.get("framework", key)),
                "attack": str(item.get("attack", "")),
                "asr": float(item.get("asr", 0.0)),
                "clean_accuracy": float(item.get("clean_accuracy", 0.0)),
                "robust_accuracy": float(item.get("robust_accuracy", 0.0)),
                "wall_s": float(item.get("elapsed_seconds", item.get("wall_s", 0.0))),
                "n_samples": int(item.get("n_samples", 0)),
                "n_attackable": int(item.get("n_attackable", 0)),
            }
        )

    return {
        "kind": "baseline_framework",
        "job_id": job_id,
        "dataset": dataset,
        "defense": defense,
        "epsilon": float(epsilon),
        "norm": str(norm),
        "num_samples": int(num_samples),
        "seed": seed,
        "budget_parity": {"steps": int(steps), "restarts": int(restarts), "eps_step": eps_step_f},
        "frameworks": list(frameworks),
        "results": results_out,
        "timing": {"total_seconds": elapsed},
    }


def _flatten_characterization_for_calibration(char_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a flat metric dict suitable for ROC calibration.

    `DefenseCharacterization.to_dict()` places some Layer-1 features under
    `metadata`; we mirror key fields at top-level for convenience.
    """
    out = dict(char_dict)
    meta = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    if isinstance(meta, dict):
        for k in ("spectral_entropy", "spectral_entropy_norm", "high_freq_ratio", "paper_style"):
            if k in meta and k not in out:
                out[k] = meta.get(k)
        # Preserve thresholds used (helps audit calibrated-vs-default runs).
        if "thresholds" in meta and "thresholds" not in out:
            out["thresholds"] = meta.get("thresholds")
    return out


def _plot_roc_curves(roc: Dict[str, Any], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#888888", linewidth=1.0)

    for metric, item in (roc.get("metrics") or {}).items():
        curve = item.get("roc_curve") or {}
        fpr = curve.get("fpr") or []
        tpr = curve.get("tpr") or []
        if not fpr or not tpr:
            continue
        auc = float(item.get("auc", float("nan")))
        ax.plot(fpr, tpr, label=f"{metric} (AUC={auc:.3f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Detection Calibration ROC")
    ax.legend(fontsize=8, loc="lower right")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _run_job_detection_calibration(
    ctx: click.Context,
    *,
    raw: Dict[str, Any],
    job: Dict[str, Any],
    job_id: str,
    job_dir: Path,
    device: str,
    strict_dataset_budgets: bool,
) -> Dict[str, Any]:
    dataset = str(job.get("dataset") or "")
    if not dataset:
        raise click.ClickException(f"{job_id}: detection_calibration job requires 'dataset'.")
    defenses = job.get("defenses")
    if not isinstance(defenses, list) or not defenses:
        raise click.ClickException(f"{job_id}: detection_calibration job requires non-empty 'defenses: [...]'.")

    ds_cfg = _resolve_dataset_cfg(raw, dataset)
    model_ref = _resolve_model_ref(raw, dataset, job)
    epsilon, _norm = _resolve_budget(raw=raw, dataset=dataset, job=job, strict_dataset_budgets=strict_dataset_budgets)

    seed = int(job.get("seed", raw.get("seed", 42)))
    repeats = int(job.get("repeats", job.get("num_repeats", 5)))
    repeats = max(1, repeats)
    num_samples = int(job.get("num_samples", ds_cfg.get("num_samples", 1000)))
    batch_size = int(job.get("batch_size", ds_cfg.get("batch_size", 128)))
    num_workers = int(ds_cfg.get("num_workers", 0))
    split = str(job.get("split", ds_cfg.get("split", "test")))

    metrics = job.get("metrics") or ["etd_score", "alpha_volterra"]
    if not isinstance(metrics, list) or not metrics:
        raise click.ClickException(f"{job_id}: metrics must be a non-empty list.")
    metrics = [str(m) for m in metrics]

    target_fpr = float(job.get("target_fpr", 0.05))

    # Collect negative ("clean") samples.
    pos_reports: List[Dict[str, Any]] = []
    neg_reports: List[Dict[str, Any]] = []

    for r in range(repeats):
        set_seed(seed + r)
        loader, _x, _y = load_dataset(
            dataset,
            data_path=ds_cfg.get("path") or ds_cfg.get("root"),
            labels_path=ds_cfg.get("labels_path"),
            nuscenes_version=ds_cfg.get("version") or ds_cfg.get("nuscenes_version"),
            num_samples=num_samples,
            batch_size=batch_size,
            seed=seed + r,
            num_workers=num_workers,
            split=split,
            device=device,
        )
        clean_model = load_model(model_ref, dataset=dataset, device=device)
        analyzer = DefenseAnalyzer(clean_model, device=device, verbose=False)
        char = analyzer.characterize(loader, eps=float(epsilon))
        rep = _flatten_characterization_for_calibration(char.to_dict())
        rep.update({"dataset": dataset, "defense": "none", "seed": int(seed + r)})
        neg_reports.append(rep)

    # Collect positive samples from the configured defenses.
    for defense_name in defenses:
        dname = str(defense_name)
        dp = job.get("defense_params") or job.get("params") or {}
        if isinstance(dp, dict) and isinstance(dp.get(dname), dict):
            defense_params = dict(dp.get(dname) or {})
        elif isinstance(dp, dict):
            defense_params = dict(dp)
        else:
            defense_params = {}
        for r in range(repeats):
            set_seed(seed + r)
            loader, _x, _y = load_dataset(
                dataset,
                data_path=ds_cfg.get("path") or ds_cfg.get("root"),
                labels_path=ds_cfg.get("labels_path"),
                nuscenes_version=ds_cfg.get("version") or ds_cfg.get("nuscenes_version"),
                num_samples=num_samples,
                batch_size=batch_size,
                seed=seed + r,
                num_workers=num_workers,
                split=split,
                device=device,
            )
            base_model = load_model(model_ref, dataset=dataset, device=device)
            defended = build_defense(
                _normalize_defense_type(dname),
                base_model,
                params=defense_params,
                device=device,
            )
            eval_model = defended or base_model
            analyzer = DefenseAnalyzer(eval_model, device=device, verbose=False)
            char = analyzer.characterize(loader, eps=float(epsilon))
            rep = _flatten_characterization_for_calibration(char.to_dict())
            rep.update({"dataset": dataset, "defense": dname, "seed": int(seed + r)})
            pos_reports.append(rep)

    # Calibrate thresholds for each metric.
    metric_out: Dict[str, Any] = {}
    overrides: Dict[str, Any] = {}

    # Mapping from user-facing metric name to DefenseAnalyzer threshold attribute.
    # Keep this in sync with `neurinspectre calibrate-thresholds`.
    metric_to_threshold_attr = {
        "etd_score": "ETD_THRESHOLD_SEVERE",
        "alpha_volterra": "ALPHA_RL_THRESHOLD",
        "spectral_entropy_norm": "SPECTRAL_ENTROPY_OBFUSCATED_THRESHOLD",
        "high_freq_ratio": "HIGH_FREQ_RATIO_SHATTERED_THRESHOLD",
        "jacobian_rank": "RANK_VANISHING_THRESHOLD",
        "gradient_variance": "VARIANCE_STOCHASTIC_THRESHOLD",
    }

    for metric in metrics:
        cal = calibrate_threshold(
            positive_samples=pos_reports,
            negative_samples=neg_reports,
            metric=metric,
            rule=str(job.get("rule", "greater_equal")),
            target_fpr=float(target_fpr),
            include_curve=True,
        )
        metric_out[metric] = cal
        thr = cal.get("threshold")
        attr = metric_to_threshold_attr.get(str(metric))
        if attr and thr is not None:
            overrides[attr] = float(thr)

    roc_json = {
        "kind": "detection_calibration",
        "job_id": job_id,
        "dataset": dataset,
        "seed": seed,
        "repeats": repeats,
        "num_samples": int(num_samples),
        "epsilon": float(epsilon),
        "target_fpr": float(target_fpr),
        "metrics": metric_out,
        "n_positive": int(len(pos_reports)),
        "n_negative": int(len(neg_reports)),
    }

    plots_dir = job_dir / "plots"
    roc_path = job_dir / "roc.json"
    save_json(roc_json, roc_path)

    roc_png = plots_dir / "roc.png"
    try:
        _plot_roc_curves(roc_json, roc_png)
    except Exception:
        # Plotting should not be a hard failure in headless environments.
        roc_png = None

    thresholds_payload = {
        "metadata": {
            "tool": "NeurInSpectre",
            "kind": "threshold_calibration",
            "dataset": dataset,
            "seed": seed,
            "repeats": repeats,
            "num_samples": int(num_samples),
            "epsilon": float(epsilon),
            "target_fpr": float(target_fpr),
        },
        "metric_calibration": metric_out,
        "defense_analyzer_threshold_overrides": overrides,
    }
    thresholds_path = job_dir / "thresholds.json"
    save_json(thresholds_payload, thresholds_path)

    outputs = {"roc_json": str(roc_path), "thresholds_json": str(thresholds_path)}
    if roc_png is not None:
        outputs["roc_png"] = str(roc_png)

    return {
        "kind": "detection_calibration",
        "job_id": job_id,
        "dataset": dataset,
        "defenses": [str(d) for d in defenses],
        "metrics": metrics,
        "target_fpr": float(target_fpr),
        "n_positive": int(len(pos_reports)),
        "n_negative": int(len(neg_reports)),
        "defense_analyzer_threshold_overrides": overrides,
        "outputs": outputs,
    }


def _run_job_stateful_defense_eval(
    ctx: click.Context,
    *,
    raw: Dict[str, Any],
    job: Dict[str, Any],
    job_id: str,
    job_dir: Path,
    device: str,
    strict_dataset_budgets: bool,
    common_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    dataset = str(job.get("dataset") or "")
    if not dataset:
        raise click.ClickException(f"{job_id}: stateful_defense_eval job requires 'dataset'.")
    defense = str(job.get("defense") or "tent")

    # We implement this as a specialized core evaluation with two NeurInSpectre variants.
    job2 = dict(job)
    job2["defense"] = defense
    job2["attacks"] = [
        {"name": "neurinspectre_no_mem", "type": "neurinspectre", "volterra_mode": "off"},
        {"name": "neurinspectre_mem", "type": "neurinspectre", "volterra_mode": "on"},
    ]
    res = _run_job_core_evasion(
        ctx,
        raw=raw,
        job=job2,
        job_id=job_id,
        job_dir=job_dir,
        device=device,
        strict_dataset_budgets=strict_dataset_budgets,
        common_kwargs=common_kwargs,
    )
    res["kind"] = "stateful_defense_eval"
    return res


def _run_job_volterra_confound(
    *,
    raw: Dict[str, Any],
    job: Dict[str, Any],
    job_id: str,
    job_dir: Path,
    device: str,
    strict_dataset_budgets: bool,
) -> Dict[str, Any]:
    dataset = str(job.get("dataset") or "")
    if not dataset:
        raise click.ClickException(f"{job_id}: volterra_confound job requires 'dataset'.")
    defense = str(job.get("defense") or "")
    if not defense:
        raise click.ClickException(f"{job_id}: volterra_confound job requires 'defense'.")

    ds_cfg = _resolve_dataset_cfg(raw, dataset)
    model_ref = _resolve_model_ref(raw, dataset, job)
    epsilon, _norm = _resolve_budget(raw=raw, dataset=dataset, job=job, strict_dataset_budgets=strict_dataset_budgets)

    seed = int(job.get("seed", raw.get("seed", 42)))
    num_samples = int(job.get("num_samples", ds_cfg.get("num_samples", 1000)))
    batch_size = int(job.get("batch_size", ds_cfg.get("batch_size", 128)))
    num_workers = int(ds_cfg.get("num_workers", 0))
    split = str(job.get("split", ds_cfg.get("split", "test")))

    loader, _x, _y = load_dataset(
        dataset,
        data_path=ds_cfg.get("path") or ds_cfg.get("root"),
        labels_path=ds_cfg.get("labels_path"),
        nuscenes_version=ds_cfg.get("version") or ds_cfg.get("nuscenes_version"),
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        split=split,
        device=device,
    )
    base_model = load_model(model_ref, dataset=dataset, device=device)
    dp = job.get("defense_params") or job.get("params") or {}
    if isinstance(dp, dict):
        defense_params = dict(dp)
    else:
        defense_params = {}
    defended = build_defense(_normalize_defense_type(defense), base_model, params=defense_params, device=device)
    eval_model = defended or base_model

    # Collect one gradient history (raw ∇x L) via the analyzer helper.
    analyzer_raw = DefenseAnalyzer(eval_model, device=device, verbose=False, volterra_gradient_source="pre_optimizer")
    grads, _images, _labels = analyzer_raw.collect_gradient_samples(loader, eps=float(epsilon))
    if not grads:
        raise click.ClickException(f"{job_id}: failed to collect gradients for Volterra confound.")

    # Fit alpha under different sources / optimizer post-processing.
    optimizers = job.get("optimizers") or ["sgd", "sgd_momentum", "adam", "rmsprop"]
    if not isinstance(optimizers, list) or not optimizers:
        raise click.ClickException(f"{job_id}: optimizers must be a non-empty list.")

    # Pre-optimizer alpha (raw gradients): baseline.
    alpha_pre, rmse_pre, rmse_scaled_pre, info_pre = analyzer_raw._fit_volterra_kernel(grads)

    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "gradient_source": "pre_optimizer",
            "optimizer": None,
            "alpha": float(alpha_pre),
            "rmse": float(rmse_pre) if np.isfinite(rmse_pre) else None,
            "rmse_scaled": float(rmse_scaled_pre) if np.isfinite(rmse_scaled_pre) else None,
            "info": info_pre,
        }
    )

    for opt in optimizers:
        analyzer_post = DefenseAnalyzer(
            eval_model,
            device=device,
            verbose=False,
            volterra_gradient_source="post_optimizer",
            volterra_optimizer=str(opt),
        )
        alpha, rmse, rmse_scaled, info = analyzer_post._fit_volterra_kernel(grads)
        rows.append(
            {
                "gradient_source": "post_optimizer",
                "optimizer": str(opt),
                "alpha": float(alpha),
                "rmse": float(rmse) if np.isfinite(rmse) else None,
                "rmse_scaled": float(rmse_scaled) if np.isfinite(rmse_scaled) else None,
                "info": info,
            }
        )

    result = {
        "kind": "volterra_confound",
        "job_id": job_id,
        "dataset": dataset,
        "defense": defense,
        "epsilon": float(epsilon),
        "seed": seed,
        "num_samples": int(num_samples),
        "optimizers": [str(o) for o in optimizers],
        "results": rows,
    }
    return result


def _run_job_module_ablation(
    *,
    job: Dict[str, Any],
    job_id: str,
    job_dir: Path,
    base_dir: Path,
) -> Dict[str, Any]:
    """
    Tier 2: run ablations for non-core modules (EDNN, attention-security).
    """
    module = str(job.get("module") or job.get("target") or "").strip().lower().replace("-", "_")
    if not module:
        raise click.ClickException(f"{job_id}: module_ablation job requires 'module' (ednn|attention_security).")

    if module in {"ednn"}:
        from types import SimpleNamespace

        from .adversarial_ednn import run_ednn

        attack_type = str(job.get("attack_type") or "inversion")
        data_path = job.get("data")
        if not data_path:
            raise click.ClickException(f"{job_id}: ednn ablation requires jobs[].data pointing to embeddings (.npy/.npz).")
        data_path = _resolve_config_path(base_dir, str(data_path))

        dim_modes = job.get("dim_selection_modes") or job.get("dim_selection") or ["all", "spectral", "random"]
        if isinstance(dim_modes, str):
            dim_modes = [dim_modes]
        if not isinstance(dim_modes, list) or not dim_modes:
            raise click.ClickException(f"{job_id}: dim_selection_modes must be a non-empty list.")

        runs: List[Dict[str, Any]] = []
        for mode in [str(m) for m in dim_modes]:
            out_subdir = job_dir / f"dim_{_sanitize_job_id(mode)}"
            args = SimpleNamespace(
                attack_type=attack_type,
                embedding_dim=int(job.get("embedding_dim", 768)),
                data=str(data_path),
                model=job.get("model"),
                device=str(job.get("device", "auto")),
                reference_embeddings=(
                    str(_resolve_config_path(base_dir, str(job.get("reference_embeddings"))))
                    if job.get("reference_embeddings")
                    else None
                ),
                reference_texts=(
                    str(_resolve_config_path(base_dir, str(job.get("reference_texts"))))
                    if job.get("reference_texts")
                    else None
                ),
                target_query=job.get("target_query"),
                poisoned_document=job.get("poisoned_document"),
                poisoned_document_file=(
                    str(_resolve_config_path(base_dir, str(job.get("poisoned_document_file"))))
                    if job.get("poisoned_document_file")
                    else None
                ),
                target_tokens=(
                    str(_resolve_config_path(base_dir, str(job.get("target_tokens"))))
                    if job.get("target_tokens")
                    else None
                ),
                output_dir=str(out_subdir),
                verbose=bool(job.get("verbose", False)),
                dim_selection=str(mode),
                dim_top_frac=float(job.get("dim_top_frac", 0.25)),
                dim_top_k=job.get("dim_top_k"),
                seed=int(job.get("seed", 0)),
            )
            rc = int(run_ednn(args))
            result_path = out_subdir / f"ednn_{attack_type}_result.json"
            payload = None
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception:
                    payload = None
            runs.append(
                {
                    "mode": str(mode),
                    "exit_code": rc,
                    "result_json": str(result_path),
                    "parsed": payload,
                }
            )

        return {
            "kind": "module_ablation",
            "job_id": job_id,
            "module": "ednn",
            "attack_type": attack_type,
            "runs": runs,
        }

    if module in {"attention_security", "attention"}:
        from .attention_security_analysis import AttentionSecurityConfig, generate_attention_security_analysis

        model = job.get("model")
        if not model:
            raise click.ClickException(f"{job_id}: attention_security ablation requires jobs[].model (HF model id).")

        prompt = job.get("prompt")
        prompt_file = job.get("prompt_file")
        if not prompt and prompt_file:
            p = _resolve_config_path(base_dir, str(prompt_file))
            prompt = p.read_text(encoding="utf-8", errors="ignore")
        if not prompt:
            raise click.ClickException(f"{job_id}: attention_security ablation requires jobs[].prompt or prompt_file.")

        feature_sets = job.get("feature_sets") or job.get("features") or ["all", "entropy_only", "entropy_inj", "spectral_only"]
        if isinstance(feature_sets, str):
            feature_sets = [feature_sets]
        if not isinstance(feature_sets, list) or not feature_sets:
            raise click.ClickException(f"{job_id}: feature_sets/features must be a non-empty list.")

        runs2: List[Dict[str, Any]] = []
        for fs in [str(x) for x in feature_sets]:
            out_subdir = job_dir / f"features_{_sanitize_job_id(fs)}"
            out_png = out_subdir / "attention_security.png"
            out_json = out_subdir / "attention_security.json"
            out_html = out_subdir / "attention_security.html"
            cfg = AttentionSecurityConfig(
                model_name=str(model),
                prompt=str(prompt),
                layer=str(job.get("layer", "all")),
                layer_start=job.get("layer_start"),
                layer_end=job.get("layer_end"),
                max_tokens=int(job.get("max_tokens", 128)),
                device=str(job.get("device", "auto")),
                output_png=str(out_png),
                out_json=str(out_json),
                out_html=str(out_html),
                contamination=str(job.get("contamination", "auto")),
                n_estimators=int(job.get("n_estimators", 256)),
                seed=int(job.get("seed", 0)),
                features=str(fs),
                title=str(job.get("title", "NeurInSpectre — Attention Security Analysis")),
            )
            out_png_s, out_json_s, out_html_s = generate_attention_security_analysis(cfg)
            parsed = None
            try:
                parsed = json.loads(Path(out_json_s).read_text(encoding="utf-8"))
            except Exception:
                parsed = None
            runs2.append(
                {
                    "features": fs,
                    "outputs": {"png": out_png_s, "json": out_json_s, "html": out_html_s},
                    "parsed": parsed,
                }
            )

        return {
            "kind": "module_ablation",
            "job_id": job_id,
            "module": "attention_security",
            "model": str(model),
            "runs": runs2,
        }

    raise click.ClickException(f"{job_id}: unknown module_ablation module={module!r}. Expected ednn|attention_security.")


def run_table2_jobs(ctx: click.Context, **kwargs: Any) -> None:
    config_path = str(kwargs.get("config"))
    raw = kwargs.get("_raw_config")
    if raw is None:
        raise click.ClickException("Internal error: Table2 jobs runner missing raw config.")
    if not isinstance(raw, dict):
        raise click.ClickException("Table2 jobs config must be a mapping at the top level.")

    jobs = raw.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise click.ClickException("Table2 jobs config must contain a non-empty top-level 'jobs:' list.")

    output_dir = Path(str(kwargs.get("output_dir", "results/table2")))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(kwargs.get("device", "auto"))
    strict_real_data = bool(kwargs.get("strict_real_data", True))
    strict_dataset_budgets = bool(kwargs.get("strict_dataset_budgets", True))
    allow_missing = bool(kwargs.get("allow_missing", False))

    base_dir = Path(config_path).resolve().parent

    # Root run metadata (for the whole orchestrator run).
    try:
        write_run_metadata(
            output_dir,
            config_path=config_path,
            device=str(device),
            extra={"command": "table2", "mode": "jobs"},
        )
    except Exception:
        pass

    # Optional CLI-provided calibrated thresholds (applies to core_evasion-style jobs).
    thresholds_path = kwargs.get("thresholds")
    characterization_thresholds: Dict[str, Any] | None = None
    if thresholds_path:
        p = _resolve_config_path(base_dir, str(thresholds_path))
        characterization_thresholds = load_threshold_overrides(str(p))

    # Common kwargs forwarded to evaluate runner.
    common_eval_kwargs = {
        "verbose": kwargs.get("verbose", 0),
        "json_output": None,  # job outputs are always written under job_dir
        "sarif_output": None,
        "report_format": kwargs.get("report_format", "rich"),
        "report": kwargs.get("report", True),
        "brief": kwargs.get("brief", False),
        "summary_only": kwargs.get("summary_only", False),
        "color": kwargs.get("color", False),
        "no_color": kwargs.get("no_color", False),
        "no_progress": kwargs.get("no_progress", False),
        "defenses": tuple(kwargs.get("defenses") or ()),
        "attacks": tuple(kwargs.get("attacks") or ()),
        "parallel": int(kwargs.get("parallel", 1)),
        "resume": bool(kwargs.get("resume", False)),
        "device": str(device),
        "seeds": tuple(kwargs.get("seeds") or ()),
        "num_seeds": int(kwargs.get("num_seeds", 1)),
    }

    # Strict real-data validation (best-effort in jobs mode).
    if strict_real_data:
        # Scientific validity gate: in strict real-data mode, fail fast if a model
        # has ~0 clean accuracy (ASR becomes non-informative / undefined).
        gates = dict(raw.get("validity_gates", {}) or {})
        gates.setdefault("enabled", True)
        gates.setdefault("strict", True)
        gates.setdefault("min_clean_accuracy", 0.05)
        gates.setdefault("min_correct_samples", 5)
        gates.setdefault("min_correct_fraction", 0.10)
        # Chance-aware gate: require clean accuracy meaningfully above uniform chance (1/K).
        gates.setdefault("min_clean_accuracy_over_chance", 0.05)
        # Artifact integrity: for nuScenes, enforce that label_map.json matches model meta.
        gates.setdefault("require_label_map_sha256_match", True)
        raw["validity_gates"] = gates

        # Only validate datasets used by jobs that declare `dataset`.
        used = {str(j.get("dataset")).lower() for j in jobs if isinstance(j, dict) and j.get("dataset")}
        real_allowed = {"cifar10", "imagenet100", "ember", "nuscenes"}
        unknown = sorted([d for d in used if d and d not in real_allowed])
        if unknown:
            raise click.ClickException(
                "Strict real-data mode: unsupported dataset(s) in jobs config: "
                + ", ".join(unknown)
            )
        # Validate dataset roots exist (unless allow_missing).
        for ds in sorted([d for d in used if d]):
            ds_cfg = _resolve_dataset_cfg(raw, ds)
            root = ds_cfg.get("path") or ds_cfg.get("root")
            if not root:
                raise click.ClickException(f"Strict real-data mode: datasets[{ds!r}] must define path/root.")
            resolved_root = _resolve_config_path(base_dir, str(root))
            if not resolved_root.exists() and not allow_missing:
                raise click.ClickException(
                    f"Strict real-data mode: dataset root missing for {ds!r}: {resolved_root}"
                )

    # Execute jobs in order.
    job_summaries: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise click.ClickException("Each jobs[] entry must be a mapping.")
        kind = str(job.get("kind") or "").strip()
        if kind not in SUPPORTED_JOB_KINDS:
            raise click.ClickException(
                f"Unsupported job kind {kind!r}. Supported: {', '.join(sorted(SUPPORTED_JOB_KINDS))}"
            )
        jid = _job_id(job, idx)
        job_dir = output_dir / jid
        job_dir.mkdir(parents=True, exist_ok=True)

        job_cfg_path = _write_job_config(job_dir, job)
        # Per-job run metadata.
        try:
            write_run_metadata(
                job_dir,
                config_path=str(job_cfg_path),
                device=str(device),
                extra={"command": "table2", "mode": "jobs", "job_id": jid, "kind": kind},
            )
        except Exception:
            pass

        # If thresholds were provided at the CLI, inject into raw config for evaluation jobs.
        if characterization_thresholds and kind in {"core_evasion", "stateful_defense_eval"}:
            raw_with_thr = dict(raw)
            raw_with_thr["characterization_thresholds"] = dict(characterization_thresholds)
        else:
            raw_with_thr = raw

        if kind == "core_evasion":
            result = _run_job_core_evasion(
                ctx,
                raw=raw_with_thr,
                job=job,
                job_id=jid,
                job_dir=job_dir,
                device=str(device),
                strict_dataset_budgets=strict_dataset_budgets,
                common_kwargs=common_eval_kwargs,
            )
        elif kind == "baseline_framework":
            result = _run_job_baseline_framework(
                ctx,
                raw=raw_with_thr,
                job=job,
                job_id=jid,
                job_dir=job_dir,
                device=str(device),
                strict_dataset_budgets=strict_dataset_budgets,
            )
        elif kind == "detection_calibration":
            result = _run_job_detection_calibration(
                ctx,
                raw=raw_with_thr,
                job=job,
                job_id=jid,
                job_dir=job_dir,
                device=str(device),
                strict_dataset_budgets=strict_dataset_budgets,
            )
        elif kind == "stateful_defense_eval":
            result = _run_job_stateful_defense_eval(
                ctx,
                raw=raw_with_thr,
                job=job,
                job_id=jid,
                job_dir=job_dir,
                device=str(device),
                strict_dataset_budgets=strict_dataset_budgets,
                common_kwargs=common_eval_kwargs,
            )
        elif kind == "volterra_confound":
            result = _run_job_volterra_confound(
                raw=raw_with_thr,
                job=job,
                job_id=jid,
                job_dir=job_dir,
                device=str(device),
                strict_dataset_budgets=strict_dataset_budgets,
            )
        elif kind == "module_ablation":
            result = _run_job_module_ablation(job=job, job_id=jid, job_dir=job_dir, base_dir=base_dir)
        elif kind == "query_sweep":
            result = _run_job_query_sweep(
                ctx,
                raw=raw_with_thr,
                job=job,
                job_id=jid,
                job_dir=job_dir,
                device=str(device),
                strict_dataset_budgets=strict_dataset_budgets,
                common_kwargs=common_eval_kwargs,
            )
        else:
            raise click.ClickException(f"Unhandled job kind: {kind}")

        result_path = job_dir / "result.json"
        save_json(result, result_path)
        job_summaries.append({"job_id": jid, "kind": kind, "result_json": str(result_path)})

    save_json({"kind": "table2_jobs", "config": str(config_path), "jobs": job_summaries}, output_dir / "jobs_summary.json")

