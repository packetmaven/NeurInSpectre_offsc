"""
Evaluation command implementation for NeurInSpectre CLI.

Implements comprehensive defense evaluation suite from Paper Section 5.
Evaluates multiple defenses against ensemble of attacks with:
  - Parallel execution (Enhancement #8)
  - Rich progress reporting (Enhancement #11)
  - Attack orchestration (APGD-CE, APGD-DLR, NeurInSpectre, Square)
  - Cross-product evaluation matrix (defenses x attacks)

PAPER ALIGNMENT:
- Evaluates N defenses x M attacks from config YAML
- Table 1 reproduction pipeline (12 defenses, 4 attacks per defense)
- Coordinated evaluation per Paper Section 3.3
- Ensemble methodology per AutoAttack integration

ENHANCEMENTS:
#8  - ThreadPoolExecutor for parallel (defense, attack) pair evaluation
#11 - Rich progress bars with real-time ASR/RA updates per pair

Cross-ref: Paper Section 5 "Evaluation"
Cross-ref: Paper Table 1 "Attack success rate against evaluated defenses"
Cross-ref: Paper Section 3.3 "Phase 3: Coordinated Evaluation"
Cross-ref: Paper Section 4 "Implementation" (parallelization)

Version: 2.0.1 (WOOT 2026 submission aligned)
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click

from ..attacks import AttackFactory
from ..evaluation.datasets import DatasetFactory
from .exporters import export_evaluation_json, export_evaluation_sarif
from .formatters import build_console, render_evaluation_report
from .progress import ProgressReporter
from .utils import (
    build_defense,
    evaluate_attack_runner,
    load_dataset,
    load_model,
    load_yaml,
    resolve_device,
    save_json,
    set_seed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper-aligned constants
# ---------------------------------------------------------------------------

# Paper Table 1: 12 defenses evaluated
DEFAULT_NUM_DEFENSES = 12

# Paper Section 3.3: "ensemble of attacks" (4 attacks per defense)
DEFAULT_ATTACKS_PER_DEFENSE = 4

# Paper Section 4: "full evaluation takes 2-8 hours depending on defense complexity"
EXPECTED_EVALUATION_HOURS_MIN = 2
EXPECTED_EVALUATION_HOURS_MAX = 8

# Paper Section 5.1: 1000 samples for evaluation
DEFAULT_EVAL_SAMPLES = 1000

# Enhancement #8: Parallelization
# Paper Section 4: "All attacks are fully batched and GPU-accelerated"
DEFAULT_PARALLEL_WORKERS = 1  # Conservative default (user must opt-in)
RECOMMENDED_MAX_WORKERS = 4   # Balances GPU memory vs throughput


def run_evaluation(ctx: click.Context, **kwargs: Any) -> None:
    cmd_verbose = int(kwargs.get("verbose", 0) or 0)
    if ctx is not None:
        ctx.obj = ctx.obj or {}
        if cmd_verbose:
            ctx.obj["verbose"] = max(int(ctx.obj.get("verbose", 0)), cmd_verbose)
    device = resolve_device(kwargs.get("device", "cpu"))
    config_path = kwargs.get("config")
    config = load_yaml(config_path)

    quiet = bool(ctx.obj.get("quiet", False)) if ctx and ctx.obj else False
    no_progress = bool(kwargs.get("no_progress", False))
    report_format = str(kwargs.get("report_format", "rich"))
    no_color = bool(kwargs.get("no_color", False))
    force_color = bool(kwargs.get("color", False))
    console = build_console(no_color=no_color, force_color=force_color)

    out_dir = Path(kwargs.get("output_dir", "evaluation_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    defense_filter = set(kwargs.get("defenses") or [])
    attack_filter = set(kwargs.get("attacks") or [])
    parallel = max(1, int(kwargs.get("parallel", DEFAULT_PARALLEL_WORKERS)))
    resume = bool(kwargs.get("resume", False))
    summary_only = bool(kwargs.get("summary_only", False))

    seed = int(config.get("seed", 42))
    set_seed(seed)

    defenses = _normalize_defenses(config.get("defenses", []))
    attacks = _normalize_attacks(config.get("attacks", []))
    if defense_filter:
        defenses = [d for d in defenses if d["name"] in defense_filter or d["type"] in defense_filter]
    if attack_filter:
        attacks = [a for a in attacks if a["name"] in attack_filter]

    if parallel > RECOMMENDED_MAX_WORKERS:
        logger.warning(
            "Requested parallel workers=%d exceeds recommended maximum (%d). "
            "Paper Section 4: GPU memory contention may reduce throughput.",
            parallel,
            RECOMMENDED_MAX_WORKERS,
        )

    if len(defenses) == DEFAULT_NUM_DEFENSES and len(attacks) == DEFAULT_ATTACKS_PER_DEFENSE:
        logger.info(
            "Table 1 reproduction mode detected: %d defenses x %d attacks.",
            DEFAULT_NUM_DEFENSES,
            DEFAULT_ATTACKS_PER_DEFENSE,
        )
    else:
        logger.info(
            "Evaluation matrix: %d defenses x %d attacks (Paper Table 1 default: %d x %d).",
            len(defenses),
            len(attacks),
            DEFAULT_NUM_DEFENSES,
            DEFAULT_ATTACKS_PER_DEFENSE,
        )

    base_attack_cfg = _base_attack_config(config)

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    total_units = len(defenses) if parallel > 1 else len(defenses) * max(1, len(attacks))
    with ProgressReporter(
        description="Evaluation suite...",
        total=total_units,
        enabled=not no_progress and not quiet,
        console=console,
    ) as progress:
        if parallel > 1:
            logger.info(
                "Running parallel evaluation with %d workers (Paper Section 4).",
                parallel,
            )
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(
                        _evaluate_defense,
                        defense,
                        attacks,
                        config,
                        base_attack_cfg,
                        device,
                        out_dir,
                        resume,
                        seed,
                        None,
                    ): defense
                    for defense in defenses
                }
                for future in as_completed(futures):
                    defense_entry = futures[future]
                    defense_name = defense_entry.get("name", defense_entry.get("type", "defense"))
                    try:
                        result = future.result()
                        results.append(result)
                        avg_asr = _compute_avg_asr(result)
                        progress.advance(
                            1,
                            description=f"Evaluation suite... [{defense_name}: ASR={avg_asr:.1%}]",
                        )
                    except Exception as exc:
                        logger.error(
                            "Defense %s failed: %s",
                            defense_name,
                            exc,
                            exc_info=True,
                        )
        else:
            for defense in defenses:
                results.append(
                    _evaluate_defense(
                        defense,
                        attacks,
                        config,
                        base_attack_cfg,
                        device,
                        out_dir,
                        resume,
                        seed,
                        progress.advance,
                    )
                )

    summary = {
        "config": config,
        "results": results,
        "timing": {
            "total_seconds": float(time.time() - start_time),
            "defenses": len(results),
        },
    }
    summary["highlights"] = _summarize_evaluation_findings(results, config=config)
    save_json(summary, out_dir / "summary.json")
    click.echo(f"Evaluation summary written to {out_dir / 'summary.json'}")

    report = bool(kwargs.get("report", True))
    brief = bool(kwargs.get("brief", False))
    if report and not quiet:
        render_evaluation_report(
            console,
            results=results,
            config=config,
            output_dir=str(out_dir),
            report_format=report_format,
            brief=brief,
            summary_only=summary_only,
        )

    json_output = kwargs.get("json_output")
    if json_output:
        export_evaluation_json(
            {
                "config": config,
                "results": results,
            },
            json_output,
        )
        click.echo(f"JSON report written to {json_output}")

    sarif_output = kwargs.get("sarif_output")
    if sarif_output:
        export_evaluation_sarif(
            {
                "pairs": _evaluation_pairs(results),
            },
            sarif_output,
        )
        click.echo(f"SARIF report written to {sarif_output}")


def _normalize_defenses(defenses_cfg: Any) -> List[Dict[str, Any]]:
    if isinstance(defenses_cfg, dict):
        return [{"name": name, **(cfg or {})} for name, cfg in defenses_cfg.items()]
    if isinstance(defenses_cfg, list):
        entries = []
        for item in defenses_cfg:
            if isinstance(item, str):
                entries.append({"name": item, "type": item})
            elif isinstance(item, dict):
                entry = dict(item)
                entry.setdefault("name", entry.get("type", "defense"))
                entries.append(entry)
            else:
                raise ValueError("Defense entry must be a string or mapping.")
        return entries
    raise ValueError("defenses must be a list or mapping.")


def _normalize_attacks(attacks_cfg: Any) -> List[Dict[str, Any]]:
    if isinstance(attacks_cfg, dict):
        return [{"name": name, **(cfg or {})} for name, cfg in attacks_cfg.items()]
    if isinstance(attacks_cfg, list):
        entries = []
        for item in attacks_cfg:
            if isinstance(item, str):
                entries.append({"name": item})
            elif isinstance(item, dict):
                entry = dict(item)
                entry.setdefault("name", entry.get("type", "attack"))
                entries.append(entry)
            else:
                raise ValueError("Attack entry must be a string or mapping.")
        return entries
    raise ValueError("attacks must be a list or mapping.")


def _base_attack_config(config: Dict[str, Any]) -> Dict[str, Any]:
    perturbation = config.get("perturbation", {})
    return {
        "epsilon": float(perturbation.get("epsilon", 8 / 255)),
        "norm": str(perturbation.get("norm", "Linf")),
        "n_iterations": int(config.get("iterations", config.get("attack_iterations", 100))),
        "batch_size": int(config.get("attack_batch_size", 128)),
        "seed": int(config.get("seed", 42)),
    }


def _compute_avg_asr(result: Dict[str, Any]) -> float:
    attacks = result.get("attacks", {}) or {}
    if not attacks:
        return 0.0
    values = [float(v.get("attack_success_rate", 0.0)) for v in attacks.values()]
    return float(sum(values) / max(1, len(values)))


def _summarize_evaluation_findings(
    results: List[Dict[str, Any]],
    *,
    config: Dict[str, Any],
    top_n: int = 5,
) -> Tuple[str, ...]:
    entries: List[Tuple[float, float, str, str, float, float]] = []
    per_defense: List[Tuple[float, str, str, float]] = []

    for res in results:
        defense = str(res.get("defense", res.get("type", "defense")))
        attacks = res.get("attacks", {}) or {}
        best_asr = -1.0
        best_attack = "unknown"
        best_robust = 0.0
        for attack_name, metrics in attacks.items():
            asr = float(metrics.get("attack_success_rate", 0.0))
            robust = float(metrics.get("robust_accuracy", 0.0))
            clean = float(metrics.get("clean_accuracy", 0.0))
            drop = clean - robust
            entries.append((asr, drop, defense, str(attack_name), robust, clean))
            if asr > best_asr:
                best_asr = asr
                best_attack = str(attack_name)
                best_robust = robust
        if best_asr >= 0:
            per_defense.append((best_asr, defense, best_attack, best_robust))

    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_entries = entries[: max(1, int(top_n))]
    top_str = ", ".join(
        f"{defense}/{attack} ASR={asr:.3f} RA={robust:.3f}"
        for asr, _drop, defense, attack, robust, _clean in top_entries
    )

    per_defense.sort(key=lambda item: item[0], reverse=True)
    weak_str = ", ".join(
        f"{defense}({attack} ASR={asr:.3f} RA={robust:.3f})"
        for asr, defense, attack, robust in per_defense[: max(1, min(3, len(per_defense)))]
    )

    flagged = [
        entry
        for entry in entries
        if entry[0] >= 0.5 or entry[1] >= 0.3
    ]
    epsilon = config.get("perturbation", {}).get("epsilon", 8 / 255)
    norm = config.get("perturbation", {}).get("norm", "Linf")

    lines = [
        f"Threat model: norm={norm}, epsilon={epsilon}",
        f"Top compromise pairs (ASR): {top_str}" if top_str else "Top compromise pairs (ASR): n/a",
        f"Weakest defenses (best attack ASR): {weak_str}" if weak_str else "Weakest defenses: n/a",
        f"Flagged pairs (ASR>=0.5 or drop>=0.3): {len(flagged)}/{len(entries)}",
    ]
    return tuple(lines)


def _evaluation_pairs(results: List[Dict[str, Any]]) -> List[Tuple[str, str, float, float]]:
    pairs: List[Tuple[str, str, float, float]] = []
    for res in results:
        defense = str(res.get("defense", res.get("type", "defense")))
        attacks = res.get("attacks", {}) or {}
        for attack_name, metrics in attacks.items():
            asr = float(metrics.get("attack_success_rate", 0.0))
            clean = float(metrics.get("clean_accuracy", 0.0))
            robust = float(metrics.get("robust_accuracy", 0.0))
            drop = clean - robust
            pairs.append((defense, str(attack_name), asr, drop))
    return pairs


def _evaluate_defense(
    defense_entry: Dict[str, Any],
    attacks: List[Dict[str, Any]],
    config: Dict[str, Any],
    base_attack_cfg: Dict[str, Any],
    device: str,
    out_dir: Path,
    resume: bool,
    seed: int,
    progress_callback: Any,
) -> Dict[str, Any]:
    defense_name = defense_entry.get("name", defense_entry.get("type", "defense"))
    defense_type = defense_entry.get("type", defense_name)

    output_path = out_dir / f"{defense_name}.json"
    if resume and output_path.exists():
        return _load_json(output_path)

    datasets_cfg = config.get("datasets", {})
    dataset_name = _resolve_dataset(defense_entry, datasets_cfg)
    dataset_cfg = dict(datasets_cfg.get(dataset_name, {}))
    if "path" in dataset_cfg and "data_path" not in dataset_cfg:
        dataset_cfg["data_path"] = dataset_cfg["path"]

    num_samples = int(dataset_cfg.get("num_samples", dataset_cfg.get("n_samples", DEFAULT_EVAL_SAMPLES)))
    batch_size = int(dataset_cfg.get("batch_size", base_attack_cfg.get("batch_size", 128)))
    num_workers = int(dataset_cfg.get("num_workers", 4))

    loader, _x, _y = load_dataset(
        dataset_name,
        data_path=dataset_cfg.get("data_path"),
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )

    model_ref = _resolve_model_ref(defense_entry, config, dataset_name)
    model = load_model(model_ref, dataset=dataset_name, device=device)

    defense_params = {
        k: v
        for k, v in defense_entry.items()
        if k
        not in {
            "name",
            "type",
            "dataset",
            "model",
            "model_path",
            "model_name",
            "model_factory",
        }
    }
    if "defense_config" in defense_params or "config_path" in defense_params:
        cfg_path = defense_params.pop("defense_config", None) or defense_params.pop("config_path", None)
        if cfg_path:
            loaded = load_yaml(cfg_path)
            if "defense" in loaded and isinstance(loaded["defense"], dict):
                loaded = loaded["defense"]
            defense_params.update(loaded)
    defense_model = build_defense(defense_type, model, defense_params, device=device)
    eval_model = defense_model or model

    attack_results: Dict[str, Any] = {}
    characterization: Dict[str, Any] = {}

    for attack_entry in attacks:
        attack_name = attack_entry.get("name")
        attack_cfg = dict(base_attack_cfg)
        attack_cfg.update(
            {
                k: v
                for k, v in attack_entry.items()
                if k not in {"name", "type"}
            }
        )
        runner = AttackFactory.create_attack(
            attack_name,
            model,
            config=attack_cfg,
            characterization_loader=loader if attack_name == "neurinspectre" else None,
            defense=defense_model,
            device=device,
        )
        summary = evaluate_attack_runner(
            runner,
            eval_model,
            loader,
            num_samples=num_samples,
            device=device,
            targeted=bool(attack_cfg.get("targeted", False)),
            norm=attack_cfg.get("norm", "Linf"),
        )
        attack_results[attack_name] = summary

        if progress_callback:
            asr = float(summary.get("attack_success_rate", 0.0))
            robust = float(summary.get("robust_accuracy", 0.0))
            progress_callback(
                1,
                description=f"{defense_name} vs {attack_name} ASR={asr:.1%} RA={robust:.1%}",
            )

        if attack_name == "neurinspectre" and hasattr(runner, "characterization"):
            try:
                characterization = runner.characterization.to_dict()
            except Exception:
                characterization = {}

    result = {
        "defense": defense_name,
        "type": defense_type,
        "dataset": dataset_name,
        "attacks": attack_results,
        "characterization": characterization,
    }
    save_json(result, output_path)
    return result


def _resolve_dataset(defense_entry: Dict[str, Any], datasets_cfg: Dict[str, Any]) -> str:
    if "dataset" in defense_entry:
        return str(defense_entry["dataset"])

    defense_key = str(defense_entry.get("type", defense_entry.get("name", ""))).lower()
    alias_map = {
        "jpeg": "jpeg_compression",
        "bitdepth": "bit_depth_reduction",
        "randsmooth": "randomized_smoothing",
        "thermometer": "thermometer_encoding",
        "distillation": "defensive_distillation",
        "ensemble": "ensemble_diversity",
    }
    defense_key = alias_map.get(defense_key, defense_key)
    mapped = DatasetFactory.DEFENSE_TO_DATASET.get(defense_key)
    if mapped:
        return mapped

    if len(datasets_cfg) == 1:
        return next(iter(datasets_cfg.keys()))

    raise ValueError("Dataset not specified for defense and could not be inferred.")


def _resolve_model_ref(defense_entry: Dict[str, Any], config: Dict[str, Any], dataset_name: str) -> Any:
    if "model" in defense_entry:
        return defense_entry["model"]
    if "model_path" in defense_entry:
        return defense_entry["model_path"]
    if "model_name" in defense_entry:
        return {"model_name": defense_entry["model_name"]}

    models_cfg = config.get("models") or {}
    if isinstance(models_cfg, dict):
        if defense_entry.get("name") in models_cfg:
            return models_cfg[defense_entry["name"]]
        if dataset_name in models_cfg:
            return models_cfg[dataset_name]

    datasets_cfg = config.get("datasets", {})
    dataset_cfg = datasets_cfg.get(dataset_name, {}) if isinstance(datasets_cfg, dict) else {}
    if isinstance(dataset_cfg, dict):
        if "model" in dataset_cfg:
            return dataset_cfg["model"]
        if "model_name" in dataset_cfg:
            return {"model_name": dataset_cfg["model_name"]}

    global_model = config.get("model") or config.get("model_name")
    if global_model:
        return global_model

    default_models = {
        "cifar10": "resnet18",
        "cifar100": "resnet18",
        "imagenet": "resnet50",
        "imagenet100": "resnet50",
    }
    if dataset_name in default_models:
        return {"model_name": default_models[dataset_name], "domain": "vision"}

    raise ValueError("Model configuration missing for evaluation.")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
