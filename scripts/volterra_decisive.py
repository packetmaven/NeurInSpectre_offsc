#!/usr/bin/env python3
"""
Issue 6: Volterra Necessity — Decisive Experiment

Runs core evasion against an RL-style obfuscation defense twice:
  1) BPDA+EOT only (Volterra OFF)
  2) BPDA+EOT + Volterra memory (Volterra ON)

The intent is to make the gap obvious in a single, reproducible run.
This script uses real datasets/models available locally (no synthetic data).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class RunMetrics:
    attack_success_rate: float
    clean_accuracy: float
    robust_accuracy: float
    selected_attack_impl: Optional[str]


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _maybe_reset_defense_state(defense: Any) -> None:
    # Best-effort: only our RL-style defense exposes this today.
    if defense is None:
        return
    reset = getattr(defense, "reset_state", None)
    if callable(reset):
        reset()


def _extract_metrics(summary: Dict[str, Any]) -> RunMetrics:
    return RunMetrics(
        attack_success_rate=float(summary.get("attack_success_rate", 0.0)),
        clean_accuracy=float(summary.get("clean_accuracy", 0.0)),
        robust_accuracy=float(summary.get("robust_accuracy", 0.0)),
        selected_attack_impl=summary.get("selected_attack_impl"),
    )


def _print_table(rows: Tuple[Tuple[str, RunMetrics], ...]) -> None:
    # Keep output dependency-free (no rich required).
    headers = ("run", "ASR", "clean_acc", "robust_acc", "impl")
    lines = [headers]
    for name, m in rows:
        lines.append(
            (
                name,
                f"{m.attack_success_rate:.3f}",
                f"{m.clean_accuracy:.3f}",
                f"{m.robust_accuracy:.3f}",
                str(m.selected_attack_impl or "n/a"),
            )
        )
    widths = [max(len(str(col)) for col in col_i) for col_i in zip(*lines)]

    def _fmt(row) -> str:
        return "  ".join(str(col).ljust(w) for col, w in zip(row, widths))

    print(_fmt(lines[0]))
    print(_fmt(tuple("-" * w for w in widths)))
    for row in lines[1:]:
        print(_fmt(row))


def main() -> int:
    p = argparse.ArgumentParser(description="NeurInSpectre decisive Volterra experiment")
    p.add_argument(
        "--model",
        default="models/cifar10_resnet20_norm_ts.pt",
        help="Model path (.pt/.pth/.onnx). Default: models/cifar10_resnet20_norm_ts.pt",
    )
    p.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "imagenet100", "ember", "nuscenes", "custom"],
        help="Dataset name (real datasets only).",
    )
    p.add_argument(
        "--data-path",
        default="data/cifar10",
        help="Dataset root (e.g. data/cifar10).",
    )
    p.add_argument(
        "--labels-path",
        default=None,
        help="nuScenes label map JSON (required if dataset=nuscenes).",
    )
    p.add_argument(
        "--nuscenes-version",
        default="v1.0-mini",
        help="nuScenes version (e.g., v1.0-mini, v1.0-trainval).",
    )
    p.add_argument(
        "--defense",
        default="rl_obfuscation",
        help="Defense name (default: rl_obfuscation).",
    )
    p.add_argument(
        "--defense-config",
        default=None,
        help="Optional defense YAML (overrides built-in defaults).",
    )
    p.add_argument("--epsilon", type=float, default=8 / 255, help="Perturbation budget.")
    p.add_argument(
        "--norm",
        default="Linf",
        choices=["Linf", "L2", "L1"],
        help="Threat model norm.",
    )
    p.add_argument("--iterations", type=int, default=100, help="Attack iterations.")
    p.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="PGD step size alpha used for both runs (default: attack default, typically 2/255).",
    )
    p.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    p.add_argument("--num-samples", type=int, default=200, help="Number of samples to evaluate.")
    p.add_argument(
        "--eot-samples",
        type=int,
        default=10,
        help="EOT samples per step (>=10). Keep fixed across both runs.",
    )
    p.add_argument(
        "--eot-importance-weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use importance-weighted EOT gradients (default: True).",
    )
    p.add_argument(
        "--volterra-kernel",
        default="power_law",
        choices=["power_law", "exponential", "uniform"],
        help="Kernel family for Volterra weighting.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed.")
    p.add_argument(
        "--volterra-alpha",
        type=float,
        default=0.60,
        help="Volterra alpha for memory attack (hybrid-volterra).",
    )
    p.add_argument(
        "--volterra-memory-length",
        type=int,
        default=20,
        help="Volterra memory length k for memory attack (hybrid-volterra).",
    )
    p.add_argument(
        "--characterize",
        action="store_true",
        help="Also run DefenseAnalyzer and print RL/Volterra characterization.",
    )
    p.add_argument(
        "--characterization-samples",
        type=int,
        default=64,
        help="Gradient probes for characterization (only with --characterize).",
    )
    p.add_argument(
        "--rl-bits",
        type=int,
        default=6,
        help="RL obfuscation defense: bit-depth levels (only if defense=rl_obfuscation and no --defense-config).",
    )
    p.add_argument(
        "--rl-std",
        type=float,
        default=0.15,
        help="RL obfuscation defense: noise std (only if defense=rl_obfuscation and no --defense-config).",
    )
    p.add_argument(
        "--rl-alpha",
        type=float,
        default=0.60,
        help="RL obfuscation defense: temporal correlation alpha (only if defense=rl_obfuscation and no --defense-config).",
    )
    p.add_argument(
        "--rl-n-samples",
        type=int,
        default=64,
        help="RL obfuscation defense: eval-time logit averaging samples (only if defense=rl_obfuscation and no --defense-config).",
    )
    p.add_argument(
        "--baseline-max-asr",
        type=float,
        default=0.60,
        help="Assertion gate: BPDA+EOT (no Volterra) must be <= this ASR.",
    )
    p.add_argument(
        "--volterra-min-asr",
        type=float,
        default=0.90,
        help="Assertion gate: Volterra-augmented run must be >= this ASR.",
    )
    p.add_argument(
        "--no-assert",
        action="store_true",
        help="Do not fail even if thresholds are not met (still prints results).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for JSON summary (default: _output/volterra_decisive/<timestamp>/)",
    )

    args = p.parse_args()

    # Import lazily so `--help` works without torch deps.
    from neurinspectre.attacks import AttackFactory
    from neurinspectre.cli.utils import (
        build_defense,
        evaluate_attack_runner,
        load_dataset,
        load_model,
        load_yaml,
        resolve_device,
        set_seed,
    )

    device = resolve_device(args.device)
    set_seed(int(args.seed))

    # Load dataset + model + defense
    if args.dataset == "nuscenes" and not args.labels_path:
        raise SystemExit("dataset=nuscenes requires --labels-path")

    loader, _x, _y = load_dataset(
        args.dataset,
        data_path=args.data_path,
        labels_path=args.labels_path,
        nuscenes_version=args.nuscenes_version,
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
    )
    model = load_model(args.model, dataset=args.dataset, device=device)

    defense_params: Dict[str, Any] = {}
    if args.defense_config:
        defense_params = load_yaml(args.defense_config)
        if "defense" in defense_params and isinstance(defense_params["defense"], dict):
            defense_params = dict(defense_params["defense"])
    elif str(args.defense).lower().replace("-", "_") == "rl_obfuscation":
        # Make the decisive experiment self-contained: default to a tuned
        # RL-style defense parameterization without requiring a YAML file.
        defense_params = {
            "bits": int(args.rl_bits),
            "std": float(args.rl_std),
            "alpha": float(args.rl_alpha),
            "n_samples": int(args.rl_n_samples),
        }
    defense_model = build_defense(str(args.defense), model, defense_params, device=device)
    eval_model = defense_model or model

    if defense_model is None:
        raise SystemExit("This decisive experiment requires a defense (try --defense rl_obfuscation).")

    # Optional characterization (debug/AE evidence)
    characterization: Dict[str, Any] = {}
    if args.characterize:
        from neurinspectre.characterization.defense_analyzer import DefenseAnalyzer

        # Speed: our RL-style defense averages logits at eval-time; enable single-sample
        # forward for gradient probing.
        if hasattr(defense_model, "enable_eot"):
            defense_model.enable_eot()
        try:
            analyzer = DefenseAnalyzer(
                eval_model,
                n_samples=int(args.characterization_samples),
                device=str(device),
                verbose=False,
            )
            char = analyzer.characterize(loader, eps=float(args.epsilon))
            characterization = char.to_dict() if hasattr(char, "to_dict") else {}
        finally:
            if hasattr(defense_model, "disable_eot"):
                defense_model.disable_eot()
            _maybe_reset_defense_state(defense_model)

    # Common attack config across both runs (only difference: Volterra memory).
    base_cfg: Dict[str, Any] = {
        "epsilon": float(args.epsilon),
        "norm": str(args.norm),
        "n_iterations": int(args.iterations),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "eot_samples": int(args.eot_samples),
        "eot_importance_weighting": bool(args.eot_importance_weighting),
    }
    if args.step_size is not None:
        base_cfg["step_size"] = float(args.step_size)

    cfg_hybrid = dict(base_cfg)
    cfg_hybrid["volterra_kernel"] = str(args.volterra_kernel)  # recorded for completeness

    cfg_volterra = dict(base_cfg)
    cfg_volterra["alpha_volterra"] = float(args.volterra_alpha)
    cfg_volterra["memory_length"] = int(args.volterra_memory_length)
    cfg_volterra["volterra_kernel"] = str(args.volterra_kernel)

    def _run(attack_type: str, cfg: Dict[str, Any]) -> RunMetrics:
        runner = AttackFactory.create_attack(
            attack_type,
            model,
            config=cfg,
            defense=defense_model,
            device=str(device),
        )

        # Fair-ish comparison: restart the defense state and RNG for each run.
        set_seed(int(args.seed))
        _maybe_reset_defense_state(defense_model)

        summary = evaluate_attack_runner(
            runner,
            eval_model,
            loader,
            num_samples=int(args.num_samples),
            device=str(device),
            targeted=False,
            save_dir=None,
            norm=str(args.norm),
        )
        try:
            summary["selected_attack_impl"] = runner.attack.__class__.__name__
        except Exception:
            summary["selected_attack_impl"] = None
        return _extract_metrics(summary)

    start = time.time()
    print("Volterra decisive experiment")
    print(f"  model:   {args.model}")
    print(f"  dataset: {args.dataset} (root={args.data_path})")
    print(f"  defense: {args.defense}")
    if defense_params:
        print(f"  defense_params: {json.dumps(defense_params, sort_keys=True)}")
    print(
        "  volterra_params: "
        f"kernel={args.volterra_kernel} alpha={float(args.volterra_alpha):.3f} "
        f"k={int(args.volterra_memory_length)}"
    )
    if characterization:
        obf = characterization.get("obfuscation_types") or []
        print(
            "  characterization: "
            f"obf={','.join(map(str, obf))} "
            f"alpha={characterization.get('alpha_volterra')} "
            f"k={characterization.get('recommended_memory_length')} "
            f"bpda={characterization.get('requires_bpda')} "
            f"eot={characterization.get('requires_eot')} "
            f"mapgd={characterization.get('requires_mapgd')}"
        )

    no_volterra = _run("hybrid", cfg_hybrid)
    with_volterra = _run("hybrid-volterra", cfg_volterra)
    elapsed = time.time() - start

    _print_table(
        (
            ("bpda+eot (hybrid)", no_volterra),
            ("bpda+eot+volterra (hybrid-volterra)", with_volterra),
        )
    )
    print(f"elapsed_s: {elapsed:.2f}")

    out_dir = Path(args.out_dir or f"_output/volterra_decisive/{_now_tag()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "model": str(args.model),
            "dataset": str(args.dataset),
            "data_path": str(args.data_path),
            "defense": str(args.defense),
            "defense_params": dict(defense_params),
            "device": str(device),
            "seed": int(args.seed),
            "epsilon": float(args.epsilon),
            "norm": str(args.norm),
            "iterations": int(args.iterations),
            "step_size": float(args.step_size) if args.step_size is not None else None,
            "eot_samples": int(args.eot_samples),
            "eot_importance_weighting": bool(args.eot_importance_weighting),
            "volterra_kernel": str(args.volterra_kernel),
            "volterra_alpha": float(args.volterra_alpha),
            "volterra_memory_length": int(args.volterra_memory_length),
            "num_samples": int(args.num_samples),
        },
        "characterization": dict(characterization),
        "results": {
            "bpda_eot_only": asdict(no_volterra),
            "volterra_augmented": asdict(with_volterra),
        },
    }
    (out_dir / "volterra_decisive.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote: {out_dir / 'volterra_decisive.json'}")

    if not args.no_assert:
        if no_volterra.attack_success_rate > float(args.baseline_max_asr):
            print(
                "ASSERTION FAILED: BPDA+EOT baseline ASR "
                f"{no_volterra.attack_success_rate:.3f} > {float(args.baseline_max_asr):.3f}"
            )
            return 2
        if with_volterra.attack_success_rate < float(args.volterra_min_asr):
            print(
                "ASSERTION FAILED: Volterra-augmented ASR "
                f"{with_volterra.attack_success_rate:.3f} < {float(args.volterra_min_asr):.3f}"
            )
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

