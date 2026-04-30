#!/usr/bin/env python3
"""
Run NeurInSpectre 3-layer characterization on real CIFAR-10 defenses.

Uses the adversarially-trained Carmon2019 WRN-28-10 (RobustBench-style
checkpoint) with 200-step gradient trajectories.

Produces results/real_defense_characterization.json with Layer 1 (spectral),
Layer 2 (Volterra), and Layer 3 (Krylov) features for:
  1. No defense (clean baseline)
  2. JPEG compression (shattered gradients)
  3. Randomized smoothing (stochastic defense)
  4. Spatial smoothing (differentiable, vanishing-type)

Usage:
    python scripts/run_real_defense_experiments.py [--device auto] [--n-samples 200]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurinspectre.characterization.defense_analyzer import DefenseAnalyzer
from neurinspectre.defenses import DefenseFactory
from neurinspectre.evaluation.datasets import CIFAR10Dataset


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_carmon2019(
    checkpoint_path: str,
    device: str,
    *,
    cross_verify_with_robustbench: bool = False,
) -> nn.Module:
    """Load Carmon2019Unlabeled WRN-28-10 with zero runtime external deps.

    Uses the vendored, attributed :class:`WideResNetCarmon` module at
    ``neurinspectre/models/wide_resnet_carmon.py`` (MIT-licensed copy of
    the RobustBench architecture, with the unused ``sub_block1`` branch
    preserved so that ``load_state_dict(strict=True)`` succeeds on the
    canonical checkpoint). This removes the runtime dependency on
    ``robustbench`` + ``gdown`` + Google-Drive downloads that would
    otherwise be a long-term reproducibility risk for artifact
    evaluation.

    A post-load clean-accuracy probe (>=88% on 256 CIFAR-10 test images)
    catches silent regressions. For extra safety, pass
    ``cross_verify_with_robustbench=True`` to additionally cross-check
    against the canonical RobustBench loader (requires
    ``pip install robustbench``; asserts max|logit-diff| < 1e-4 on a
    seeded random probe).

    Audit history (April 2026)
    --------------------------
    A previous version of this loader used a locally-defined
    ``WideResNet`` with ``strict=False`` and silently returned a
    randomly-initialised network (~10% clean accuracy), because the
    local class's state-dict keys did not match the canonical
    ``block{i}.layer.{j}.*`` / ``convShortcut`` / ``sub_block1.*``
    layout. Every gradient / spectral / Volterra / Krylov number
    produced by the detection paper's real-defense Table 5 was
    therefore measured on random weights. The vendored
    :class:`WideResNetCarmon` preserves the canonical layout so that
    ``strict=True`` loading succeeds natively.

    Parameters
    ----------
    checkpoint_path
        Path to ``Carmon2019Unlabeled.pt`` (obtained via
        ``scripts/download_carmon2019.py``).
    device
        Target device.
    cross_verify_with_robustbench
        If True and ``robustbench`` is installed, cross-verify logits
        against RobustBench's canonical loader on a seeded probe.

    Returns
    -------
    nn.Module
        Evaluation-mode WRN-28-10 achieving 89.69% +/- (subset-variation)
        clean accuracy on CIFAR-10.
    """
    from neurinspectre.models.wide_resnet_carmon import (
        load_carmon2019_local,
    )

    return load_carmon2019_local(
        str(checkpoint_path),
        device=device,
        assert_clean_accuracy=True,
        min_clean_accuracy=0.88,
        sanity_n_samples=256,
        cross_verify_with_robustbench=cross_verify_with_robustbench,
    )


EXPERIMENTS = [
    {
        "name": "No defense (clean)",
        "defense": None,
        "defense_params": {},
        "obfuscation_type": "none",
    },
    {
        "name": "JPEG compression (q=75)",
        "defense": "jpeg_compression",
        "defense_params": {"quality": 75},
        "obfuscation_type": "shattered",
    },
    {
        "name": "Randomized smoothing (sigma=0.25)",
        "defense": "randomized_smoothing",
        "defense_params": {"sigma": 0.25, "n_samples": 50},
        "obfuscation_type": "stochastic",
    },
    {
        "name": "Spatial smoothing (3x3, sigma=1.0)",
        "defense": "spatial_smoothing",
        "defense_params": {"kernel_size": 3, "sigma": 1.0},
        "obfuscation_type": "vanishing",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Real defense experiments for CCS paper")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Gradient trajectory length (PGD steps)")
    parser.add_argument("--n-probe-images", type=int, default=100)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-path", default="models/cifar10/Linf/Carmon2019Unlabeled.pt")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"[Config] device={device}, n_samples={args.n_samples}, "
          f"n_probe_images={args.n_probe_images}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    base_model = load_carmon2019(str(model_path), device)
    print(f"[Model] Loaded Carmon2019 WRN-28-10 from {model_path}")

    x_test_sample = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        out = base_model(x_test_sample)
    print(f"[Model] Smoke test OK: output shape={out.shape}")

    loader, x_test, y_test = CIFAR10Dataset.load(
        root="./data/cifar10",
        n_samples=args.n_probe_images,
        seed=42,
        batch_size=50,
        num_workers=0,
        split="test",
        download=False,
        pin_memory=False,
    )
    print(f"[Data] CIFAR-10 test: {len(x_test)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    eps = 8.0 / 255.0

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  Experiment: {exp['name']}")
        print(f"  Defense: {exp['defense'] or 'none'}")
        print(f"{'='*60}")

        if exp["defense"] is not None:
            eval_model = DefenseFactory.create_defense(
                exp["defense"], base_model, exp["defense_params"]
            )
        else:
            eval_model = base_model

        eval_model_on_device = eval_model.to(device) if hasattr(eval_model, "to") else eval_model

        analyzer = DefenseAnalyzer(
            model=eval_model_on_device,
            n_samples=args.n_samples,
            n_probe_images=args.n_probe_images,
            device=device,
            verbose=True,
            krylov_dim=20,
        )

        t0 = time.time()
        try:
            char = analyzer.characterize(loader, eps=eps)
            elapsed = time.time() - t0

            meta = char.metadata
            result = {
                "experiment": exp["name"],
                "defense": exp["defense"] or "none",
                "expected_obfuscation": exp["obfuscation_type"],
                "detected_obfuscation": [o.value for o in char.obfuscation_types],
                "layer1_spectral_entropy_norm": float(meta.get("spectral_entropy_norm", 0.0)),
                "layer1_high_freq_ratio": float(meta.get("high_freq_ratio", 0.0)),
                "layer2_alpha_volterra": float(char.alpha_volterra),
                "layer2_volterra_rmse": float(meta.get("volterra_rmse", 0.0)),
                "layer3_krylov_rel_error_mean": float(meta.get("krylov_rel_error_mean", 0.0) or 0.0),
                "layer3_krylov_norm_ratio_mean": float(meta.get("krylov_norm_ratio_mean", 0.0) or 0.0),
                "layer3_dissipation_anomaly_score": float(meta.get("krylov_dissipation_anomaly_score", 0.0) or 0.0),
                "etd_score": float(char.etd_score),
                "confidence": float(char.confidence),
                "paper_composite_detected": bool(meta.get("paper_style", {}).get("composite_obfuscated", False)),
                "paper_triggers": list(meta.get("paper_style", {}).get("triggers", [])),
                "requires_bpda": bool(char.requires_bpda),
                "requires_eot": bool(char.requires_eot),
                "grad_norm_mean": float(meta.get("grad_norm_mean", 0.0)),
                "stochastic_score": float(meta.get("stochastic_score", 0.0)),
                "elapsed_seconds": round(elapsed, 2),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            elapsed = time.time() - t0
            result = {
                "experiment": exp["name"],
                "defense": exp["defense"] or "none",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
            }

        results.append(result)

        if "error" not in result:
            print("\n  --- Layer 1 (Spectral) ---")
            print(f"    H_S (norm):  {result['layer1_spectral_entropy_norm']:.4f}")
            print(f"    R_HF:        {result['layer1_high_freq_ratio']:.4f}")
            print("  --- Layer 2 (Volterra) ---")
            print(f"    alpha:       {result['layer2_alpha_volterra']:.4f}")
            print(f"    RMSE:        {result['layer2_volterra_rmse']:.6f}")
            print("  --- Layer 3 (Krylov) ---")
            print(f"    recon error: {result['layer3_krylov_rel_error_mean']:.6f}")
            print(f"    norm ratio:  {result['layer3_krylov_norm_ratio_mean']:.4f}")
            print("  --- Verdict ---")
            print(f"    Detected:    {result['paper_composite_detected']}")
            print(f"    Triggers:    {result['paper_triggers']}")
            print(f"    Obf types:   {result['detected_obfuscation']}")
            print(f"    Time:        {result['elapsed_seconds']}s")

    out_path = os.path.join(args.output_dir, "real_defense_characterization.json")
    with open(out_path, "w") as f:
        json.dump({"experiments": results, "config": {
            "model": str(args.model_path),
            "model_name": "Carmon2019Unlabeled (WRN-28-10)",
            "dataset": "cifar10",
            "n_samples": args.n_samples,
            "n_probe_images": args.n_probe_images,
            "eps": eps,
            "device": device,
        }}, f, indent=2)
    print(f"\n[Done] Results saved to {out_path}")

    print("\n" + "="*80)
    print("SUMMARY TABLE (for paper Table 5)")
    print("="*80)
    hdr = f"{'Defense':<40} {'H_S':>6} {'R_HF':>6} {'alpha':>6} {'Krylov':>8} {'Det':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['experiment']:<40} ERROR: {r['error']}")
            continue
        det = "Yes" if r["paper_composite_detected"] else "No"
        print(f"{r['experiment']:<40} "
              f"{r['layer1_spectral_entropy_norm']:>6.3f} "
              f"{r['layer1_high_freq_ratio']:>6.3f} "
              f"{r['layer2_alpha_volterra']:>6.3f} "
              f"{r['layer3_krylov_rel_error_mean']:>8.4f} "
              f"{det:>5}")


if __name__ == "__main__":
    main()
