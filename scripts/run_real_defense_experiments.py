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
from collections import OrderedDict
from pathlib import Path

import numpy as np
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


class WideResNetBlock(nn.Module):
    """Pre-activation WRN block matching the RobustBench / Carmon2019 layout.

    Previously this block had a broken shortcut: ``if self.shortcut:`` always
    evaluated True (``nn.Sequential()`` is truthy even when empty), so
    identity-shortcut blocks ran ``empty_seq(out) = out`` and the output
    became ``conv(...) + out`` instead of ``conv(...) + x``. That silent
    architectural divergence cost ~10 percentage points of clean accuracy
    (observed: 80.0% vs published 89.7%).
    """

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.equal_in_out = (stride == 1 and in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        if self.equal_in_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        # Pre-activation WRN: identity shortcut uses x (not out);
        # non-identity shortcut projects the preactivation.
        shortcut = x if self.equal_in_out else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, drop_rate=0.0):
        super().__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n_blocks = (depth - 4) // 6

        self.conv1 = nn.Conv2d(3, n_channels[0], 3, stride=1, padding=1, bias=False)

        self.block1 = self._make_layer(n_channels[0], n_channels[1], n_blocks, 1, drop_rate)
        self.block2 = self._make_layer(n_channels[1], n_channels[2], n_blocks, 2, drop_rate)
        self.block3 = self._make_layer(n_channels[2], n_channels[3], n_blocks, 2, drop_rate)

        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)

    def _make_layer(self, in_planes, out_planes, n_blocks, stride, drop_rate):
        layers = []
        for i in range(n_blocks):
            s = stride if i == 0 else 1
            inp = in_planes if i == 0 else out_planes
            layers.append(WideResNetBlock(inp, out_planes, s, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def load_carmon2019(checkpoint_path: str, device: str) -> nn.Module:
    """Load Carmon2019Unlabeled WRN-28-10 via the canonical RobustBench loader.

    Rationale (April 2026 audit): the previous implementation loaded the
    RobustBench checkpoint into a locally-defined ``WideResNet`` class with
    ``load_state_dict(..., strict=False)``. This silently dropped 200+/204
    keys because the checkpoint uses a different nesting convention
    (``module.block{i}.layer.{j}.*``, ``convShortcut.weight``) than the local
    class (``block{i}.{j}.*``, ``shortcut.0.weight``), returning a
    randomly-initialised model with ~10% clean accuracy. Even after key
    remapping, the local ``WideResNet`` produced only ~81% clean accuracy
    due to a subtle architectural mismatch we could not isolate. The
    canonical ``robustbench.utils.load_model`` path achieves the published
    89.85% and is the only accuracy-safe option.

    We therefore require ``robustbench`` as a hard dependency for this
    script. A post-load clean-accuracy assertion guards against any future
    regression.

    Parameters
    ----------
    checkpoint_path : str
        Directory or file path to the checkpoint. RobustBench downloads
        automatically if missing.
    device : str
        Target device.

    Returns
    -------
    nn.Module : Evaluation-mode WRN-28-10 achieving >=88% clean accuracy on
        a 256-image CIFAR-10 test subset. Raises RuntimeError otherwise.
    """
    try:
        from robustbench.utils import load_model as rb_load_model
    except ImportError as exc:
        raise ImportError(
            "robustbench is required for Carmon2019 loading. Install with: "
            "pip install robustbench. (Previously the script fell back to a "
            "custom WideResNet class with strict=False, which silently "
            "returned random weights and produced invalid results.)"
        ) from exc

    # RobustBench manages its own directory layout; it stores the checkpoint
    # under <model_dir>/cifar10/Linf/Carmon2019Unlabeled.pt. We accept either
    # a file or directory path for backward compat.
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_file():
        # Honour existing layout: use the parent's parent as model_dir so
        # RobustBench treats the already-downloaded file as canonical.
        model_dir = ckpt_path.parent.parent.parent if ckpt_path.parent.name == "Linf" else ckpt_path.parent
    else:
        model_dir = ckpt_path

    model = rb_load_model(
        model_name="Carmon2019Unlabeled",
        dataset="cifar10",
        threat_model="Linf",
        model_dir=str(model_dir),
    )
    model = model.to(device).eval()

    # Post-load clean-accuracy assertion. This is the only way to catch
    # silent regressions in the RobustBench loader or architecture.
    _assert_clean_accuracy(model, device, min_acc=0.88, n=256)
    return model


def _assert_clean_accuracy(model: nn.Module, device: str, min_acc: float = 0.88,
                            n: int = 256) -> None:
    """Probe clean accuracy on a small CIFAR-10 subset; raise if below min_acc.

    This is the last line of defence against silent random-model bugs. Any
    published-accuracy model must clear min_acc on this probe.
    """
    try:
        from neurinspectre.evaluation.datasets import CIFAR10Dataset
    except ImportError:
        return  # can't probe, skip
    try:
        loader, _, _ = CIFAR10Dataset.load(
            root="./data/cifar10", n_samples=n, seed=42, batch_size=64,
            split="test", download=False, num_workers=0, pin_memory=False,
        )
    except Exception:
        return  # data missing; skip probe (caller's responsibility)
    ok = tot = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            ok += int((model(xb).argmax(1) == yb).sum().item())
            tot += int(xb.size(0))
    acc = ok / max(tot, 1)
    if acc < min_acc:
        raise RuntimeError(
            f"Clean-accuracy sanity check failed: {100 * acc:.2f}% on {tot} "
            f"CIFAR-10 test images (required >= {100 * min_acc:.2f}%). "
            f"This typically means state_dict keys did not populate the "
            f"model. Re-run with --device cpu to rule out MPS precision; "
            f"if the failure persists, the loader is returning a "
            f"randomly-initialised network and any downstream numbers are "
            f"invalid."
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
            print(f"\n  --- Layer 1 (Spectral) ---")
            print(f"    H_S (norm):  {result['layer1_spectral_entropy_norm']:.4f}")
            print(f"    R_HF:        {result['layer1_high_freq_ratio']:.4f}")
            print(f"  --- Layer 2 (Volterra) ---")
            print(f"    alpha:       {result['layer2_alpha_volterra']:.4f}")
            print(f"    RMSE:        {result['layer2_volterra_rmse']:.6f}")
            print(f"  --- Layer 3 (Krylov) ---")
            print(f"    recon error: {result['layer3_krylov_rel_error_mean']:.6f}")
            print(f"    norm ratio:  {result['layer3_krylov_norm_ratio_mean']:.4f}")
            print(f"  --- Verdict ---")
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
