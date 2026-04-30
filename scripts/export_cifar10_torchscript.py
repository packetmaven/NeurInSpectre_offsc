#!/usr/bin/env python3
"""
Export a CIFAR-10 model to a TorchScript artifact + metadata.

This is a "from scratch" helper for reviewers:
  - downloads pretrained CIFAR-10 weights (if needed)
  - wraps the model with standard CIFAR-10 input normalization (unless disabled)
  - exports a TorchScript `.pt` consumable by `neurinspectre table2`
  - writes `<model>.meta.json` containing a SHA256 pin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neurinspectre.models.cifar10 import load_cifar10_model


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export CIFAR-10 TorchScript + meta.json")
    ap.add_argument(
        "--arch",
        type=str,
        default="resnet20",
        help="Model architecture key (default: resnet20)",
    )
    ap.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained CIFAR-10 weights (results will be meaningless)",
    )
    ap.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable CIFAR-10 input normalization wrapper",
    )
    ap.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Optional state_dict path override (default: auto-download into models/weights/cifar10/)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/cifar10_resnet20_norm_ts.pt"),
        help="Output TorchScript path (default: models/cifar10_resnet20_norm_ts.pt)",
    )
    ap.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Optional meta.json output path (default: <output>.meta.json)",
    )
    args = ap.parse_args()

    t0 = time.time()

    # Build on CPU for export determinism.
    model = load_cifar10_model(
        model_name=str(args.arch),
        pretrained=not bool(args.no_pretrained),
        device="cpu",
        normalize=not bool(args.no_normalize),
        weights_path=args.weights_path,
    )
    model = model.to("cpu").eval()

    # TorchScript export. Prefer scripting; fall back to tracing for robustness.
    example = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    try:
        scripted = torch.jit.script(model)
    except Exception as exc:
        print(f"[WARN] torch.jit.script failed ({type(exc).__name__}): {exc}")
        scripted = torch.jit.trace(model, example)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))

    sha256 = _sha256_file(out_path)
    elapsed = time.time() - t0

    meta_path = args.meta_output
    if meta_path is None:
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")

    meta = {
        "dataset": "cifar10",
        "architecture": str(args.arch),
        "format": "torchscript",
        "num_classes": 10,
        "pretrained": bool(not args.no_pretrained),
        "normalize": bool(not args.no_normalize),
        "weights_path": str(args.weights_path) if args.weights_path else None,
        "sha256": sha256,
        "torch_version": str(torch.__version__),
        "elapsed_seconds": float(elapsed),
        "torchscript_path": str(out_path.as_posix()),
        "notes": "Exported via scripts/export_cifar10_torchscript.py",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"[OK] Wrote: {out_path} (sha256={sha256})")
    print(f"[OK] Wrote: {meta_path}")


if __name__ == "__main__":
    main()

