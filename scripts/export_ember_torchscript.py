#!/usr/bin/env python3
"""
Export an EMBER MLP `state_dict` to a TorchScript artifact.

Why this exists:
- The Table 8 / Table 2 pipeline expects a TorchScript model at `models/ember_mlp_ts.pt`.
- `scripts/train_ember_real.py` trains an MLP and writes a `state_dict` checkpoint.
- Reviewers should be able to go: raw EMBER JSONL -> vectorized memmaps -> trained state_dict -> TorchScript.

This script intentionally matches the architecture defined in `scripts/train_ember_real.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_ember_mlp(*, input_dim: int, num_classes: int) -> nn.Module:
    # Keep architecture identical to `scripts/train_ember_real.py`.
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export EMBER MLP TorchScript + meta.json")
    ap.add_argument(
        "--state-dict",
        type=Path,
        default=Path("models/checkpoints/ember_mlp.pt"),
        help="Path to a PyTorch state_dict checkpoint (default: models/checkpoints/ember_mlp.pt)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/ember_mlp_ts.pt"),
        help="Output TorchScript path (default: models/ember_mlp_ts.pt)",
    )
    ap.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Optional meta.json output path (default: <output>.meta.json)",
    )
    ap.add_argument(
        "--input-dim",
        type=int,
        default=2381,
        help="EMBER feature dimension (default: 2381)",
    )
    ap.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes (default: 2)",
    )
    args = ap.parse_args()

    t0 = time.time()

    if not args.state_dict.exists():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    model = _build_ember_mlp(input_dim=int(args.input_dim), num_classes=int(args.num_classes))
    state = torch.load(str(args.state_dict), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Quick sanity forward pass (also catches obvious architecture mismatch).
    with torch.no_grad():
        y = model(torch.zeros(1, int(args.input_dim), dtype=torch.float32))
        if tuple(y.shape) != (1, int(args.num_classes)):
            raise SystemExit(f"Unexpected output shape: got {tuple(y.shape)}, expected (1, {args.num_classes})")

    # TorchScript export. Prefer scripting; fall back to tracing for robustness.
    try:
        scripted = torch.jit.script(model)
    except Exception as exc:
        print(f"[WARN] torch.jit.script failed ({type(exc).__name__}): {exc}")
        example = torch.zeros(1, int(args.input_dim), dtype=torch.float32)
        scripted = torch.jit.trace(model, example)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))

    sha256 = _sha256_file(args.output)
    elapsed = time.time() - t0

    meta_path = args.meta_output
    if meta_path is None:
        meta_path = args.output.with_name(args.output.name + ".meta.json")

    meta = {
        "dataset": "ember",
        "architecture": "mlp",
        "format": "torchscript",
        "input_dim": int(args.input_dim),
        "num_classes": int(args.num_classes),
        "sha256": sha256,
        "torch_version": str(torch.__version__),
        "elapsed_seconds": float(elapsed),
        "torchscript_path": str(args.output.as_posix()),
        "state_dict_path": str(args.state_dict.as_posix()),
        "notes": (
            "Exported via scripts/export_ember_torchscript.py from a trained state_dict. "
            "Intended for neurinspectre Table 8 / table2 runs."
        ),
    }

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"[OK] Wrote: {args.output} (sha256={sha256})")
    print(f"[OK] Wrote: {meta_path}")


if __name__ == "__main__":
    main()

