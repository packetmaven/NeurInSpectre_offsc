#!/usr/bin/env python3
"""
Download CIFAR-10 into a deterministic on-disk location.

This is a convenience helper for strict real-data / artifact-evaluation runs:
the NeurInSpectre CLI expects datasets to exist under `./data/*` by default.

Note: torchvision handles caching and will skip re-downloads when possible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torchvision


def _download(root: Path, *, train: bool) -> None:
    ds = torchvision.datasets.CIFAR10(root=str(root), train=bool(train), download=True)
    split = "train" if train else "test"
    print(f"[CIFAR-10] {split}: {len(ds):,} samples -> {root}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download CIFAR-10 via torchvision.")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("./data/cifar10"),
        help="Output root directory (default: ./data/cifar10)",
    )
    ap.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Which split(s) to download (default: both)",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.split in {"train", "both"}:
        _download(root, train=True)
    if args.split in {"test", "both"}:
        _download(root, train=False)

    print(f"[OK] CIFAR-10 ready under: {root}")


if __name__ == "__main__":
    main()

