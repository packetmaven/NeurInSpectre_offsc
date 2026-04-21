#!/usr/bin/env python3
"""
Download the adversarially-trained Carmon2019Unlabeled WRN-28-10 checkpoint
from RobustBench and place it at the path expected by the CCS detection-paper
real-defense experiments.

Target path:
    models/cifar10/Linf/Carmon2019Unlabeled.pt

This script exists because:
  * The file is ~146 MB and cannot be committed to GitHub
    (100 MB soft limit) or anonymous.4open.science (~100 MB per-file limit).
  * scripts/run_real_defense_experiments.py (Table 5 in the paper) requires
    this exact file.

Usage:
    python scripts/download_carmon2019.py
    python scripts/download_carmon2019.py --dest models/cifar10/Linf
    python scripts/download_carmon2019.py --force    # re-download even if
                                                     # file already exists

On first run, the script will:
  1. Check if the target file already exists (early exit if so).
  2. Install `robustbench` via pip if missing (prints the command first).
  3. Call robustbench.utils.load_model(...) to fetch and cache the weights.
  4. Copy / symlink the cached weights into the expected target path.
  5. Print the SHA256 so you can pin it in your artifact manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_NAME = "Carmon2019Unlabeled"
DATASET = "cifar10"
THREAT_MODEL = "Linf"

# Pinned SHA256 of the Carmon2019Unlabeled Linf WRN-28-10 checkpoint used
# to produce Table 5 in the paper. Reviewers downloading via this script
# will get an automatic integrity check against this hash on completion.
EXPECTED_SHA256: str | None = (
    "f3ea703e4e98d26947bced9580f63922e31423233bbe45eebff8c7d45b7eacfc"
)


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _ensure_robustbench() -> None:
    try:
        import robustbench  # noqa: F401
        return
    except ImportError:
        pass
    cmd = [sys.executable, "-m", "pip", "install", "robustbench"]
    print(f"[deps] robustbench not installed; running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Carmon2019Unlabeled WRN-28-10 for Table 5 reproduction"
    )
    parser.add_argument(
        "--dest",
        default="models/cifar10/Linf",
        help="Directory to place the checkpoint (default: models/cifar10/Linf)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{MODEL_NAME}.pt"

    if target.exists() and not args.force:
        print(f"[ok] Already exists: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        print(f"[ok] SHA256: {_sha256_of(target)}")
        print("[ok] Use --force to re-download.")
        return 0

    _ensure_robustbench()
    from robustbench.utils import load_model

    print(f"[fetch] robustbench.load_model(model={MODEL_NAME}, "
          f"dataset={DATASET}, threat={THREAT_MODEL}) ...")
    load_model(
        model_name=MODEL_NAME,
        dataset=DATASET,
        threat_model=THREAT_MODEL,
        model_dir=str(dest_dir),
    )

    if not target.exists():
        # RobustBench may save under a subdirectory depending on version; locate it.
        candidates = list(dest_dir.rglob(f"{MODEL_NAME}.pt"))
        if not candidates:
            print(f"[error] RobustBench did not place {MODEL_NAME}.pt under "
                  f"{dest_dir}. Found instead:", file=sys.stderr)
            for p in dest_dir.rglob("*"):
                if p.is_file():
                    print(f"  {p}", file=sys.stderr)
            return 1
        src = candidates[0]
        if src != target:
            shutil.copy2(src, target)
            print(f"[ok] Copied {src} -> {target}")

    size_mb = target.stat().st_size / 1e6
    sha = _sha256_of(target)
    print(f"[done] {target} ({size_mb:.1f} MB)")
    print(f"[done] SHA256: {sha}")

    if EXPECTED_SHA256 is not None:
        if sha.lower() != EXPECTED_SHA256.lower():
            print(f"[error] SHA256 mismatch! expected {EXPECTED_SHA256}",
                  file=sys.stderr)
            return 2
        print("[ok] SHA256 matches pinned EXPECTED_SHA256.")
    else:
        print("[note] EXPECTED_SHA256 is not pinned in this script. "
              "Paste the hash above into EXPECTED_SHA256 and re-commit to "
              "give future reviewers an integrity check.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
