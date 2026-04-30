#!/usr/bin/env python3
"""
Download + extract the EMBER 2018 dataset archive.

This script only prepares the *raw* JSONL shards under:
  ./data/ember/ember2018/

After this, generate vectorized memmaps (required by strict real-data runs):
  python scripts/vectorize_ember_safe.py

We use a safe tar extraction routine (no path traversal, no symlinks).
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path) -> None:
    dest_dir = Path(dest_dir).resolve()
    for member in tar.getmembers():
        name = str(member.name)
        if not name or name.startswith("/"):
            raise RuntimeError(f"Refusing to extract unsafe tar member: {name!r}")
        parts = Path(name).parts
        if any(p == ".." for p in parts):
            raise RuntimeError(f"Refusing to extract tar member with '..': {name!r}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing to extract symlink/hardlink from tar: {name!r}")
        target = (dest_dir / name).resolve()
        if dest_dir not in target.parents and target != dest_dir:
            raise RuntimeError(f"Refusing to extract tar member outside destination: {name!r}")
    tar.extractall(path=str(dest_dir))


def _download(url: str, *, out_path: Path, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not bool(force):
        print(f"[EMBER] Archive already exists: {out_path}")
        return

    print(f"[EMBER] Downloading: {url}")
    print(f"[EMBER] To: {out_path}")

    with urllib.request.urlopen(url) as resp:
        with out_path.open("wb") as handle:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    print("[EMBER] Download complete.")


def _check_expected_raw_files(root: Path) -> tuple[bool, list[str]]:
    raw_dir = root / "ember2018"
    expected = [f"train_features_{i}.jsonl" for i in range(6)] + ["test_features.jsonl"]
    missing = [name for name in expected if not (raw_dir / name).exists()]
    return (len(missing) == 0), missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + extract EMBER 2018 raw JSONL shards.")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("./data/ember"),
        help="Output root directory (default: ./data/ember)",
    )
    ap.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"EMBER archive URL (default: {DEFAULT_URL})",
    )
    ap.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional archive path (default: <root>/ember_dataset_2018_2.tar.bz2)",
    )
    ap.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the archive even if it already exists",
    )
    ap.add_argument(
        "--no-extract",
        action="store_true",
        help="Download only; do not extract",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Only check whether expected raw shards exist; do not download/extract",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.check_only:
        ok, missing = _check_expected_raw_files(root)
        if ok:
            print(f"[OK] Raw shards present under: {root / 'ember2018'}")
            raise SystemExit(0)
        print(f"[MISSING] Raw shards not found under: {root / 'ember2018'}")
        print("Missing: " + ", ".join(missing))
        raise SystemExit(2)

    archive = Path(args.archive) if args.archive is not None else (root / "ember_dataset_2018_2.tar.bz2")
    _download(str(args.url), out_path=archive, force=bool(args.force_download))

    if args.no_extract:
        print("[EMBER] Skipping extraction (--no-extract).")
        raise SystemExit(0)

    print(f"[EMBER] Extracting {archive} -> {root}")
    with tarfile.open(str(archive), "r:bz2") as tar:
        _safe_extract_tar(tar, root)
    print("[EMBER] Extraction complete.")

    ok, missing = _check_expected_raw_files(root)
    if not ok:
        print(f"[WARN] Expected raw shards were not found under {root / 'ember2018'}")
        print("Missing: " + ", ".join(missing))
        print("The archive may have a different layout; inspect the extracted directory tree.")
    else:
        print(f"[OK] Raw EMBER2018 shards ready under: {root / 'ember2018'}")
        print(f"Next: python scripts/vectorize_ember_safe.py")


if __name__ == "__main__":
    main()

