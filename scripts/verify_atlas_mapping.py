#!/usr/bin/env python3
"""
Verify that the MITRE ATLAS IDs referenced in `data/atlas_mapping.yaml` exist.

This script is intentionally offline-first: it validates against the vendored
STIX bundle shipped in-repo (`neurinspectre/mitre_atlas/stix-atlas.json`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from neurinspectre.mitre_atlas.coverage import validate_atlas_ids


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/object: {path}")
    return obj


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify MITRE ATLAS IDs in atlas_mapping.yaml")
    ap.add_argument(
        "--mapping",
        default=str(Path("data") / "atlas_mapping.yaml"),
        help="Path to atlas_mapping.yaml (default: data/atlas_mapping.yaml)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any unknown AML.* IDs are found",
    )
    args = ap.parse_args(argv)

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        ap.error(f"mapping file not found: {mapping_path}")

    cfg = _load_yaml(mapping_path)
    mappings = cfg.get("mappings")
    if not isinstance(mappings, list):
        raise ValueError("atlas_mapping.yaml must contain a top-level 'mappings: [...]' list")

    referenced: List[str] = []
    for i, row in enumerate(mappings):
        if not isinstance(row, dict):
            raise ValueError(f"mappings[{i}] must be a mapping/object")
        ids = row.get("atlas_ids")
        if ids is None:
            continue
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValueError(f"mappings[{i}].atlas_ids must be a list[str]")
        referenced.extend(ids)

    res = validate_atlas_ids(referenced)
    print(
        f"MITRE ATLAS catalog: tactics={res.tactic_count} techniques={res.technique_count} | "
        f"referenced={res.referenced_ids_unique} unique ({res.referenced_ids_total} total) | "
        f"unknown={len(res.unknown_ids)}"
    )
    if res.unknown_ids:
        print("Unknown AML.* IDs:")
        for x in res.unknown_ids:
            print(f"- {x}")
        return 2 if args.strict else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

