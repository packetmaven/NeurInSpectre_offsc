"""Baseline validation helpers for evaluation pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import json

import yaml


def normalize_defense_key(defense: str) -> str:
    """Normalize defense key for robust comparison."""
    return str(defense).lower().replace("_", "").replace("-", "").replace(" ", "")


def normalize_attack_key(attack: str) -> str:
    """Normalize attack key for robust comparison."""
    return str(attack).lower().replace("_", "").replace("-", "").replace(" ", "")


def _normalize_expected_map(raw: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for defense, attack_map in (raw or {}).items():
        if not isinstance(attack_map, dict):
            continue
        dkey = normalize_defense_key(defense)
        normalized[dkey] = {}
        for attack, value in attack_map.items():
            try:
                normalized[dkey][normalize_attack_key(attack)] = float(value)
            except (TypeError, ValueError):
                continue
    return normalized


def load_expected_asr(
    validation_cfg: Dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load expected ASR map from an external path.

    Supported config:
      baseline_validation:
        expected_asr_path: ./path/to/baseline.yaml
    """
    validation_cfg = dict(validation_cfg or {})
    if "expected_asr" in validation_cfg:
        raise ValueError(
            "Inline baseline_validation.expected_asr is disabled by policy. "
            "Provide an external file via baseline_validation.expected_asr_path instead."
        )

    path = validation_cfg.get("expected_asr_path")
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    if not p.exists():
        raise FileNotFoundError(f"Baseline expected_asr_path not found: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("Baseline file must contain a mapping defense -> attacks.")
    return _normalize_expected_map(raw)


def validate_asr_matrix(
    observed: Dict[str, Dict[str, float]],
    expected: Dict[str, Dict[str, float]],
    *,
    tolerance: float,
    require_all_expected: bool = True,
) -> Dict[str, Any]:
    """
    Validate observed ASR against expected values with tolerance.

    Returns a structured report, never raises.
    """
    rows: List[Dict[str, Any]] = []
    missing_expected: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str, float]] = []

    for defense, attacks in observed.items():
        dkey = normalize_defense_key(defense)
        exp_attacks = expected.get(dkey, {})
        for attack, obs_val in (attacks or {}).items():
            akey = normalize_attack_key(attack)
            if akey not in exp_attacks:
                missing_expected.append((defense, attack))
                rows.append(
                    {
                        "defense": defense,
                        "attack": attack,
                        "observed_asr": float(obs_val),
                        "expected_asr": None,
                        "delta": None,
                        "within_tolerance": None,
                    }
                )
                continue
            exp_val = float(exp_attacks[akey])
            delta = float(obs_val) - exp_val
            within = abs(delta) <= float(tolerance)
            rows.append(
                {
                    "defense": defense,
                    "attack": attack,
                    "observed_asr": float(obs_val),
                    "expected_asr": exp_val,
                    "delta": delta,
                    "within_tolerance": bool(within),
                }
            )
            if not within:
                failed.append((defense, attack, delta))

    passed = len(failed) == 0 and (
        (not require_all_expected) or (len(missing_expected) == 0)
    )
    return {
        "passed": bool(passed),
        "tolerance": float(tolerance),
        "require_all_expected": bool(require_all_expected),
        "rows": rows,
        "missing_expected_count": len(missing_expected),
        "failed_count": len(failed),
        "missing_expected": [
            {"defense": d, "attack": a} for d, a in missing_expected
        ],
        "failed": [
            {"defense": d, "attack": a, "delta": float(delta)}
            for d, a, delta in failed
        ],
    }


def build_observed_asr_matrix(
    rows: Iterable[Tuple[str, str, float]],
) -> Dict[str, Dict[str, float]]:
    """Build defense -> attack -> ASR matrix."""
    matrix: Dict[str, Dict[str, float]] = {}
    for defense, attack, asr in rows:
        matrix.setdefault(str(defense), {})[str(attack)] = float(asr)
    return matrix
