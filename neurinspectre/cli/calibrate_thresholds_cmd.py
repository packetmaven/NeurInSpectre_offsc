"""
Tier 2: ROC/AUC threshold calibration CLI.

Consumes labeled characterization reports (positive/negative sets) and emits a
JSON file with calibrated thresholds at a chosen operating point.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import click
import numpy as np

from ..statistical.roc_calibration import calibrate_threshold
from .utils import save_json


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    """
    Load either:
    - a single JSON object,
    - a JSON list of objects,
    - or a JSONL file (one object per line).
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".jsonl", ".jsonlines"}:
        out: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        return out

    obj = json.loads(text)
    if isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _expand_file_args(raw_values: Iterable[str], *, flag_name: str) -> List[Path]:
    """
    Expand file arguments that may include globs.

    Some shells (notably zsh with `nonomatch`) pass unmatched glob patterns
    through to the program unchanged. Click's ``Path(exists=True)`` then treats
    them as literal paths and fails with a confusing error.
    """
    out: List[Path] = []
    for raw in raw_values:
        s = os.path.expandvars(os.path.expanduser(str(raw)))
        p = Path(s)
        if p.exists() and p.is_file():
            out.append(p)
            continue

        matches = [Path(m) for m in glob.glob(s)]
        matches = [m for m in matches if m.exists() and m.is_file()]
        if matches:
            out.extend(sorted(matches))
            continue

        raise click.ClickException(
            f"{flag_name} did not match any files: {raw!r}. "
            "Tip: pass explicit files (repeat the flag) or ensure the glob matches at least one file."
        )

    # De-dup while preserving order.
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _unwrap_characterization(obj: Dict[str, Any]) -> Dict[str, Any]:
    # exporters.py wraps as {"type":"characterization","report":{...}}
    if isinstance(obj.get("report"), dict):
        return obj["report"]
    return obj


def _extract_field(obj: Dict[str, Any], field: str) -> Optional[float]:
    cur: Any = obj
    for part in str(field).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    try:
        return float(cur)
    except Exception:
        return None


@click.command("calibrate-thresholds")
@click.option(
    "--positive",
    "positive_paths",
    multiple=True,
    type=click.Path(exists=False, dir_okay=False),
    required=True,
    help="Characterization JSON/JSONL files for the positive class (obfuscated/defended)",
)
@click.option(
    "--negative",
    "negative_paths",
    multiple=True,
    type=click.Path(exists=False, dir_okay=False),
    required=True,
    help="Characterization JSON/JSONL files for the negative class (clean/control)",
)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    type=str,
    default=("etd_score", "alpha_volterra"),
    show_default=True,
    help="Metric field(s) to calibrate (dot-paths allowed, e.g. 'report.etd_score')",
)
@click.option(
    "--less-is-positive",
    "less_is_positive",
    multiple=True,
    type=str,
    default=("alpha_volterra", "jacobian_rank"),
    show_default=True,
    help="Metric(s) where smaller values indicate the positive class",
)
@click.option("--target-fpr", type=float, default=0.05, show_default=True)
@click.option("--target-tpr", type=float, default=None, show_default=False)
@click.option("--include-curve", is_flag=True, help="Include full ROC arrays in output JSON")
@click.option("--output", type=click.Path(dir_okay=False), default="results/calibrated_thresholds.json")
def calibrate_thresholds_cmd(**kwargs: Any) -> None:
    """
    Calibrate decision thresholds using ROC/AUC.

    Example:
        neurinspectre calibrate-thresholds \\
          --positive results/char/defended_*.json \\
          --negative results/char/clean_*.json \\
          --metric etd_score --metric alpha_volterra \\
          --target-fpr 0.05 \\
          --output results/thresholds.json
    """
    pos_paths = _expand_file_args((kwargs.get("positive_paths") or ()), flag_name="--positive")
    neg_paths = _expand_file_args((kwargs.get("negative_paths") or ()), flag_name="--negative")
    metrics = [str(m) for m in (kwargs.get("metrics") or ())]
    less_pos = {str(m) for m in (kwargs.get("less_is_positive") or ())}
    target_fpr = kwargs.get("target_fpr")
    target_tpr = kwargs.get("target_tpr")
    include_curve = bool(kwargs.get("include_curve", False))

    pos_records: List[Dict[str, Any]] = []
    neg_records: List[Dict[str, Any]] = []
    for p in pos_paths:
        pos_records.extend(_load_json_records(p))
    for p in neg_paths:
        neg_records.extend(_load_json_records(p))

    pos_reports = [_unwrap_characterization(r) for r in pos_records]
    neg_reports = [_unwrap_characterization(r) for r in neg_records]

    if len(pos_reports) == 0 or len(neg_reports) == 0:
        raise click.ClickException("Need at least one positive and one negative report.")

    metric_out: Dict[str, Any] = {}
    warnings_out: List[str] = []

    for metric in metrics:
        pos_scores = []
        neg_scores = []
        for r in pos_reports:
            v = _extract_field(r, metric)
            if v is not None:
                pos_scores.append(v)
        for r in neg_reports:
            v = _extract_field(r, metric)
            if v is not None:
                neg_scores.append(v)

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            warnings_out.append(
                f"metric {metric!r} missing in at least one class; skipping (pos={len(pos_scores)}, neg={len(neg_scores)})"
            )
            continue

        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores), dtype=int)
        scores = np.array(pos_scores + neg_scores, dtype=float)
        greater_is_positive = metric not in less_pos

        cal = calibrate_threshold(
            metric=metric,
            y_true=y_true,
            scores=scores,
            greater_is_positive=greater_is_positive,
            target_fpr=float(target_fpr) if target_fpr is not None else None,
            target_tpr=None if target_tpr is None else float(target_tpr),
            include_curve=include_curve,
        )
        metric_out[metric] = cal.to_dict()

    # Best-effort mapping from calibrated metric thresholds -> DefenseAnalyzer overrides.
    # This is intentionally conservative: we only map when the calibrated decision
    # rule matches the semantic direction the DefenseAnalyzer expects.
    metric_to_analyzer_key = {
        "etd_score": ("ETD_THRESHOLD_SEVERE", "score>=threshold"),
        "spectral_entropy_norm": ("SPECTRAL_ENTROPY_OBFUSCATED_THRESHOLD", "score>=threshold"),
        "high_freq_ratio": ("HIGH_FREQ_RATIO_SHATTERED_THRESHOLD", "score>=threshold"),
        "alpha_volterra": ("ALPHA_RL_THRESHOLD", "score<=threshold"),
        "jacobian_rank": ("RANK_VANISHING_THRESHOLD", "score<=threshold"),
        "gradient_variance": ("VARIANCE_STOCHASTIC_THRESHOLD", "score>=threshold"),
    }
    analyzer_overrides: Dict[str, float] = {}
    analyzer_mapping_notes: List[str] = []
    for metric, cal_dict in metric_out.items():
        base_metric = str(metric).split(".")[-1]
        mapping = metric_to_analyzer_key.get(base_metric)
        if mapping is None:
            continue
        analyzer_key, expected_rule = mapping
        rule = str(cal_dict.get("rule", ""))
        thr = cal_dict.get("threshold")
        if thr is None:
            continue
        if expected_rule and rule != expected_rule:
            warnings_out.append(
                f"metric {metric!r} calibrated with rule {rule!r}, expected {expected_rule!r} "
                f"for mapping to {analyzer_key}; override not generated"
            )
            continue
        try:
            analyzer_overrides[analyzer_key] = float(thr)
            analyzer_mapping_notes.append(f"{analyzer_key} <- {metric} ({rule}, threshold={float(thr):.6g})")
        except Exception:
            continue

    payload = {
        "type": "threshold_calibration",
        "config": {
            "target_fpr": None if target_fpr is None else float(target_fpr),
            "target_tpr": None if target_tpr is None else float(target_tpr),
            "metrics": list(metrics),
            "less_is_positive": sorted(less_pos),
            "n_positive_files": int(len(pos_paths)),
            "n_negative_files": int(len(neg_paths)),
        },
        "metric_calibration": metric_out,
        "defense_analyzer_threshold_overrides": analyzer_overrides,
    }
    if warnings_out:
        payload["warnings"] = list(warnings_out)
    if analyzer_mapping_notes:
        payload["notes"] = list(analyzer_mapping_notes)

    out_path = Path(str(kwargs["output"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, out_path)
    click.echo(str(out_path))

