"""
Drift detection command implementation (Click wrapper lives in `cli/main.py`).

Consumes real `.npy`/`.npz` arrays and emits JSON (optionally a PNG plot). This
intentionally avoids shipping any baseline/reference numbers in-repo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..statistical.drift_detection_enhanced import create_enhanced_drift_detector

logger = logging.getLogger(__name__)


def _safe_makedirs(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _jsonify(obj: Any) -> Any:
    """Best-effort JSON conversion for numpy containers."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _to_2d_samples(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 0:
        X = np.asarray([X])
    if X.ndim == 1:
        return X.reshape(-1, 1).astype(np.float64, copy=False)
    if X.ndim == 2:
        return X.astype(np.float64, copy=False)
    return X.reshape(int(X.shape[0]), -1).astype(np.float64, copy=False)


def _load_array(path: str) -> np.ndarray:
    """Load a .npy/.npz array-like object and sanitize non-finite numeric values."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".npz":
        npz = np.load(str(p), allow_pickle=True)
        for k in (
            "data",
            "X",
            "x",
            "A",
            "arr",
            "activations",
            "series",
            "reference",
            "current",
            "benign",
            "attack",
        ):
            if k in npz:
                arr = np.asarray(npz[k])
                break
        else:
            if len(npz.files) == 0:
                raise ValueError(f"Empty .npz: {p}")
            arr = np.asarray(npz[npz.files[0]])
    else:
        obj = np.load(str(p), allow_pickle=True)
        if getattr(obj, "dtype", None) is object and getattr(obj, "shape", ()) == ():
            obj = obj.item()
        if isinstance(obj, dict):
            for k in (
                "data",
                "X",
                "x",
                "A",
                "arr",
                "activations",
                "series",
                "reference",
                "current",
                "benign",
                "attack",
            ):
                if k in obj:
                    obj = obj[k]
                    break
        arr = np.asarray(obj)

    if np.issubdtype(arr.dtype, np.number):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(arr)


def _save_drift_plot(
    ref: np.ndarray,
    cur: np.ndarray,
    *,
    consensus: Any,
    per_method: Dict[str, Any],
    out_path: str,
    feature_index: int = 0,
) -> None:
    """Create a concise drift plot (distribution + per-method scores) and save as PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    ref = np.asarray(ref, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    p = int(min(ref.shape[1], cur.shape[1])) if ref.ndim == 2 and cur.ndim == 2 else 1
    fi = int(np.clip(int(feature_index), 0, max(0, p - 1)))

    x_ref = ref[:, fi] if ref.ndim == 2 else ref.reshape(-1)
    x_cur = cur[:, fi] if cur.ndim == 2 else cur.reshape(-1)
    x_ref = x_ref[np.isfinite(x_ref)]
    x_cur = x_cur[np.isfinite(x_cur)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel 1: distribution drift for one feature
    ax = axes[0]
    if x_ref.size and x_cur.size:
        ax.hist(x_ref, bins=40, alpha=0.55, density=True, label="Reference")
        ax.hist(x_cur, bins=40, alpha=0.55, density=True, label="Current")
    ax.set_title(f"Feature {fi} Distribution Drift")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(loc="best", frameon=True)

    # Panel 2: per-method drift scores
    ax2 = axes[1]
    method_names = list(per_method.keys())
    scores = []
    pvals = []
    for m in method_names:
        r = per_method[m]
        scores.append(float(getattr(r, "drift_score", 0.0)))
        pvals.append(float(getattr(r, "p_value", 1.0)))
    ax2.bar(method_names, scores, alpha=0.85)
    ax2.set_title("Per-Method Drift Scores")
    ax2.set_ylabel("Drift score")
    ax2.tick_params(axis="x", rotation=25)

    # Annotate p-values
    for i, (s, pv) in enumerate(zip(scores, pvals)):
        ax2.text(i, s, f"p={pv:.3g}", ha="center", va="bottom", fontsize=9)

    # Suptitle
    detected = bool(getattr(consensus, "drift_detected", False))
    score = float(getattr(consensus, "drift_score", 0.0))
    pval = float(getattr(consensus, "p_value", 1.0))
    fig.suptitle(
        f"NeurInSpectre Drift Detect — {'DRIFT DETECTED' if detected else 'No drift'} "
        f"(score={score:.3f}, p={pval:.3g})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    outp = Path(out_path)
    _safe_makedirs(outp)
    fig.savefig(outp, dpi=200)
    plt.close(fig)


def run_drift_detect(
    *,
    reference: str,
    current: str,
    methods: str,
    confidence_level: float,
    output: str | None,
    plot: str | None,
    plot_feature_index: int,
) -> Tuple[str, Path | None]:
    """
    Run drift detection and return (payload_json, output_path_or_None).
    """
    ref = _to_2d_samples(_load_array(reference))
    cur = _to_2d_samples(_load_array(current))

    # Align feature dimensions (best-effort).
    p = int(min(ref.shape[1], cur.shape[1]))
    ref = ref[:, :p]
    cur = cur[:, :p]

    method_list = [m.strip().lower() for m in str(methods or "").split(",") if m.strip()]
    if not method_list:
        method_list = ["hotelling", "ks", "bayesian"]

    detector = create_enhanced_drift_detector(confidence_level=float(confidence_level), methods=method_list)
    per_method = detector.detect_drift_ensemble(ref, cur)
    consensus = detector.get_consensus_result(per_method)

    payload_obj: Dict[str, Any] = {
        "reference_shape": list(ref.shape),
        "current_shape": list(cur.shape),
        "methods": method_list,
        "consensus": _jsonify(asdict(consensus)),
        "per_method": {k: _jsonify(asdict(v)) for k, v in per_method.items()},
    }
    payload = json.dumps(payload_obj, indent=2)

    out_path = Path(str(output)) if output else None
    if out_path is not None:
        _safe_makedirs(out_path)
        out_path.write_text(payload, encoding="utf-8")

    if plot:
        try:
            _save_drift_plot(
                ref,
                cur,
                consensus=consensus,
                per_method=per_method,
                out_path=str(plot),
                feature_index=int(plot_feature_index),
            )
        except Exception as exc:
            # Plot is optional; do not fail drift detection.
            logger.warning("Drift plot failed: %s", exc)

    return payload, out_path

