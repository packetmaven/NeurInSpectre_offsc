#!/usr/bin/env python3
"""
Leave-One-Algorithm-Out Cross-Validation for RL-Obfuscation Detection (Issue 9).

Motivation:
  RL-obfuscation detectors can look artificially strong if they are calibrated and
  evaluated on gradients produced by the *same* RL training algorithm. This script
  enforces a leave-one-algorithm-out (LOAO) protocol:

    Train on PPO + SAC + TD3  -> Test on A2C
    Train on A2C + SAC + TD3  -> Test on PPO
    Train on A2C + PPO + TD3  -> Test on SAC
    Train on A2C + PPO + SAC  -> Test on TD3

What this script does:
  1) Loads gradient artifacts from a dataset directory (real data expected).
  2) Extracts an 8-D feature vector using NeurInSpectre's
     CriticalRLObfuscationDetector component scores.
  3) Trains a calibration model on 3 algorithms (logistic regression on features)
     and evaluates on the held-out algorithm.
  4) Calibrates an overall-threat threshold on the training algorithms only and
     tests on the held-out algorithm (simple but defensible baseline).
  5) Writes a JSON report with per-fold sample sizes and mean ± std across folds.

Dataset layout (default):
  <input_root>/
    a2c/{obfuscated,clean}/*.npy
    ppo/{obfuscated,clean}/*.npy
    sac/{obfuscated,clean}/*.npy
    td3/{obfuscated,clean}/*.npy

Where:
  - "obfuscated" are RL-obfuscated gradients (positive class y=1)
  - "clean" are benign gradients (negative class y=0)

Strict metadata option (recommended for AE-grade traceability):
  If --strict-metadata is set, each *.npy must have an adjacent metadata JSON:
    - either <file>.npy.meta.json
    - or <file>.meta.json
  with at least:
    {"algorithm": "...", "label": "obfuscated"|"clean", "source": "..."}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _isfinite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _mean_std(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    xs: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        xs.append(fv)
    n = int(len(xs))
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = float(sum(xs) / n)
    if n == 1:
        return {"mean": mean, "std": 0.0, "n": 1}
    var = float(sum((x - mean) ** 2 for x in xs) / float(n - 1))
    return {"mean": mean, "std": float(math.sqrt(max(0.0, var))), "n": n}


@dataclass(frozen=True)
class Sample:
    path: str
    algorithm: str
    label: str  # "obfuscated" or "clean"
    y: int      # 1 for obfuscated, 0 for clean


@dataclass
class FeatureRow:
    sample: Sample
    features: List[float]
    feature_names: List[str]
    overall_threat: float
    detector_confidence: float


FEATURE_NAMES = [
    "policy_fingerprint",
    "semantic_consistency",
    "conditional_triggers",
    "periodic_patterns",
    "evasion_signatures",
    "reward_optimization",
    "training_artifacts",
    "adversarial_patterns",
]


def _load_npy_safely(path: Path) -> np.ndarray:
    arr = np.load(str(path), allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _find_meta_for_npy(path: Path) -> Optional[Path]:
    # Two common conventions:
    #  - grads.npy.meta.json
    #  - grads.meta.json
    cand1 = Path(str(path) + ".meta.json")
    cand2 = path.with_suffix(".meta.json")
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def _validate_meta(
    meta: Dict[str, Any],
    *,
    algorithm: str,
    label: str,
) -> List[str]:
    reasons: List[str] = []
    if not isinstance(meta, dict):
        return ["meta_not_a_dict"]
    if str(meta.get("algorithm", "")).lower() != str(algorithm).lower():
        reasons.append("meta_algorithm_mismatch")
    if str(meta.get("label", "")).lower() != str(label).lower():
        reasons.append("meta_label_mismatch")
    src = str(meta.get("source", "")).strip()
    if not src:
        reasons.append("meta_missing_source")
    return reasons


def _discover_samples(
    input_root: Path,
    *,
    algorithms: Sequence[str],
    pos_dir: str,
    neg_dir: str,
    pattern: str,
    strict_metadata: bool,
) -> List[Sample]:
    samples: List[Sample] = []

    for alg in algorithms:
        alg_dir = input_root / str(alg)
        pos_path = alg_dir / pos_dir
        neg_path = alg_dir / neg_dir
        if not alg_dir.exists():
            raise SystemExit(f"[loocv] Missing algorithm directory: {alg_dir}")
        if not pos_path.exists():
            raise SystemExit(f"[loocv] Missing positive directory: {pos_path}")
        if not neg_path.exists():
            raise SystemExit(f"[loocv] Missing negative directory: {neg_path}")

        pos_files = sorted(pos_path.glob(pattern))
        neg_files = sorted(neg_path.glob(pattern))
        if not pos_files:
            raise SystemExit(f"[loocv] No positive samples for {alg}: {pos_path}/{pattern}")
        if not neg_files:
            raise SystemExit(f"[loocv] No negative samples for {alg}: {neg_path}/{pattern}")

        for p in pos_files:
            if strict_metadata:
                meta_path = _find_meta_for_npy(p)
                if meta_path is None:
                    raise SystemExit(f"[loocv] strict-metadata: missing meta for {p}")
                meta = _read_json(meta_path)
                reasons = _validate_meta(meta, algorithm=alg, label="obfuscated")
                if reasons:
                    raise SystemExit(f"[loocv] strict-metadata: invalid meta for {p}: {','.join(reasons)}")
            samples.append(Sample(path=str(p.resolve()), algorithm=str(alg).lower(), label="obfuscated", y=1))

        for p in neg_files:
            if strict_metadata:
                meta_path = _find_meta_for_npy(p)
                if meta_path is None:
                    raise SystemExit(f"[loocv] strict-metadata: missing meta for {p}")
                meta = _read_json(meta_path)
                reasons = _validate_meta(meta, algorithm=alg, label="clean")
                if reasons:
                    raise SystemExit(f"[loocv] strict-metadata: invalid meta for {p}: {','.join(reasons)}")
            samples.append(Sample(path=str(p.resolve()), algorithm=str(alg).lower(), label="clean", y=0))

    return samples


def _extract_features(
    samples: Sequence[Sample],
    *,
    sensitivity: str,
    cache_path: Optional[Path],
) -> List[FeatureRow]:
    try:
        from neurinspectre.security.critical_rl_obfuscation import CriticalRLObfuscationDetector
    except Exception as exc:
        raise SystemExit(
            "[loocv] Failed to import CriticalRLObfuscationDetector. "
            "Run from repo root with venv activated. Error: " + str(exc)
        ) from exc

    cache: Dict[str, Any] = {}
    if cache_path and cache_path.exists():
        try:
            cache = _read_json(cache_path)
        except Exception:
            cache = {}

    detector = CriticalRLObfuscationDetector(sensitivity_level=str(sensitivity))
    rows: List[FeatureRow] = []

    for s in samples:
        p = Path(s.path)
        key = str(p)
        st = p.stat()
        mtime = float(st.st_mtime)
        size = int(st.st_size)

        cached = cache.get(key) if isinstance(cache, dict) else None
        if isinstance(cached, dict) and cached.get("mtime") == mtime and cached.get("size") == size:
            feats = cached.get("features")
            if isinstance(feats, list) and len(feats) == len(FEATURE_NAMES):
                rows.append(
                    FeatureRow(
                        sample=s,
                        features=[float(x) for x in feats],
                        feature_names=list(FEATURE_NAMES),
                        overall_threat=float(cached.get("overall_threat", 0.0)),
                        detector_confidence=float(cached.get("detector_confidence", 0.0)),
                    )
                )
                continue

        arr = _load_npy_safely(p)
        res = detector.detect_rl_obfuscation(arr, metadata=None)
        comp = dict(res.get("component_scores", {}) or {})
        feats_out = [float(comp.get(name, 0.0)) for name in FEATURE_NAMES]
        overall = float(res.get("overall_threat_level", 0.0))
        conf = float(res.get("detection_confidence", 0.0))

        rows.append(
            FeatureRow(
                sample=s,
                features=feats_out,
                feature_names=list(FEATURE_NAMES),
                overall_threat=overall,
                detector_confidence=conf,
            )
        )

        if isinstance(cache, dict) and cache_path is not None:
            cache[key] = {
                "mtime": mtime,
                "size": size,
                "features": feats_out,
                "overall_threat": overall,
                "detector_confidence": conf,
            }

    if isinstance(cache, dict) and cache_path is not None:
        _write_json(cache_path, cache)

    return rows


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, Any]:
    y_true = y_true.astype(int).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    y_pred = (y_score >= float(threshold)).astype(int)

    counts = _confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    n = int(y_true.size)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    acc = float((tp + tn) / n) if n else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) else 0.0
    bal_acc = float(0.5 * (tpr + tnr)) if (n_pos and n_neg) else acc
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    f1 = float((2.0 * prec * tpr) / (prec + tpr)) if (prec + tpr) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0

    roc_auc = None
    ap = None
    try:
        # Only defined when both classes are present.
        if n_pos > 0 and n_neg > 0:
            from sklearn.metrics import average_precision_score, roc_auc_score

            roc_auc = float(roc_auc_score(y_true, y_score))
            ap = float(average_precision_score(y_true, y_score))
    except Exception:
        roc_auc, ap = None, None

    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": float(threshold),
        "confusion": counts,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": tpr,
        "f1": f1,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "avg_precision": ap,
        "score_mean": float(np.mean(y_score)) if y_score.size else 0.0,
        "score_std": float(np.std(y_score, ddof=0)) if y_score.size else 0.0,
        "score_mean_pos": float(np.mean(y_score[y_true == 1])) if n_pos else None,
        "score_mean_neg": float(np.mean(y_score[y_true == 0])) if n_neg else None,
    }


def _choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    strategy: str,
    fixed_threshold: float,
    target_fpr: float,
) -> float:
    """
    Choose a threshold using TRAINING data only.

    - fixed: uses fixed_threshold
    - max_f1: maximizes F1 on the training split
    - target_fpr: picks the smallest threshold that achieves FPR <= target_fpr
      (ties broken by higher recall)
    """
    strategy = str(strategy).lower()
    if strategy == "fixed":
        return float(fixed_threshold)

    y_true = y_true.astype(int).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    if y_true.size == 0:
        return float(fixed_threshold)

    # Candidate thresholds: unique scores (plus a tiny epsilon for edge behavior).
    uniq = np.unique(y_score)
    if uniq.size == 0:
        return float(fixed_threshold)
    cands = np.sort(uniq)

    best_t = float(fixed_threshold)
    best_val = -1.0
    best_recall = -1.0

    for t in cands:
        m = _metrics_from_scores(y_true, y_score, threshold=float(t))
        if strategy == "max_f1":
            val = float(m.get("f1", 0.0))
            if val > best_val:
                best_val = val
                best_t = float(t)
            continue

        if strategy == "target_fpr":
            fpr = float(m.get("fpr", 1.0))
            recall = float(m.get("recall", 0.0))
            if fpr <= float(target_fpr) + 1e-12:
                # Prefer higher recall; break ties by higher threshold (more conservative).
                if recall > best_recall or (abs(recall - best_recall) < 1e-12 and float(t) > best_t):
                    best_recall = recall
                    best_t = float(t)
            continue

    return float(best_t)


def _run_fold(
    rows: Sequence[FeatureRow],
    *,
    algorithms_train: Sequence[str],
    algorithm_test: str,
    seed: int,
    threshold_strategy: str,
    fixed_threshold: float,
    target_fpr: float,
) -> Dict[str, Any]:
    # Build arrays.
    train = [r for r in rows if r.sample.algorithm in set(algorithms_train)]
    test = [r for r in rows if r.sample.algorithm == str(algorithm_test).lower()]
    if not train or not test:
        raise SystemExit(f"[loocv] Empty train/test split. train={len(train)} test={len(test)}")

    X_train = np.asarray([r.features for r in train], dtype=np.float64)
    y_train = np.asarray([r.sample.y for r in train], dtype=int)
    X_test = np.asarray([r.features for r in test], dtype=np.float64)
    y_test = np.asarray([r.sample.y for r in test], dtype=int)

    # Baseline score: overall_threat_level from the detector.
    s_train = np.asarray([r.overall_threat for r in train], dtype=np.float64)
    s_test = np.asarray([r.overall_threat for r in test], dtype=np.float64)

    # 1) Calibrate a baseline threshold on training algorithms only.
    base_thr = _choose_threshold(
        y_train,
        s_train,
        strategy=threshold_strategy,
        fixed_threshold=fixed_threshold,
        target_fpr=target_fpr,
    )
    baseline_train = _metrics_from_scores(y_train, s_train, threshold=base_thr)
    baseline_test = _metrics_from_scores(y_test, s_test, threshold=base_thr)

    # 2) Train a lightweight calibrated model on component scores.
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise SystemExit("[loocv] scikit-learn required for LOOCV training. Error: " + str(exc)) from exc

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logreg",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    p_train = clf.predict_proba(X_train)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]

    model_thr = _choose_threshold(
        y_train,
        p_train,
        strategy=threshold_strategy,
        fixed_threshold=0.5,
        target_fpr=target_fpr,
    )
    model_train = _metrics_from_scores(y_train, p_train, threshold=model_thr)
    model_test = _metrics_from_scores(y_test, p_test, threshold=model_thr)

    # Coefficients for interpretability.
    coef = None
    intercept = None
    try:
        lr = clf.named_steps["logreg"]
        coef = [float(x) for x in lr.coef_.reshape(-1).tolist()]
        intercept = float(lr.intercept_.reshape(-1)[0])
    except Exception:
        coef, intercept = None, None

    # Sample sizes (explicitly logged as requested).
    split_counts = {
        "train": {
            "n": int(y_train.size),
            "n_pos": int(np.sum(y_train == 1)),
            "n_neg": int(np.sum(y_train == 0)),
        },
        "test": {
            "n": int(y_test.size),
            "n_pos": int(np.sum(y_test == 1)),
            "n_neg": int(np.sum(y_test == 0)),
        },
    }

    return {
        "train_algorithms": list(algorithms_train),
        "test_algorithm": str(algorithm_test).lower(),
        "sample_sizes": split_counts,
        "feature_names": list(FEATURE_NAMES),
        "threshold_strategy": str(threshold_strategy),
        "baseline": {
            "score": "overall_threat_level",
            "trained_threshold": float(base_thr),
            "train_metrics": baseline_train,
            "test_metrics": baseline_test,
        },
        "model": {
            "type": "logistic_regression",
            "trained_threshold": float(model_thr),
            "train_metrics": model_train,
            "test_metrics": model_test,
            "coef": coef,
            "intercept": intercept,
        },
    }


def _folds(algorithms: Sequence[str]) -> List[Tuple[List[str], str]]:
    algs = [str(a).lower() for a in algorithms]
    out: List[Tuple[List[str], str]] = []
    for test_alg in algs:
        train_algs = [a for a in algs if a != test_alg]
        out.append((train_algs, test_alg))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="NeurInSpectre RL-detection leave-one-algorithm-out CV")
    p.add_argument("--input-root", required=True, help="Root directory containing algorithm subdirs")
    p.add_argument("--output-dir", default="_cli_runs/rl_detection_loocv", help="Directory to write results")
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["a2c", "ppo", "sac", "td3"],
        help="Algorithm directory names (default: a2c ppo sac td3)",
    )
    p.add_argument("--pos-dir", default="obfuscated", help="Positive subdir name under each algorithm (default: obfuscated)")
    p.add_argument("--neg-dir", default="clean", help="Negative subdir name under each algorithm (default: clean)")
    p.add_argument("--pattern", default="*.npy", help="Glob pattern for gradient artifacts (default: *.npy)")
    p.add_argument("--sensitivity", default="high", choices=["critical", "high", "medium"], help="Detector sensitivity")
    p.add_argument("--seed", type=int, default=42, help="Training seed (default: 42)")
    p.add_argument(
        "--threshold-strategy",
        default="target_fpr",
        choices=["fixed", "max_f1", "target_fpr"],
        help="How to choose the classification threshold on training algorithms",
    )
    p.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.6,
        help="Used when --threshold-strategy=fixed (default: 0.6)",
    )
    p.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="Used when --threshold-strategy=target_fpr (default: 0.05)",
    )
    p.add_argument(
        "--strict-metadata",
        action="store_true",
        help="Require per-sample metadata JSON to prevent ambiguous provenance",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable feature cache (default: cache enabled)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    input_root = Path(str(args.input_root)).expanduser().resolve()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    _safe_mkdir(output_dir)

    algorithms = [str(a).lower() for a in (args.algorithms or [])]
    if len(algorithms) != 4:
        # LOAO is defined over the 4 algorithm family in the issue.
        raise SystemExit("[loocv] Expected 4 algorithms (A2C/PPO/SAC/TD3); got: " + " ".join(algorithms))

    cache_path = None if bool(args.no_cache) else (output_dir / "feature_cache.json")

    samples = _discover_samples(
        input_root,
        algorithms=algorithms,
        pos_dir=str(args.pos_dir),
        neg_dir=str(args.neg_dir),
        pattern=str(args.pattern),
        strict_metadata=bool(args.strict_metadata),
    )

    rows = _extract_features(samples, sensitivity=str(args.sensitivity), cache_path=cache_path)

    # Basic dataset summary (explicit sample sizes).
    per_alg: Dict[str, Dict[str, int]] = {}
    for r in rows:
        a = r.sample.algorithm
        per_alg.setdefault(a, {"n_pos": 0, "n_neg": 0, "n": 0})
        per_alg[a]["n"] += 1
        if r.sample.y == 1:
            per_alg[a]["n_pos"] += 1
        else:
            per_alg[a]["n_neg"] += 1

    folds = _folds(algorithms)
    fold_results: List[Dict[str, Any]] = []
    for train_algs, test_alg in folds:
        fold_results.append(
            _run_fold(
                rows,
                algorithms_train=train_algs,
                algorithm_test=test_alg,
                seed=int(args.seed),
                threshold_strategy=str(args.threshold_strategy),
                fixed_threshold=float(args.fixed_threshold),
                target_fpr=float(args.target_fpr),
            )
        )

    # Aggregate (mean ± std) across 4 folds.
    def _collect(metric_key: str, *, section: str) -> List[Optional[float]]:
        outv: List[Optional[float]] = []
        for fr in fold_results:
            try:
                v = fr[section]["test_metrics"].get(metric_key)
            except Exception:
                v = None
            outv.append(None if v is None else float(v))
        return outv

    agg = {
        "baseline": {
            "balanced_accuracy": _mean_std(_collect("balanced_accuracy", section="baseline")),
            "roc_auc": _mean_std(_collect("roc_auc", section="baseline")),
            "avg_precision": _mean_std(_collect("avg_precision", section="baseline")),
            "f1": _mean_std(_collect("f1", section="baseline")),
        },
        "model": {
            "balanced_accuracy": _mean_std(_collect("balanced_accuracy", section="model")),
            "roc_auc": _mean_std(_collect("roc_auc", section="model")),
            "avg_precision": _mean_std(_collect("avg_precision", section="model")),
            "f1": _mean_std(_collect("f1", section="model")),
        },
    }

    report = {
        "run": {
            "started_at": _now_iso(),
            "cwd": os.getcwd(),
            "input_root": str(input_root),
            "output_dir": str(output_dir),
            "algorithms": algorithms,
            "pos_dir": str(args.pos_dir),
            "neg_dir": str(args.neg_dir),
            "pattern": str(args.pattern),
            "sensitivity": str(args.sensitivity),
            "seed": int(args.seed),
            "threshold_strategy": str(args.threshold_strategy),
            "fixed_threshold": float(args.fixed_threshold),
            "target_fpr": float(args.target_fpr),
            "strict_metadata": bool(args.strict_metadata),
            "feature_cache": None if cache_path is None else str(cache_path),
        },
        "dataset": {
            "sample_sizes_by_algorithm": per_alg,
            "total_samples": int(len(rows)),
        },
        "folds": fold_results,
        "aggregate_across_folds": agg,
    }

    out_path = output_dir / "rl_detection_loocv.json"
    _write_json(out_path, report)
    print(f"[loocv] Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

