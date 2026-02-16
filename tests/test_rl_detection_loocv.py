import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


def _load_loocv_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "rl_detection_loocv.py"
    spec = importlib.util.spec_from_file_location("rl_detection_loocv", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Ensure the module is visible to dataclasses when using
    # `from __future__ import annotations` (string annotations).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_loocv_folds_rotation():
    m = _load_loocv_module()
    folds = m._folds(["a2c", "ppo", "sac", "td3"])
    assert len(folds) == 4
    tests = [t for _tr, t in folds]
    assert sorted(tests) == ["a2c", "ppo", "sac", "td3"]
    for tr, te in folds:
        assert te not in tr
        assert len(tr) == 3


def test_choose_threshold_target_fpr_prefers_recall():
    m = _load_loocv_module()
    # Two negatives with low scores; both positives >= 0.3, so threshold 0.3
    # yields FPR=0 and recall=1.
    y = np.array([0, 0, 1, 1], dtype=int)
    s = np.array([0.10, 0.20, 0.30, 0.90], dtype=float)
    thr = m._choose_threshold(y, s, strategy="target_fpr", fixed_threshold=0.6, target_fpr=0.0)
    assert math.isclose(float(thr), 0.30, rel_tol=0.0, abs_tol=1e-12)
    met = m._metrics_from_scores(y, s, threshold=float(thr))
    assert math.isclose(float(met["fpr"]), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(met["recall"]), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_metrics_from_scores_confusion_counts():
    m = _load_loocv_module()
    y = np.array([0, 0, 1, 1], dtype=int)
    s = np.array([0.2, 0.7, 0.1, 0.9], dtype=float)
    met = m._metrics_from_scores(y, s, threshold=0.5)
    # preds = [0,1,0,1] -> tn=1, fp=1, fn=1, tp=1
    assert met["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_choose_threshold_max_f1_returns_candidate():
    m = _load_loocv_module()
    y = np.array([0, 0, 0, 1, 1], dtype=int)
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.9], dtype=float)
    thr = m._choose_threshold(y, s, strategy="max_f1", fixed_threshold=0.6, target_fpr=0.05)
    # Should be chosen from candidate thresholds (unique scores).
    assert float(thr) in set(float(x) for x in np.unique(s))

