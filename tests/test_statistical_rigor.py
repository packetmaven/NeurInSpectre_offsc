import math

import numpy as np

from neurinspectre.cli import evaluate_cmd as _evaluate_cmd
from neurinspectre.statistical.drift_detection_enhanced import (
    KSDADCvMDriftDetector,
    _bonferroni_adjust,
)


def test_bonferroni_adjust_basic():
    adj = _bonferroni_adjust([0.01, 0.2, 1.0], m=3)
    exp = [0.03, 0.6, 1.0]
    assert len(adj) == len(exp)
    assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(adj, exp))


def test_ks_ad_cvm_detector_reports_sample_sizes_and_bonferroni():
    # Deterministic, well-separated distributions (avoid flakiness).
    ref = np.linspace(0.0, 1.0, 200, dtype=np.float64)
    cur = np.linspace(2.0, 3.0, 200, dtype=np.float64)
    reference = np.stack([ref, ref**2], axis=1)
    current = np.stack([cur, cur**2], axis=1)

    det = KSDADCvMDriftDetector(confidence_level=0.95, projection="pca1")
    res = det.detect_drift(reference, current)

    assert res.drift_detected is True
    assert 0.0 <= float(res.p_value) <= 1.0

    sig = res.statistical_significance
    assert sig.get("sample_sizes") == (200, 200)
    assert sig.get("bonferroni_m") == 3

    tests = sig.get("tests")
    assert isinstance(tests, dict)
    for name in ("ks", "ad", "cvm"):
        assert name in tests
        assert "p_value" in tests[name]
        assert "p_value_bonferroni" in tests[name]


def test_evaluate_seed_resolution_priority():
    cfg = {"seed": 7, "seeds": [1, 2]}
    assert _evaluate_cmd._resolve_seed_list(cfg, cli_seeds=(), num_seeds=5) == [1, 2]
    assert _evaluate_cmd._resolve_seed_list(cfg, cli_seeds=(9, 10), num_seeds=5) == [9, 10]
    assert _evaluate_cmd._resolve_seed_list({"seed": 3}, cli_seeds=(), num_seeds=3) == [3, 4, 5]


def test_aggregate_results_across_seeds_mean_std():
    s0 = {
        "results": [
            {
                "defense": "jpeg",
                "type": "jpeg",
                "dataset": "cifar10",
                "attacks": {
                    "pgd": {
                        "attack_success_rate": 0.1,
                        "robust_accuracy": 0.9,
                        "clean_accuracy": 0.95,
                        "correct_samples": 100,
                        "samples": 100,
                    }
                },
            }
        ]
    }
    s1 = {
        "results": [
            {
                "defense": "jpeg",
                "type": "jpeg",
                "dataset": "cifar10",
                "attacks": {
                    "pgd": {
                        "attack_success_rate": 0.3,
                        "robust_accuracy": 0.7,
                        "clean_accuracy": 0.90,
                        "correct_samples": 90,
                        "samples": 100,
                    }
                },
            }
        ]
    }

    agg = _evaluate_cmd._aggregate_results_across_seeds([s0, s1], seeds=[0, 1])
    assert len(agg) == 1
    metrics = agg[0]["attacks"]["pgd"]

    assert math.isclose(float(metrics["attack_success_rate"]), 0.2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(metrics["attack_success_rate_std"]), math.sqrt(0.02), rel_tol=1e-12, abs_tol=0.0)
    assert int(metrics["attack_success_rate_n"]) == 2
    assert metrics["attack_success_rate_by_seed"] == {"0": 0.1, "1": 0.3}

