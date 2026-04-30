import math

import numpy as np

from neurinspectre.statistical.drift_detection_enhanced import (
    PerDimKSDADCvMFisherBHDriftDetector,
    _bh_fdr_adjust,
)
from neurinspectre.statistical.evasion import iterative_evasion_loop


def test_bh_fdr_adjust_basic():
    adj = _bh_fdr_adjust([0.01, 0.02, 0.9])
    exp = [0.03, 0.03, 0.9]
    assert len(adj) == len(exp)
    assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(adj, exp))


def test_per_dim_ks_ad_cvm_fisher_bh_detector_detects_drift_and_reports_top_features():
    # Deterministic, well-separated distributions (avoid flakiness).
    ref = np.linspace(0.0, 1.0, 200, dtype=np.float64)
    cur = np.linspace(2.0, 3.0, 200, dtype=np.float64)
    reference = np.stack([ref, ref**2], axis=1)
    current = np.stack([cur, cur**2], axis=1)

    det = PerDimKSDADCvMFisherBHDriftDetector(confidence_level=0.95, report_top_k=5)
    res = det.detect_drift(reference, current)

    assert res.drift_detected is True
    assert 0.0 <= float(res.p_value) <= 1.0

    sig = res.statistical_significance
    assert sig.get("aggregation") == "fisher"
    assert sig.get("correction") == "bh_fdr"
    assert sig.get("sample_sizes") == (200, 200)
    assert sig.get("n_features") == 2
    per_dim = sig.get("per_dimension")
    assert isinstance(per_dim, dict)
    assert isinstance(per_dim.get("top_features"), list)
    assert per_dim.get("top_k") == 2  # min(report_top_k, p)


def test_iterative_evasion_loop_reduces_detected_drift():
    ref = np.linspace(0.0, 1.0, 200, dtype=np.float64)
    cur = np.linspace(2.0, 3.0, 200, dtype=np.float64)
    reference = np.stack([ref, ref**2], axis=1)
    current = np.stack([cur, cur**2], axis=1)

    det = PerDimKSDADCvMFisherBHDriftDetector(confidence_level=0.95, report_top_k=10)
    res0 = det.detect_drift(reference, current)
    assert res0.drift_detected is True

    cur_evasive, history = iterative_evasion_loop(
        reference,
        current,
        detector=det,
        max_iters=3,
        top_k_dims=2,       # adjust all dims here
        step_fraction=1.0,  # strong evasion path (copy ref columns when shapes match)
        seed=0,
    )
    assert isinstance(history, list)
    assert len(history) >= 1

    res1 = det.detect_drift(reference, cur_evasive)
    assert res1.drift_detected is False
    assert float(res1.p_value) >= 0.2

