import numpy as np

from neurinspectre.statistical.drift_detection_enhanced import MMDDriftDetector


def test_mmd_detector_no_drift_identical_samples():
    # Deterministic: identical samples -> MMD^2 should be ~0 and drift should not be detected.
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, size=(200, 8)).astype(np.float64)
    cur = ref.copy()

    det = MMDDriftDetector(confidence_level=0.99, n_permutations=200, random_state=0)
    res = det.detect_drift(ref, cur)

    assert res.drift_detected is False
    assert 0.0 <= float(res.p_value) <= 1.0
    assert float(res.p_value) >= 0.5
    assert float(res.drift_score) >= 0.0
    assert res.statistical_significance.get("sample_sizes") == (200, 200)
    assert int(res.statistical_significance.get("n_features")) == 8


def test_mmd_detector_detects_strong_mean_shift():
    # Strong multivariate mean shift should be detected reliably.
    rng = np.random.default_rng(123)
    ref = rng.normal(0.0, 1.0, size=(250, 10)).astype(np.float64)
    cur = rng.normal(2.5, 1.0, size=(250, 10)).astype(np.float64)

    det = MMDDriftDetector(confidence_level=0.95, n_permutations=250, random_state=1)
    res = det.detect_drift(ref, cur)

    assert res.drift_detected is True
    assert 0.0 <= float(res.p_value) <= 1.0
    assert float(res.p_value) < 0.05
    assert float(res.drift_score) > 0.0
    assert res.statistical_significance.get("sample_sizes") == (250, 250)
    assert int(res.statistical_significance.get("n_features")) == 10

