import json
import math
from pathlib import Path

import numpy as np

from neurinspectre.security.critical_rl_obfuscation import CriticalRLObfuscationDetector


def _sigmoid(x: float) -> float:
    xf = float(x)
    if xf >= 0.0:
        z = math.exp(-xf)
        return 1.0 / (1.0 + z)
    z = math.exp(xf)
    return z / (1.0 + z)


def test_rl_obfuscation_srl_logreg_zero_weights(tmp_path: Path):
    # If coef==0 and intercept==0, SRL must be sigmoid(0)=0.5 regardless of component scores.
    weights_path = tmp_path / "srl_weights_zero.json"
    weights_path.write_text(
        json.dumps(
            {
                "feature_names": [
                    "policy_fingerprint",
                    "semantic_consistency",
                    "conditional_triggers",
                    "periodic_patterns",
                    "evasion_signatures",
                    "reward_optimization",
                    "training_artifacts",
                    "adversarial_patterns",
                ],
                "coef": [0.0] * 8,
                "intercept": 0.0,
                "threshold": 0.6,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    det = CriticalRLObfuscationDetector(sensitivity_level="high", srl_weights_path=str(weights_path))
    g = np.random.default_rng(0).normal(size=(256,)).astype(np.float32)
    out = det.detect_rl_obfuscation(g)

    assert "srl" in out and isinstance(out["srl"], dict)
    assert abs(float(out["srl"]["score"]) - 0.5) < 1e-6
    assert out["srl"]["passed"] is False
    # When SRL weights are loaded, SRL is used as the overall threat scalar.
    assert abs(float(out["overall_threat_level"]) - 0.5) < 1e-6


def test_rl_obfuscation_srl_named_weight_map(tmp_path: Path):
    # Named-weight map format: SRL = sigmoid(bias + w * policy_fingerprint).
    weights_path = tmp_path / "srl_weights_policy_only.json"
    weights_path.write_text(
        json.dumps(
            {
                "weights": {"policy_fingerprint": 1.0},
                "bias": 0.0,
                "threshold": 0.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    det = CriticalRLObfuscationDetector(sensitivity_level="high", srl_weights_path=str(weights_path))
    g = np.random.default_rng(1).normal(size=(128,)).astype(np.float32)
    out = det.detect_rl_obfuscation(g)

    policy = float((out.get("component_scores") or {}).get("policy_fingerprint", 0.0) or 0.0)
    expected = _sigmoid(policy)
    assert abs(float(out["srl"]["score"]) - expected) < 1e-6
