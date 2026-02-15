"""Experimental validity gates for evaluation outputs.

These checks are not "baselines" and do not encode paper numbers. They exist to
prevent *meaningless* robustness reporting when the underlying clean accuracy is
near-zero (e.g., a stub/random checkpoint, a label-map mismatch, or a broken
data pipeline).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def resolve_validity_gates(
    config: Dict[str, Any],
    *,
    default_enabled: bool = False,
    default_strict: bool = False,
) -> Dict[str, Any]:
    raw = dict(config.get("validity_gates", {}) or {})
    enabled = bool(raw.get("enabled", default_enabled))
    strict = bool(raw.get("strict", default_strict))

    # Conservative defaults: catch "0 correct samples" / "near zero" without
    # over-constraining legitimate low-accuracy scenarios (especially on small
    # smoke-test sample counts).
    min_clean_accuracy = float(raw.get("min_clean_accuracy", 0.05))
    min_correct_samples = int(raw.get("min_correct_samples", raw.get("min_clean_correct_samples", 5)))
    min_correct_fraction = float(raw.get("min_correct_fraction", 0.0))
    min_clean_accuracy_over_chance = float(raw.get("min_clean_accuracy_over_chance", 0.0))

    # Artifact integrity gate (dataset-specific). This does not encode baselines; it prevents
    # a common "looks like it ran but is meaningless" failure mode (e.g., nuScenes label-map mismatch).
    require_label_map_sha256_match = bool(raw.get("require_label_map_sha256_match", False))

    return {
        "enabled": enabled,
        "strict": strict,
        "min_clean_accuracy": min_clean_accuracy,
        "min_correct_samples": min_correct_samples,
        "min_correct_fraction": min_correct_fraction,
        "min_clean_accuracy_over_chance": min_clean_accuracy_over_chance,
        "require_label_map_sha256_match": require_label_map_sha256_match,
    }


def evaluate_clean_validity(
    summary: Dict[str, Any],
    gates: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a structured validity report for a single (defense, attack) evaluation.

    `summary` is the return value of `cli.utils.evaluate_attack_runner`.
    """
    enabled = bool(gates.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    samples = int(summary.get("samples", 0) or 0)
    correct = int(summary.get("correct_samples", 0) or 0)
    clean_acc = float(summary.get("clean_accuracy", 0.0) or 0.0)
    num_classes = int(summary.get("num_classes", 0) or 0)

    min_clean_accuracy = float(gates.get("min_clean_accuracy", 0.05))
    min_correct_samples = int(gates.get("min_correct_samples", 10))
    min_correct_fraction = float(gates.get("min_correct_fraction", 0.0) or 0.0)
    min_clean_accuracy_over_chance = float(gates.get("min_clean_accuracy_over_chance", 0.0) or 0.0)

    required_correct = int(min_correct_samples)
    if samples > 0 and min_correct_fraction > 0.0:
        required_correct = max(required_correct, int(math.ceil(min_correct_fraction * samples)))

    chance_acc = None
    required_clean_acc = float(min_clean_accuracy)
    if min_clean_accuracy_over_chance > 0.0:
        # Use a simple uniform-chance baseline: 1 / num_classes.
        # `num_classes` is recorded by the evaluation runner from the model's logits shape.
        if num_classes >= 2:
            chance_acc = float(1.0 / float(num_classes))
            required_clean_acc = max(required_clean_acc, float(chance_acc + min_clean_accuracy_over_chance))

    reasons: List[str] = []
    if samples <= 0:
        reasons.append("no_samples")
    if correct <= 0:
        reasons.append("no_clean_correct_samples")
    if correct < required_correct:
        reasons.append("too_few_clean_correct_samples")
    if min_clean_accuracy_over_chance > 0.0 and chance_acc is None:
        # If this gate is configured but we can't infer the class count, do not
        # silently pass; the reported ASR may be meaningless.
        reasons.append("missing_num_classes_for_chance_gate")
    if clean_acc < required_clean_acc:
        reasons.append("clean_accuracy_below_threshold")
    if chance_acc is not None and min_clean_accuracy_over_chance > 0.0:
        if clean_acc < float(chance_acc + min_clean_accuracy_over_chance):
            reasons.append("clean_accuracy_not_above_chance")

    passed = len(reasons) == 0
    return {
        "enabled": True,
        "passed": bool(passed),
        "reasons": reasons,
        "thresholds": {
            "min_clean_accuracy": float(min_clean_accuracy),
            "min_correct_samples": int(min_correct_samples),
            "min_correct_fraction": float(min_correct_fraction),
            "required_correct_samples": int(required_correct),
            "min_clean_accuracy_over_chance": float(min_clean_accuracy_over_chance),
            "chance_accuracy": float(chance_acc) if chance_acc is not None else None,
            "required_clean_accuracy": float(required_clean_acc),
        },
        "observed": {
            "samples": int(samples),
            "correct_samples": int(correct),
            "clean_accuracy": float(clean_acc),
            "num_classes": int(num_classes),
            "clean_accuracy_over_chance": float(clean_acc - chance_acc) if chance_acc is not None else None,
        },
        "asr_defined": bool(correct > 0),
    }

