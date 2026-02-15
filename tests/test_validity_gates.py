from __future__ import annotations


from neurinspectre.evaluation.validity_gates import evaluate_clean_validity, resolve_validity_gates


def test_validity_gates_disabled_returns_disabled_report() -> None:
    gates = resolve_validity_gates({})
    report = evaluate_clean_validity({"samples": 16, "correct_samples": 0, "clean_accuracy": 0.0}, gates)
    assert report == {"enabled": False}


def test_validity_gates_flags_zero_correct_samples() -> None:
    gates = {"enabled": True, "strict": True, "min_clean_accuracy": 0.05, "min_correct_samples": 5}
    report = evaluate_clean_validity(
        {"samples": 16, "correct_samples": 0, "clean_accuracy": 0.0},
        gates,
    )
    assert report["enabled"] is True
    assert report["passed"] is False
    assert report["asr_defined"] is False
    assert "no_clean_correct_samples" in report["reasons"]


def test_validity_gates_passes_reasonable_clean_accuracy() -> None:
    gates = {"enabled": True, "strict": False, "min_clean_accuracy": 0.05, "min_correct_samples": 5}
    report = evaluate_clean_validity(
        {"samples": 16, "correct_samples": 7, "clean_accuracy": 0.4375},
        gates,
    )
    assert report["enabled"] is True
    assert report["passed"] is True
    assert report["reasons"] == []


def test_validity_gates_min_correct_fraction_scales_with_samples() -> None:
    gates = {
        "enabled": True,
        "strict": True,
        "min_clean_accuracy": 0.05,
        "min_correct_samples": 5,
        "min_correct_fraction": 0.2,  # require at least 20% correct
    }
    report = evaluate_clean_validity(
        {"samples": 100, "correct_samples": 10, "clean_accuracy": 0.10},
        gates,
    )
    assert report["enabled"] is True
    assert report["passed"] is False
    assert "too_few_clean_correct_samples" in report["reasons"]


def test_validity_gates_chance_aware_gate_requires_margin_over_uniform_chance() -> None:
    gates = {
        "enabled": True,
        "strict": True,
        "min_clean_accuracy": 0.05,
        "min_correct_samples": 5,
        "min_clean_accuracy_over_chance": 0.05,  # require >= (1/K + 0.05)
    }
    # 10-class task: chance=0.10, required>=0.15
    report = evaluate_clean_validity(
        {"samples": 100, "correct_samples": 12, "clean_accuracy": 0.12, "num_classes": 10},
        gates,
    )
    assert report["enabled"] is True
    assert report["passed"] is False
    assert "clean_accuracy_not_above_chance" in report["reasons"]

