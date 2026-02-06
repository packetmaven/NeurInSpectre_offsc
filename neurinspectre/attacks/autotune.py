"""Lightweight hyperparameter tuning for attacks."""

from __future__ import annotations

import itertools

from ..utils.evaluation import AttackEvaluator


def tune_attack_params(attack_cls, model, x_val, y_val, param_grid, max_trials: int = 10, device: str = "auto"):
    """
    Lightweight hyperparameter tuner for attacks.
    param_grid: dict of {name: [values]}.
    """
    keys = list(param_grid.keys())
    candidates = list(itertools.product(*[param_grid[k] for k in keys]))[: int(max_trials)]

    best_asr = -1.0
    best_kwargs = None

    resolved_device = device
    if str(device) == "auto":
        import torch

        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

    evaluator = AttackEvaluator(model, device=resolved_device)

    for vals in candidates:
        kwargs = dict(zip(keys, vals))
        if "device" not in kwargs:
            kwargs["device"] = resolved_device
        attack = attack_cls(model, **kwargs)
        res = evaluator.evaluate_single_batch(attack, x_val, y_val)
        asr = res["attack_success_rate"]
        if asr > best_asr:
            best_asr = asr
            best_kwargs = kwargs

    return best_kwargs, best_asr
