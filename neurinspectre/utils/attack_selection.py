"""NeurInSpectre-guided attack selection."""

from __future__ import annotations

from typing import Any

from ..attacks import APGD, BPDA, EOT, FAB, MemoryAugmentedPGD, PGD, SquareAttack


def select_attack_suite(model, features: dict[str, Any], *, device: str = "cuda") -> list:
    """
    Choose an attack suite based on NeurInSpectre characterization features.

    Expected feature keys (optional):
      - volterra_alpha: float
      - stochastic_defense: bool
      - nondiff_defense: bool
      - bpda_approx: str
      - eps: float
      - steps: int
    """
    eps = float(features.get("eps", 0.031))
    steps = max(int(features.get("steps", 40)), 10)
    alpha = float(features.get("alpha", 0.003))
    volterra_alpha = float(features.get("volterra_alpha", 0.95))
    norm = str(features.get("norm", "linf"))
    memory_length = int(features.get("memory_length", 10))
    kernel = str(features.get("kernel", "power_law"))
    stochastic = bool(features.get("stochastic_defense", False))
    nondiff = bool(features.get("nondiff_defense", False))
    bpda_approx = str(features.get("bpda_approx", "identity"))

    attacks = []

    if nondiff:
        defense_fn = features.get("defense_fn")
        if defense_fn is None:
            raise ValueError("nondiff_defense=True requires 'defense_fn' in features.")
        attacks.append(BPDA(model, defense=defense_fn, approx_name=bpda_approx, eps=eps, steps=steps, device=device))

    if stochastic:
        transform_fn = features.get("transform_fn")
        if transform_fn is None:
            raise ValueError("stochastic_defense=True requires 'transform_fn' in features.")
        attacks.append(EOT(model, transform_fn=transform_fn, eps=eps, steps=steps, device=device))

    if volterra_alpha < 0.8:
        attacks.append(
            MemoryAugmentedPGD(
                model,
                alpha_volterra=volterra_alpha,
                memory_length=memory_length,
                eps=eps,
                alpha=alpha,
                steps=steps,
                norm=norm,
                kernel=kernel,
                device=device,
            )
        )
    else:
        attacks.append(PGD(model, eps=eps, alpha=alpha, steps=steps, norm=norm, device=device))

    # Strong baselines
    attacks.append(APGD(model, eps=eps, steps=max(steps, 100), loss="dlr", device=device))
    attacks.append(FAB(model, norm=norm, steps=max(steps, 50), n_restarts=1, device=device))
    attacks.append(SquareAttack(model, eps=eps, n_queries=max(1000, steps), device=device))

    return attacks
