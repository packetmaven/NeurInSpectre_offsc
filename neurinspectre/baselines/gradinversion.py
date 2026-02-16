"""
GradInversion baseline wrapper.

NeurInSpectre includes a research-oriented gradient inversion implementation in
`neurinspectre.attacks.gradient_inversion_attack`.

This module exists to provide a baseline-named import surface:
  - "GradInversion" (Yin et al., CVPR 2021)
  - compatible with NeurInSpectre's GradientInversionAttack(method="gradinversion")

For end-to-end runs (dataset + model + gradient capture), prefer:
  `neurinspectre baselines gradient-inversion run --method gradinversion ...`
"""

from __future__ import annotations

from neurinspectre.attacks.gradient_inversion_attack import (  # re-export
    GradientInversionAttack,
    GradientInversionConfig,
)

__all__ = ["GradientInversionAttack", "GradientInversionConfig"]

