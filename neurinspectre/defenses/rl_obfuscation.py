"""RL-style gradient obfuscation defense (lightweight placeholder)."""

from __future__ import annotations

import torch


def rl_obfuscation(x: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
    """
    Apply a small stochastic perturbation to mimic RL-driven obfuscation noise.
    """
    return x + torch.randn_like(x) * float(scale)
