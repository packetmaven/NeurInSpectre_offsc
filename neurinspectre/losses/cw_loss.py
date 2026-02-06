"""Carlini-Wagner (CW) style margin loss."""

from __future__ import annotations

import torch


def cw_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    targeted: bool = False,
    target: torch.Tensor | None = None,
    kappa: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    CW margin loss. For untargeted attacks, maximize max_other - true.
    For targeted attacks, maximize target - max_other.
    """
    b = logits.size(0)
    device = logits.device
    kappa = float(kappa)

    if targeted and target is None:
        raise ValueError("Targeted CW loss requires target labels.")

    if targeted:
        z_t = logits[torch.arange(b, device=device), target]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(b, device=device), target] = False
        z_max_other = logits.masked_fill(~mask, float("-inf")).max(dim=1)[0]
        loss = (z_t - z_max_other - kappa)
    else:
        z_y = logits[torch.arange(b, device=device), y]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(b, device=device), y] = False
        z_max_other = logits.masked_fill(~mask, float("-inf")).max(dim=1)[0]
        loss = (z_max_other - z_y - kappa)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
