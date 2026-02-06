"""Difference of Logits Ratio (DLR) loss and CE wrapper."""

from __future__ import annotations

import torch


def dlr_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    targeted: bool = False,
    target: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    DLR loss as used in AutoAttack.

    For untargeted attacks, the loss is maximized to reduce the margin of the true class.
    For targeted attacks, the loss is maximized to increase the target-vs-true margin.
    """
    b, c = logits.shape
    if c < 2:
        raise ValueError("DLR loss requires at least 2 classes.")

    device = logits.device
    u = torch.arange(b, device=device)
    sorted_logits, sorted_idx = torch.sort(logits, dim=1)
    z1 = sorted_logits[:, -1]
    z2 = sorted_logits[:, -2]
    z3 = sorted_logits[:, -3] if c >= 3 else sorted_logits[:, -2]
    denom = (z1 - z3).clamp_min(1e-12)

    z_y = logits[u, y]

    if targeted:
        if target is None:
            raise ValueError("Targeted DLR requires target labels.")
        z_t = logits[u, target]
        loss = (z_t - z_y) / denom
    else:
        is_max = sorted_idx[:, -1] == y
        z_max = torch.where(is_max, z2, z1)
        loss = -(z_y - z_max) / denom

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def ce_loss(logits: torch.Tensor, y: torch.Tensor, *, targeted: bool = False, reduction: str = "mean") -> torch.Tensor:
    """
    Cross-entropy loss wrapper for untargeted/targeted attacks.

    Untargeted: maximize CE (return positive CE).
    Targeted: minimize CE to target (return negative CE for ascent).
    """
    import torch.nn.functional as F

    loss = F.cross_entropy(logits, y, reduction=reduction)
    if targeted:
        return -loss
    return loss
