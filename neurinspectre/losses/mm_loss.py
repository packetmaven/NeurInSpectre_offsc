"""Minimum Margin (MM) loss."""

from __future__ import annotations

import torch


def minimum_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    use_rescaling: bool = True,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Minimum Margin loss (Gao et al., ICML 2022).

    MM(x, y) = (z_y - max_{i!=y} z_i) / scale
    """
    batch_size, num_classes = logits.shape

    z_y = logits[torch.arange(batch_size, device=logits.device), labels]
    logits_except_y = logits.clone()
    logits_except_y[torch.arange(batch_size, device=logits.device), labels] = float("-inf")
    z_max_other = logits_except_y.max(dim=1)[0]
    raw_margin = z_y - z_max_other

    if use_rescaling:
        sorted_logits, _ = logits.sort(dim=1, descending=True)
        if num_classes >= 3:
            scale = (sorted_logits[:, 0] - sorted_logits[:, 2]).abs() + 1e-8
        else:
            scale = torch.ones_like(z_y)
        loss = raw_margin / scale
    else:
        loss = raw_margin

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def minimum_margin_targeted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    use_rescaling: bool = True,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Targeted MM loss.

    MM-T(x, y, t) = (z_y - z_t) / scale
    """
    batch_size, num_classes = logits.shape

    z_y = logits[torch.arange(batch_size, device=logits.device), labels]
    z_t = logits[torch.arange(batch_size, device=logits.device), target_labels]
    raw_margin = z_y - z_t

    if use_rescaling:
        sorted_logits, _ = logits.sort(dim=1, descending=True)
        if num_classes >= 3:
            scale = (sorted_logits[:, 0] - sorted_logits[:, 2]).abs() + 1e-8
        else:
            scale = torch.ones_like(z_y)
        loss = raw_margin / scale
    else:
        loss = raw_margin

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


mm_loss = minimum_margin_loss


__all__ = ["minimum_margin_loss", "minimum_margin_targeted_loss", "mm_loss"]
