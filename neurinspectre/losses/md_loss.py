"""Minimal Difference (MD) loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def minimal_difference_loss(
    logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Minimal Difference (MD) loss (Lin et al., ICML 2022).

    MD(x, y) = max_{i!=y} softmax(z)_i
    """
    batch_size = logits.shape[0]
    probs = F.softmax(logits, dim=1)
    probs_except_y = probs.clone()
    probs_except_y[torch.arange(batch_size, device=logits.device), labels] = 0.0
    max_incorrect_prob = probs_except_y.max(dim=1)[0]
    loss = -max_incorrect_prob

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def md_loss(logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return minimal_difference_loss(logits, labels, reduction=reduction)


__all__ = ["minimal_difference_loss", "md_loss"]
