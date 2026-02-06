"""Optional memory-informed loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def memory_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    memory_penalty: torch.Tensor | None = None,
    weight: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Base CE loss optionally augmented with a memory penalty term.
    """
    ce = F.cross_entropy(logits, y, reduction=reduction)
    if memory_penalty is None:
        return ce
    if reduction == "none":
        return ce + weight * memory_penalty
    return ce + weight * memory_penalty.mean()
