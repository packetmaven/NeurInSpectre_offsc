"""Logit margin loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_margin_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    targeted: bool = False,
    target: torch.Tensor | None = None,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Logit margin loss.

    Untargeted: max_{i!=y} z_i - z_y
    Targeted: z_t - max_{i!=t} z_i
    """
    b, c = logits.shape
    device = logits.device
    if targeted:
        if target is None:
            raise ValueError("Targeted logit margin requires target labels")
        z_t = logits[torch.arange(b, device=device), target]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(b, device=device), target] = False
        z_max_other = logits.masked_fill(~mask, float("-inf")).max(dim=1)[0]
        loss = z_t - z_max_other
    else:
        z_y = logits[torch.arange(b, device=device), y]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(b, device=device), y] = False
        z_max_other = logits.masked_fill(~mask, float("-inf")).max(dim=1)[0]
        loss = z_max_other - z_y

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def enhanced_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    use_softmax_weighting: bool = False,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Enhanced logit margin loss with temperature scaling.
    """
    batch_size = logits.shape[0]

    scaled_logits = logits / float(temperature) if temperature != 1.0 else logits
    z_y = scaled_logits[torch.arange(batch_size, device=logits.device), labels]

    logits_except_y = scaled_logits.clone()
    logits_except_y[torch.arange(batch_size, device=logits.device), labels] = float("-inf")
    z_max_other = logits_except_y.max(dim=1)[0]

    margin = z_max_other - z_y
    if use_softmax_weighting:
        probs = F.softmax(scaled_logits, dim=1)
        confidence = probs[torch.arange(batch_size, device=logits.device), labels]
        margin = margin * confidence

    if reduction == "mean":
        return margin.mean()
    if reduction == "sum":
        return margin.sum()
    return margin


def enhanced_margin_targeted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    use_softmax_weighting: bool = False,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Targeted enhanced margin loss.
    """
    batch_size = logits.shape[0]

    scaled_logits = logits / float(temperature) if temperature != 1.0 else logits
    z_t = scaled_logits[torch.arange(batch_size, device=logits.device), target_labels]

    logits_except_t = scaled_logits.clone()
    logits_except_t[torch.arange(batch_size, device=logits.device), target_labels] = float("-inf")
    z_max_other = logits_except_t.max(dim=1)[0]

    margin = z_t - z_max_other
    if use_softmax_weighting:
        probs = F.softmax(scaled_logits, dim=1)
        confidence = probs[torch.arange(batch_size, device=logits.device), labels]
        margin = margin * confidence

    if reduction == "mean":
        return margin.mean()
    if reduction == "sum":
        return margin.sum()
    return margin
