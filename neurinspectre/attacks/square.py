"""Square Attack (query-efficient, black-box)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Attack


class SquareAttack(Attack):
    """
    Query-efficient black-box adversarial attack via random search.

    Square Attack perturbs random square regions of the input and accepts
    improvements in a query-efficient manner. This is critical for validating
    that improvements aren't due to gradient masking.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        n_queries: int = 5000,
        p_init: float = 0.8,
        loss_type: str = "margin",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.eps = float(eps)
        self.n_queries = int(n_queries)
        self.p_init = float(p_init)
        self.loss_type = str(loss_type)

        if self.loss_type not in {"margin", "ce"}:
            raise ValueError("loss_type must be 'margin' or 'ce'.")

        # Ensure sufficient queries for convergence
        assert self.n_queries >= 1000, "Square Attack requires ≥1000 queries for reliable results"

    def _margin_loss(self, logits: torch.Tensor, y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        """
        Compute margin loss for Square Attack.

        Margin loss = z_y - max_{i≠y} z_i
        """
        b = logits.size(0)
        z_y = logits[torch.arange(b), y]

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(b), y] = False
        z_max_other = logits[mask].view(b, -1).max(dim=1)[0]

        margin = z_y - z_max_other
        return -margin if not targeted else margin

    def _square_size_schedule(self, query: int) -> float:
        """Compute square size ratio p(q) = p_0 * (1 - q/Q)."""
        return self.p_init * (1.0 - query / self.n_queries)

    def _get_square_coordinates(self, img_size: Tuple[int, int], p: float) -> Tuple[int, int, int, int]:
        """Sample random square coordinates."""
        h, w = img_size
        s_h = max(1, int(np.sqrt(p) * h))
        s_w = max(1, int(np.sqrt(p) * w))
        h_start = np.random.randint(0, h - s_h + 1)
        w_start = np.random.randint(0, w - s_w + 1)
        return h_start, h_start + s_h, w_start, w_start + s_w

    def _initialize_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize perturbation with random uniform noise in [-eps, eps]."""
        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = torch.clamp(x + delta, 0, 1) - x
        return delta

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples via Square Attack.

        Returns:
            (adversarial examples, stats dict)
        """
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = x.size(0)
        if x.ndim != 4:
            raise ValueError("SquareAttack expects 4D inputs (N,C,H,W).")
        _, c, h, w = x.shape

        delta = self._initialize_delta(x)

        with torch.no_grad():
            logits_init = self.model(x + delta)
            if self.loss_type == "margin":
                loss_best = self._margin_loss(logits_init, y, targeted)
            else:
                loss_best = F.cross_entropy(logits_init, y, reduction="none")
                if targeted:
                    loss_best = -loss_best

        queries_used = torch.zeros(batch_size, device=self.device)
        success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for query in range(self.n_queries):
            p = self._square_size_schedule(query)
            delta_new = delta.clone()

            for b in range(batch_size):
                if success[b]:
                    continue

                h_start, h_end, w_start, w_end = self._get_square_coordinates((h, w), p)
                square_delta = torch.zeros_like(delta[b])
                square_delta[:, h_start:h_end, w_start:w_end] = torch.zeros_like(
                    delta[b, :, h_start:h_end, w_start:w_end]
                ).uniform_(-self.eps, self.eps)

                delta_new[b] = delta[b].clone()
                delta_new[b, :, h_start:h_end, w_start:w_end] = square_delta[:, h_start:h_end, w_start:w_end]
                delta_new[b] = torch.clamp(delta_new[b], -self.eps, self.eps)
                delta_new[b] = torch.clamp(x[b] + delta_new[b], 0, 1) - x[b]

            with torch.no_grad():
                logits_new = self.model(x + delta_new)
                if self.loss_type == "margin":
                    loss_new = self._margin_loss(logits_new, y, targeted)
                else:
                    loss_new = F.cross_entropy(logits_new, y, reduction="none")
                    if targeted:
                        loss_new = -loss_new

                improved = loss_new > loss_best
                if improved.any():
                    delta[improved] = delta_new[improved]
                    loss_best[improved] = loss_new[improved]

                queries_used += (~success).float()

                preds = logits_new.argmax(1)
                if targeted:
                    candidate_success = preds == y
                else:
                    candidate_success = preds != y
                success = success | (candidate_success & improved)

            if success.all():
                if verbose:
                    print(f"[Square Attack] All samples adversarial at query {query+1}/{self.n_queries}")
                break

        with torch.no_grad():
            logits_final = self.model(x + delta)
            final_margin = self._margin_loss(logits_final, y, targeted)

        stats = {
            "queries_used": queries_used.cpu().numpy(),
            "success": success.cpu().numpy(),
            "final_margin": final_margin.cpu().numpy(),
            "asr": success.float().mean().item(),
        }

        if verbose:
            print(f"[Square Attack] ASR: {stats['asr']*100:.1f}% | Avg queries: {queries_used.mean().item():.0f}")

        return (x + delta).detach(), stats


class SquareAttackL2(Attack):
    """Square Attack variant for L2 perturbations."""

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.5,
        n_queries: int = 5000,
        p_init: float = 0.8,
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.eps = float(eps)
        self.n_queries = int(n_queries)
        self.p_init = float(p_init)

    def _project_l2(self, delta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        norms = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = torch.min(self.eps / norms, torch.ones_like(norms))
        delta_proj = delta * factors.view(-1, 1, 1, 1)
        delta_proj = torch.clamp(x + delta_proj, 0, 1) - x
        return delta_proj

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = x.size(0)
        if x.ndim != 4:
            raise ValueError("SquareAttackL2 expects 4D inputs (N,C,H,W).")
        _, _, h, w = x.shape

        delta = torch.randn_like(x)
        delta = self._project_l2(delta, x)

        with torch.no_grad():
            logits_init = self.model(x + delta)
            loss_best = -(logits_init[torch.arange(batch_size), y] - logits_init.scatter(1, y.unsqueeze(1), -1e10).max(1)[0])

        queries_used = torch.zeros(batch_size, device=self.device)
        success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for query in range(self.n_queries):
            p = self.p_init * (1.0 - query / self.n_queries)
            delta_new = delta.clone()

            for b in range(batch_size):
                if success[b]:
                    continue

                s = max(1, int(np.sqrt(p) * h))
                h_start = np.random.randint(0, h - s + 1)
                w_start = np.random.randint(0, w - s + 1)

                square_delta = torch.randn_like(delta[b, :, h_start : h_start + s, w_start : w_start + s])
                delta_new[b, :, h_start : h_start + s, w_start : w_start + s] = square_delta

                delta_new[b] = self._project_l2(delta_new[b].unsqueeze(0), x[b].unsqueeze(0)).squeeze(0)

            with torch.no_grad():
                logits_new = self.model(x + delta_new)
                loss_new = -(logits_new[torch.arange(batch_size), y] - logits_new.scatter(1, y.unsqueeze(1), -1e10).max(1)[0])

                improved = loss_new > loss_best
                if improved.any():
                    delta[improved] = delta_new[improved]
                    loss_best[improved] = loss_new[improved]

                queries_used += (~success).float()
                preds = logits_new.argmax(1)
                candidate_success = preds != y
                success = success | (candidate_success & improved)

            if success.all():
                break

        stats = {
            "queries_used": queries_used.cpu().numpy(),
            "success": success.cpu().numpy(),
            "asr": success.float().mean().item(),
        }

        return (x + delta).detach(), stats
