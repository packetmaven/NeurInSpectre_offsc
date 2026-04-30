"""
Fast Adaptive Boundary (FAB) Attack implementation.

FAB is a minimum-norm attack that finds adversarial examples on the decision
boundary by iteratively projecting onto linear approximations of the boundary.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Attack
from .numerics import safe_flat_norm, infer_data_range, check_grad_sanity


class FAB(Attack):
    """
    Fast Adaptive Boundary (FAB) attack.
    """

    def __init__(
        self,
        model: nn.Module,
        norm: str = "l2",
        steps: int = 100,
        n_restarts: int = 5,
        alpha_max: float = 0.1,
        eta: float = 1.05,
        beta: float = 0.9,
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.norm_type = norm
        self.steps = int(steps)
        self.n_restarts = int(n_restarts)
        self.alpha_max = float(alpha_max)
        self.eta = float(eta)
        self.beta = float(beta)

        assert norm in ["l2", "linf"], f"FAB only supports L2 and Linf norms, got {norm}"

    def _margin_loss(self, logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = logits.size(0)
        z_y = logits[torch.arange(b), y]

        mask = torch.ones_like(logits).bool()
        mask[torch.arange(b), y] = False
        other_logits = logits[mask].view(b, -1)
        z_max_other, runner_up_idx = other_logits.max(dim=1)

        margin = z_y - z_max_other
        all_classes = torch.arange(logits.size(1), device=logits.device).unsqueeze(0).expand(b, -1)
        all_classes_masked = all_classes[mask].view(b, -1)
        runner_up = all_classes_masked[torch.arange(b), runner_up_idx]
        return margin, runner_up

    def _project_onto_boundary(
        self,
        x: torch.Tensor,
        x_adv: torch.Tensor,
        grad_margin: torch.Tensor,
        margin: torch.Tensor,
        alpha: float,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> torch.Tensor:
        b = x.size(0)

        if self.norm_type == "l2":
            grad_norm = safe_flat_norm(grad_margin, p=2).view(b, 1, 1, 1)
            direction = grad_margin / grad_norm
            step_sizes = alpha * margin.abs().view(b, 1, 1, 1)
            delta = -step_sizes * direction
        else:
            grad_norm = grad_margin.abs().view(b, -1).sum(dim=1).view(b, 1, 1, 1).clamp_min(1e-12)
            direction = grad_margin.sign()
            step_sizes = alpha * margin.abs().view(b, 1, 1, 1)
            delta = -step_sizes * direction

        x_new = x_adv + delta
        x_new = torch.clamp(x_new, x_min, x_max)
        return x_new

    def _backtrack_search(
        self,
        x: torch.Tensor,
        x_adv: torch.Tensor,
        x_best: torch.Tensor,
        y: torch.Tensor,
        *,
        targeted: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        with torch.no_grad():
            logits_best = self.model(x_best)
            preds_best = logits_best.argmax(1)
            if targeted:
                is_adv = preds_best == y
            else:
                is_adv = preds_best != y

        for sample_idx in range(b):
            if is_adv[sample_idx]:
                continue

            left = x_adv[sample_idx].clone()
            right = x_best[sample_idx].clone()
            right = left + self.beta * (right - left)

            for _ in range(10):
                mid = (left + right) / 2
                with torch.no_grad():
                    logits_mid = self.model(mid.unsqueeze(0))
                    pred_mid = logits_mid.argmax(1).item()

                if targeted:
                    mid_is_adv = pred_mid == y[sample_idx]
                else:
                    mid_is_adv = pred_mid != y[sample_idx]

                if mid_is_adv:
                    left = mid
                else:
                    right = mid

            x_best[sample_idx] = left
            is_adv[sample_idx] = True

        return x_best, is_adv

    def _compute_perturbation_norm(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        delta = x_adv - x
        if self.norm_type == "l2":
            return safe_flat_norm(delta, p=2).squeeze(1)
        return delta.abs().view(delta.size(0), -1).max(dim=1)[0]

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_classes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        b = x.size(0)
        x_min, x_max = infer_data_range(x)

        if targeted:
            if target_classes is None:
                raise ValueError("Targeted FAB requires target_classes.")
            target_classes = target_classes.to(self.device)

        best_adv = x.clone()
        best_norm = torch.full((b,), float("inf"), device=self.device)
        found_adv = torch.zeros(b, dtype=torch.bool, device=self.device)

        for restart in range(self.n_restarts):
            if restart == 0:
                if self.norm_type == "l2":
                    delta_init = torch.randn_like(x)
                    delta_init = delta_init / safe_flat_norm(delta_init, p=2).view(b, 1, 1, 1) * 0.1
                else:
                    delta_init = torch.zeros_like(x).uniform_(-0.01, 0.01)
            else:
                if self.norm_type == "l2":
                    delta_init = torch.randn_like(x)
                    delta_init = delta_init / safe_flat_norm(delta_init, p=2).view(b, 1, 1, 1) * (
                        0.5 * restart
                    )
                else:
                    delta_init = torch.zeros_like(x).uniform_(-0.05 * restart, 0.05 * restart)

            x_adv = torch.clamp(x + delta_init, x_min, x_max)

            with torch.no_grad():
                logits_init = self.model(x_adv)
                preds_init = logits_init.argmax(1)
                if targeted:
                    is_adv_init = preds_init == target_classes
                else:
                    is_adv_init = preds_init != y

                norms_init = self._compute_perturbation_norm(x, x_adv)
                improved = is_adv_init & (norms_init < best_norm)
                best_adv[improved] = x_adv[improved]
                best_norm[improved] = norms_init[improved]
                found_adv |= is_adv_init

            alpha = self.alpha_max
            for step in range(self.steps):
                x_adv.requires_grad = True
                logits = self.model(x_adv)

                if targeted:
                    z_target = logits[torch.arange(b), target_classes]
                    mask = torch.ones_like(logits).bool()
                    mask[torch.arange(b), target_classes] = False
                    z_max_other = logits[mask].view(b, -1).max(dim=1)[0]
                    margin = z_max_other - z_target
                else:
                    margin, _ = self._margin_loss(logits, y)

                with torch.no_grad():
                    preds = logits.argmax(1)
                    if targeted:
                        is_adv = preds == target_classes
                    else:
                        is_adv = preds != y
                    if is_adv.all():
                        break

                margin_sum = margin.sum()
                margin_sum.backward()
                grad_margin = x_adv.grad

                if not check_grad_sanity(grad_margin, f"FAB-restart{restart}-step{step}"):
                    break

                with torch.no_grad():
                    x_projected = self._project_onto_boundary(
                        x, x_adv.detach(), grad_margin, margin.detach(), alpha, x_min, x_max
                    )
                    x_updated, _improved_mask = self._backtrack_search(
                        x,
                        x_adv.detach(),
                        x_projected,
                        target_classes if targeted else y,
                        targeted=targeted,
                    )
                    x_adv = x_updated.detach()

                    norms = self._compute_perturbation_norm(x, x_adv)
                    preds_updated = self.model(x_adv).argmax(1)
                    if targeted:
                        is_adv_updated = preds_updated == target_classes
                    else:
                        is_adv_updated = preds_updated != y

                    improvement = is_adv_updated & (norms < best_norm)
                    if improvement.any():
                        best_adv[improvement] = x_adv[improvement]
                        best_norm[improvement] = norms[improvement]
                        found_adv |= is_adv_updated
                    else:
                        alpha = alpha / self.eta
                        if alpha < 1e-6:
                            break

                x_adv.grad = None

        best_adv[~found_adv] = x[~found_adv]
        return best_adv.detach()


class FABT(Attack):
    """Targeted variant of FAB attack."""

    def __init__(
        self,
        model: nn.Module,
        norm: str = "l2",
        n_target_classes: int = 9,
        steps: int = 100,
        n_restarts: int = 1,
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.norm_type = norm
        self.n_target_classes = int(n_target_classes)
        self.base_attack = FAB(model, norm=norm, steps=steps, n_restarts=n_restarts, device=device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        b = x.size(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            num_classes = int(probs.size(1)) if probs.ndim == 2 else 0
        if num_classes <= 1:
            raise ValueError("FABT requires a classifier with at least 2 classes.")

        # Robustness: for small-C models (e.g., toy tests, binary classifiers),
        # cap k so `topk(k)` is always valid.
        k = min(int(self.n_target_classes), max(1, num_classes - 1))

        target_candidates = []
        for sample_idx in range(b):
            true_class = y[sample_idx].item()
            sample_probs = probs[sample_idx].clone()
            sample_probs[true_class] = -1
            _, top_classes = sample_probs.topk(k)
            target_candidates.append(top_classes)

        best_adv = x.clone()
        best_norm = torch.full((b,), float("inf"), device=self.device)

        for target_idx in range(k):
            targets = torch.stack([target_candidates[sample_idx][target_idx] for sample_idx in range(b)])
            x_adv_target = self.base_attack(x, y, targeted=True, target_classes=targets)

            with torch.no_grad():
                logits_target = self.model(x_adv_target)
                preds_target = logits_target.argmax(1)
                is_adv = preds_target == targets

                if self.norm_type == "l2":
                    norms = safe_flat_norm(x_adv_target - x, p=2).squeeze(1)
                else:
                    norms = (x_adv_target - x).abs().view(b, -1).max(dim=1)[0]

                improvement = is_adv & (norms < best_norm)
                best_adv[improvement] = x_adv_target[improvement]
                best_norm[improvement] = norms[improvement]

        return best_adv.detach()


class FABEnsemble(Attack):
    """FAB ensemble: combines FAB (untargeted) and FAB-T (targeted)."""

    def __init__(
        self,
        model: nn.Module,
        norm: str = "l2",
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.fab = FAB(model, norm=norm, device=device)
        self.fabt = FABT(model, norm=norm, device=device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        b = x.size(0)

        x_adv_fab = self.fab(x, y)
        x_adv_fabt = self.fabt(x, y)

        with torch.no_grad():
            if self.fab.norm_type == "l2":
                norm_fab = safe_flat_norm(x_adv_fab - x, p=2).squeeze(1)
                norm_fabt = safe_flat_norm(x_adv_fabt - x, p=2).squeeze(1)
            else:
                norm_fab = (x_adv_fab - x).abs().view(b, -1).max(dim=1)[0]
                norm_fabt = (x_adv_fabt - x).abs().view(b, -1).max(dim=1)[0]

            preds_fab = self.model(x_adv_fab).argmax(1)
            preds_fabt = self.model(x_adv_fabt).argmax(1)

            is_adv_fab = preds_fab != y
            is_adv_fabt = preds_fabt != y

            best_adv = x.clone()
            use_fab = is_adv_fab & ((norm_fab <= norm_fabt) | ~is_adv_fabt)
            best_adv[use_fab] = x_adv_fab[use_fab]

            use_fabt = is_adv_fabt & (norm_fabt < norm_fab)
            best_adv[use_fabt] = x_adv_fabt[use_fabt]

        return best_adv
