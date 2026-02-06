"""Auto-PGD (APGD) with adaptive step size."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .base import Attack
from .base_interface import APGDAttack as _APGDAttack, AttackConfig


class APGD(Attack):
    """
    Auto-PGD with adaptive step size and multiple loss functions.
    """

    def __init__(
        self,
        model,
        eps: float = 0.031,
        norm: str = "linf",
        steps: int = 100,
        loss: str = "dlr",
        n_restarts: int = 1,
        use_tg: bool = False,
        loss_params: Optional[Dict] = None,
        device: str = "cuda",
    ):
        super().__init__(model, device)
        loss_params = loss_params or {}

        self.config = AttackConfig(
            norm=norm,
            epsilon=float(eps),
            n_iterations=int(steps),
            n_restarts=int(n_restarts),
            loss=loss,
            loss_temperature=float(loss_params.get("temperature", 1.0)),
            loss_softmax_weighting=bool(loss_params.get("use_softmax_weighting", False)),
            use_tg=bool(use_tg),
        )
        self.loss_type = loss
        self.n_restarts = int(n_restarts)
        self._attack = _APGDAttack(
            self.config,
            device=device,
            n_restarts=int(n_restarts),
            loss_type=loss,
            eot_iter=1,
            rho=self.config.rho,
            verbose=False,
            use_tg=bool(use_tg),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, targeted: bool = False, target=None) -> torch.Tensor:
        result = self._attack.run(
            self.model,
            x,
            y,
            targeted=targeted,
            target_labels=target,
        )
        return result.x_adv


class APGDEnsemble(Attack):
    """
    Ensemble of APGD attacks with different loss functions.
    """

    def __init__(
        self,
        model,
        eps: float = 0.031,
        norm: str = "linf",
        steps: int = 100,
        losses: list[str] | None = None,
        device: str = "cuda",
    ):
        super().__init__(model, device)
        losses = losses or ["ce", "dlr"]
        self.attacks = [
            APGD(model, eps=eps, norm=norm, steps=steps, loss=loss, device=device)
            for loss in losses
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        adv = x.clone()
        still_robust = torch.ones(x.size(0), dtype=torch.bool, device=self.device)

        for attack in self.attacks:
            if not still_robust.any():
                break

            adv_subset = adv[still_robust]
            y_subset = y[still_robust]
            adv_attacked = attack(adv_subset, y_subset)

            with torch.no_grad():
                logits = self.model(adv_attacked)
                preds = logits.argmax(1)
                newly_adv = preds != y_subset

            adv[still_robust] = adv_attacked
            still_robust_indices = still_robust.nonzero(as_tuple=False).squeeze(1)
            still_robust[still_robust_indices[newly_adv]] = False

        return adv


__all__ = ["APGD", "APGDEnsemble"]
