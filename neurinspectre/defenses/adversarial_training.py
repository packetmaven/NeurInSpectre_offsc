"""
Adversarial training defense implementations.

These defenses train models on adversarial examples generated during training,
improving robustness but potentially introducing gradient obfuscation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialTraining:
    """Standard adversarial training (Madry et al. 2018)."""

    def __init__(self, model: nn.Module, attack: nn.Module, eps: float = 8 / 255, device: str = "cuda"):
        self.model = model
        self.attack = attack
        self.eps = eps
        self.device = device

    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        self.attack.model.train()

        x = x.to(self.device)
        y = y.to(self.device)

        with torch.enable_grad():
            x_adv = self.attack(x, y)

        optimizer.zero_grad()
        logits_adv = self.model(x_adv)
        loss = F.cross_entropy(logits_adv, y)

        loss.backward()
        optimizer.step()
        return float(loss.item())


class TRADES:
    """TRADES: Theoretically Principled Trade-off (Zhang et al. 2019)."""

    def __init__(self, model: nn.Module, attack: nn.Module, beta: float = 6.0, device: str = "cuda"):
        self.model = model
        self.attack = attack
        self.beta = beta
        self.device = device

    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()

        x = x.to(self.device)
        y = y.to(self.device)

        logits_clean = self.model(x)
        loss_natural = F.cross_entropy(logits_clean, y)

        with torch.enable_grad():
            x_adv = self.attack(x, y)

        logits_adv = self.model(x_adv)
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean, dim=1),
            reduction="batchmean",
        )

        loss = loss_natural + self.beta * loss_robust
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())


class MART:
    """MART: Misclassification Aware Adversarial Training (Wang et al. 2020)."""

    def __init__(self, model: nn.Module, attack: nn.Module, beta: float = 5.0, device: str = "cuda"):
        self.model = model
        self.attack = attack
        self.beta = beta
        self.device = device

    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()

        x = x.to(self.device)
        y = y.to(self.device)

        logits_clean = self.model(x)
        loss_natural = F.cross_entropy(logits_clean, y)
        pred_clean = logits_clean.argmax(1)
        correct_mask = pred_clean == y

        with torch.enable_grad():
            x_adv = self.attack(x, y)

        logits_adv = self.model(x_adv)
        pred_adv = logits_adv.argmax(1)

        weights = torch.ones_like(y, dtype=torch.float)
        misclassified = correct_mask & (pred_adv != y)
        weights[misclassified] = 2.0

        loss_adv = F.cross_entropy(logits_adv, y, reduction="none")
        loss_robust = (loss_adv * weights).mean()

        loss = loss_natural + self.beta * loss_robust
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())


class RLAdversarialTraining:
    """
    RL-trained adversarial training defense (simplified).

    Uses a policy network to adapt defense parameters per input.
    """

    def __init__(self, model: nn.Module, policy_net: Optional[nn.Module] = None, device: str = "cuda"):
        self.model = model
        self.policy_net = policy_net or self._default_policy_net()
        self.device = device
        self.prev_params: Optional[torch.Tensor] = None

    def _default_policy_net(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "features"):
            raise AttributeError("model must expose .features() for RLAdversarialTraining")

        with torch.no_grad():
            features = self.model.features(x)
            features_pooled = F.adaptive_avg_pool2d(features, 1).view(x.size(0), -1)

        defense_params = self.policy_net(features_pooled)
        x_defended = x.clone()

        for i in range(x.size(0)):
            noise_std = defense_params[i, 0].item() * 0.1
            smooth_factor = defense_params[i, 1].item()
            clip_threshold = defense_params[i, 2].item() * 0.05

            noise = torch.randn_like(x[i : i + 1]) * noise_std
            x_i = x[i : i + 1] + noise

            if smooth_factor > 0.5:
                x_i = F.avg_pool2d(x_i, 3, stride=1, padding=1)

            x_i = torch.clamp(x_i, clip_threshold, 1 - clip_threshold)
            x_defended[i] = x_i.squeeze(0)

        self.prev_params = defense_params.detach()
        return x_defended
