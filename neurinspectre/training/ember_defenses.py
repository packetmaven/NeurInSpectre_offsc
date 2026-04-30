"""
EMBER defense training routines (implementation parity).

Focus:
- gradient regularization (Jacobian penalty)
- defensive distillation (teacher->student with temperature)
- adversarial training with transforms (noise + FGSM-style L2 perturbation)

These are intentionally lightweight and designed to run on CPU/MPS for
artifact evaluation. For full-size training, increase epochs and sample counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TrainStats:
    epochs: int
    last_loss: float

    def to_dict(self) -> Dict[str, float]:
        return {"epochs": float(self.epochs), "last_loss": float(self.last_loss)}


def build_mlp(
    *,
    input_dim: int,
    num_classes: int = 2,
    hidden1: int = 128,
    hidden2: int = 64,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Small MLP suitable for fast tests and EMBER-like tabular inputs.
    """
    return nn.Sequential(
        nn.Linear(int(input_dim), int(hidden1)),
        nn.ReLU(),
        nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        nn.Linear(int(hidden1), int(hidden2)),
        nn.ReLU(),
        nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        nn.Linear(int(hidden2), int(num_classes)),
    )


def _iter_batches(loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for x, y in loader:
        yield x, y


def train_standard(
    model: nn.Module,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str,
    epochs: int = 1,
    lr: float = 1e-3,
) -> TrainStats:
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    crit = nn.CrossEntropyLoss()

    last = 0.0
    for _ep in range(int(epochs)):
        for x, y in _iter_batches(train_loader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            last = float(loss.detach().item())
    return TrainStats(epochs=int(epochs), last_loss=float(last))


def train_gradient_regularized(
    model: nn.Module,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str,
    epochs: int = 1,
    lr: float = 1e-3,
    lambda_grad: float = 0.1,
) -> TrainStats:
    """
    Gradient regularization via penalty on ||dL/dx||^2 (Jacobian-style).
    """
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    crit = nn.CrossEntropyLoss()

    lam = float(lambda_grad)
    last = 0.0
    for _ep in range(int(epochs)):
        for x, y in _iter_batches(train_loader):
            x = x.to(device).detach().requires_grad_(True)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss_ce = crit(logits, y)

            # Compute input-gradient penalty. create_graph=True so the penalty
            # participates in backward (second-order).
            grad = torch.autograd.grad(loss_ce, x, create_graph=True, retain_graph=True)[0]
            grad_pen = (grad.view(int(grad.size(0)), -1).pow(2).sum(dim=1)).mean()

            loss = loss_ce + lam * grad_pen
            loss.backward()
            opt.step()
            last = float(loss.detach().item())
    return TrainStats(epochs=int(epochs), last_loss=float(last))


def train_distilled_student(
    *,
    teacher: nn.Module,
    student: nn.Module,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: str,
    epochs: int = 1,
    lr: float = 1e-3,
    temperature: float = 20.0,
    alpha_hard: float = 0.0,
) -> TrainStats:
    """
    Defensive distillation training for a student model.

    Args:
        alpha_hard: optional weight for hard-label CE loss (0.0 => pure distillation).
    """
    T = float(max(1e-6, temperature))
    alpha = float(max(0.0, min(1.0, alpha_hard)))

    teacher = teacher.to(device).eval()
    student = student.to(device).train()
    opt = torch.optim.Adam(student.parameters(), lr=float(lr))
    crit_ce = nn.CrossEntropyLoss()

    last = 0.0
    for _ep in range(int(epochs)):
        for x, y in _iter_batches(train_loader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                t_logits = teacher(x)
                t_prob = F.softmax(t_logits / T, dim=1)

            s_logits = student(x)
            s_logprob = F.log_softmax(s_logits / T, dim=1)
            loss_kd = F.kl_div(s_logprob, t_prob, reduction="batchmean") * (T * T)

            if alpha > 0.0:
                loss_ce = crit_ce(s_logits, y)
                loss = (1.0 - alpha) * loss_kd + alpha * loss_ce
            else:
                loss = loss_kd

            loss.backward()
            opt.step()
            last = float(loss.detach().item())
    return TrainStats(epochs=int(epochs), last_loss=float(last))


def _l2_normalize_per_sample(g: torch.Tensor) -> torch.Tensor:
    flat = g.view(int(g.size(0)), -1)
    n = flat.norm(p=2, dim=1).clamp(min=1e-12).view(int(g.size(0)), *([1] * (g.ndim - 1)))
    return g / n


def train_at_transform(
    model: nn.Module,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str,
    epochs: int = 1,
    lr: float = 1e-3,
    noise_std: float = 0.05,
    epsilon_l2: float = 0.5,
    pgd_steps: int = 7,
    pgd_step_size: Optional[float] = None,
    random_init: bool = True,
) -> TrainStats:
    """
    Adversarial training with transforms (draft-parity):
      - apply stochastic noise transform
      - generate an L2-bounded adversarial example (PGD-k; default k=7)
      - train on adversarially-perturbed inputs
    """
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    crit = nn.CrossEntropyLoss()

    ns = float(max(0.0, noise_std))
    eps = float(max(0.0, epsilon_l2))

    k = int(max(1, pgd_steps))
    step = float(pgd_step_size) if pgd_step_size is not None else (eps / float(k) if eps > 0.0 else 0.0)

    last = 0.0
    for _ep in range(int(epochs)):
        for x, y in _iter_batches(train_loader):
            x = x.to(device)
            y = y.to(device)

            # Transform (stochastic noise)
            if ns > 0.0:
                x_t = x + torch.randn_like(x) * ns
            else:
                x_t = x

            # L2 PGD on transformed inputs.
            x_base = x_t.detach()
            if bool(random_init) and eps > 0.0:
                delta0 = torch.randn_like(x_base)
                delta0 = _l2_normalize_per_sample(delta0)
                # Uniform radius in [0, eps] (per-sample).
                r = torch.rand(int(x_base.size(0)), device=x_base.device, dtype=x_base.dtype).view(
                    int(x_base.size(0)), *([1] * (x_base.ndim - 1))
                )
                x_adv = (x_base + delta0 * (r * eps)).detach()
            else:
                x_adv = x_base

            for _ in range(k):
                x_adv = x_adv.detach().requires_grad_(True)
                logits = model(x_adv)
                loss = crit(logits, y)
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                if step > 0.0:
                    x_adv = x_adv + float(step) * _l2_normalize_per_sample(grad)

                if eps > 0.0:
                    delta = x_adv - x_base
                    flat = delta.view(int(delta.size(0)), -1)
                    nrm = flat.norm(p=2, dim=1).clamp(min=1e-12)
                    scale = torch.clamp(torch.tensor(eps, device=nrm.device, dtype=nrm.dtype) / nrm, max=1.0)
                    scale = scale.view(int(delta.size(0)), *([1] * (delta.ndim - 1)))
                    x_adv = (x_base + delta * scale).detach()
                else:
                    x_adv = x_adv.detach()

            opt.zero_grad(set_to_none=True)
            logits_adv = model(x_adv)
            loss_adv = crit(logits_adv, y)
            loss_adv.backward()
            opt.step()
            last = float(loss_adv.detach().item())

    return TrainStats(epochs=int(epochs), last_loss=float(last))

