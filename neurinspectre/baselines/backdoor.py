"""
Training-stage backdoor baselines for Issue 4 (Subnetwork Hijack §5.3).

Baselines requested:
- BadNets (Gu et al. 2017): patch trigger + label override
- WaNet (Nguyen & Tran 2021): imperceptible warping trigger
- Subnet Replacement (Qi et al. 2022): deployment-stage backdoor via subnet transplant

Scope note:
- BadNets and WaNet are implemented directly (small, self-contained).
- Subnet Replacement is implemented as a practical "subnet transplant" helper:
  replace a selected module prefix from a donor (backdoored) model into a victim
  (clean) model. This captures the core *mechanism* (deployment-time weight
  replacement) while keeping the implementation architecture-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class BackdoorMetrics:
    clean_accuracy: float
    asr_all: float
    asr_non_target: float
    total: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "clean_accuracy": float(self.clean_accuracy),
            "asr_all": float(self.asr_all),
            "asr_non_target": float(self.asr_non_target),
            "total": float(self.total),
        }


def _identity_grid(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # grid_sample expects normalized coords in [-1, 1]
    ys = torch.linspace(-1.0, 1.0, steps=int(h), device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=int(w), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)  # (H,W,2)
    return grid


class BadNetsTrigger:
    def __init__(self, *, size: int = 3, value: float = 1.0, location: str = "br"):
        self.size = int(max(1, size))
        self.value = float(value)
        self.location = str(location)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W)
        if x.ndim != 3:
            raise ValueError(f"BadNetsTrigger expects (C,H,W), got {tuple(x.shape)}")
        c, h, w = x.shape
        s = int(min(self.size, h, w))
        out = x.clone()
        if self.location in {"br", "bottom_right"}:
            out[:, h - s : h, w - s : w] = self.value
        elif self.location in {"bl", "bottom_left"}:
            out[:, h - s : h, 0:s] = self.value
        elif self.location in {"tr", "top_right"}:
            out[:, 0:s, w - s : w] = self.value
        elif self.location in {"tl", "top_left"}:
            out[:, 0:s, 0:s] = self.value
        else:
            raise ValueError(f"Unknown location={self.location!r}")
        return out


class WaNetTrigger:
    def __init__(self, *, strength: float = 0.5, noise_strength: float = 0.2, k: int = 4, seed: int = 0):
        self.strength = float(strength)
        self.noise_strength = float(noise_strength)
        self.k = int(max(2, k))
        self.seed = int(seed)

    def _base_noise(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rng = torch.Generator(device=device)
        rng.manual_seed(int(self.seed))
        return (torch.rand((1, 2, self.k, self.k), generator=rng, device=device, dtype=dtype) * 2.0) - 1.0

    def grid(self, h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = self._base_noise(device=device, dtype=dtype)
        base_up = F.interpolate(base, size=(int(h), int(w)), mode="bicubic", align_corners=True)
        grid = _identity_grid(h, w, device=device, dtype=dtype).unsqueeze(0)  # (1,H,W,2)
        # base_up: (1,2,H,W) -> (1,H,W,2)
        delta = base_up.permute(0, 2, 3, 1)
        out = grid + float(self.strength) * delta
        return torch.clamp(out, -1.0, 1.0)

    def noisy_grid(self, h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Noise-mode warping to reduce simple trigger detection.
        rng = torch.Generator(device=device)
        rng.manual_seed(int(self.seed) + 1337)
        noise = (torch.rand((1, 2, self.k, self.k), generator=rng, device=device, dtype=dtype) * 2.0) - 1.0
        noise_up = F.interpolate(noise, size=(int(h), int(w)), mode="bicubic", align_corners=True)
        grid = self.grid(h, w, device=device, dtype=dtype)
        delta = noise_up.permute(0, 2, 3, 1)
        out = grid + float(self.noise_strength) * delta
        return torch.clamp(out, -1.0, 1.0)

    def __call__(self, x: torch.Tensor, *, noise_mode: bool = False) -> torch.Tensor:
        # x: (C,H,W)
        if x.ndim != 3:
            raise ValueError(f"WaNetTrigger expects (C,H,W), got {tuple(x.shape)}")
        c, h, w = x.shape
        device = x.device
        dtype = x.dtype
        g = self.noisy_grid(h, w, device=device, dtype=dtype) if noise_mode else self.grid(h, w, device=device, dtype=dtype)
        x_b = x.unsqueeze(0)  # (1,C,H,W)
        warped = F.grid_sample(x_b, g, mode="bilinear", padding_mode="reflection", align_corners=True)
        return warped.squeeze(0).clamp(0.0, 1.0)


class PoisonedDataset(Dataset):
    """
    Wrap a base (x,y) dataset, poisoning a fixed subset of indices.
    """

    def __init__(
        self,
        base: Dataset,
        *,
        poison_indices: Iterable[int],
        target_label: int,
        trigger,
        wanet_noise_mode_frac: float = 0.0,
        seed: int = 0,
    ):
        self.base = base
        self.poison_set = set(int(i) for i in poison_indices)
        self.target_label = int(target_label)
        self.trigger = trigger
        self.wanet_noise_mode_frac = float(max(0.0, min(1.0, wanet_noise_mode_frac)))
        self.seed = int(seed)

        # Deterministic subset for WaNet "noise mode".
        if self.wanet_noise_mode_frac > 0.0:
            rng = np.random.default_rng(int(seed) + 2022)
            idxs = np.array(sorted(self.poison_set), dtype=np.int64)
            n_noise = int(math.floor(self.wanet_noise_mode_frac * float(idxs.size)))
            noise_idxs = set(int(i) for i in rng.choice(idxs, size=n_noise, replace=False)) if n_noise > 0 else set()
            self._noise_mode = noise_idxs
        else:
            self._noise_mode = set()

    def __len__(self) -> int:  # noqa: D401
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[int(idx)]
        if int(idx) in self.poison_set:
            if isinstance(self.trigger, WaNetTrigger):
                x = self.trigger(x, noise_mode=(int(idx) in self._noise_mode))
            else:
                x = self.trigger(x)
            y = self.target_label
        return x, y


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


@torch.no_grad()
def evaluate_backdoor(
    model: nn.Module,
    loader: DataLoader,
    *,
    trigger=None,
    target_label: int,
    device: torch.device,
) -> Tuple[float, float, float, int]:
    model.eval()
    total = 0
    clean_correct = 0
    asr_hits_all = 0
    asr_hits_non_target = 0
    non_target_total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        total += int(y.numel())
        clean_correct += int((preds == y).sum().item())

        if trigger is not None:
            # Apply trigger in eval for ASR.
            x_trig = []
            for i in range(int(x.size(0))):
                xi = x[i].detach().cpu()
                if isinstance(trigger, WaNetTrigger):
                    xi2 = trigger(xi, noise_mode=False)
                else:
                    xi2 = trigger(xi)
                x_trig.append(xi2)
            x_trig_t = torch.stack(x_trig, dim=0).to(device)
            logits_t = model(x_trig_t)
            preds_t = logits_t.argmax(dim=1)
            hits = preds_t == int(target_label)
            asr_hits_all += int(hits.sum().item())

            non_target = y != int(target_label)
            non_target_total += int(non_target.sum().item())
            asr_hits_non_target += int((hits & non_target).sum().item())

    clean_acc = float(clean_correct / max(1, total))
    asr_all = float(asr_hits_all / max(1, total))
    asr_non_target = float(asr_hits_non_target / max(1, non_target_total))
    return clean_acc, asr_all, asr_non_target, int(total)


def subnet_transplant_state_dict(
    *,
    victim: Dict[str, torch.Tensor],
    donor: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """
    Replace a subset of weights (by key prefix) from donor into victim.
    """
    pref = str(prefix)
    if not pref:
        raise ValueError("prefix must be non-empty")
    out = dict(victim)
    replaced = 0
    for k, v in donor.items():
        if k.startswith(pref):
            out[k] = v.detach().clone()
            replaced += 1
    if replaced == 0:
        raise ValueError(f"No keys matched prefix={prefix!r} for subnet transplant.")
    return out


def pick_poison_indices(n: int, *, poison_rate: float, seed: int) -> List[int]:
    n = int(n)
    if n <= 0:
        return []
    r = float(poison_rate)
    if r <= 0.0:
        return []
    k = int(max(1, math.floor(r * float(n))))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(np.arange(n, dtype=np.int64), size=k, replace=False)
    return [int(i) for i in idx.tolist()]


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    log_every: int = 200,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    step = 0
    for ep in range(int(max(1, epochs))):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if log_every and step % int(log_every) == 0:
                acc = _accuracy_from_logits(logits.detach(), y)
                # Keep stdout minimal; callers can redirect logs if needed.
                print(f"[train] epoch={ep+1} step={step} loss={loss.item():.4f} acc={acc:.3f}")


