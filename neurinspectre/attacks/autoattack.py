"""
AutoAttack ensemble implementation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .apgd import APGD
from .fab import FABEnsemble
from .square import SquareAttack


class AutoAttackEnsemble:
    """
    Simple AutoAttack-style ensemble that applies a list of attacks sequentially.
    """

    def __init__(self, model, attacks, device: str = "cuda"):
        self.model = model.to(device)
        self.attacks = attacks
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        adv = x.clone()
        still_robust = torch.ones(x.size(0), dtype=torch.bool, device=self.device)

        for attack in self.attacks:
            if not still_robust.any():
                break

            adv_subset = adv[still_robust]
            y_subset = y[still_robust]

            adv_out = attack(adv_subset, y_subset)
            adv_attacked = adv_out[0] if isinstance(adv_out, (tuple, list)) else adv_out

            with torch.no_grad():
                logits = self.model(adv_attacked)
                preds = logits.argmax(1)
                newly_adv = preds != y_subset

            adv[still_robust] = adv_attacked
            still_robust_indices = still_robust.nonzero(as_tuple=False).squeeze(1)
            still_robust[still_robust_indices[newly_adv]] = False

        return adv


class AutoAttack:
    """
    AutoAttack ensemble for robust evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        norm: str = "linf",
        eps: float = 8 / 255,
        version: str = "standard",
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.norm = norm
        self.eps = float(eps)
        self.version = version
        self.device = device

        self._init_attacks()

    def _init_attacks(self):
        if self.version == "standard":
            self.attacks = {
                "apgd-ce": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="ce",
                    n_restarts=1,
                    device=self.device,
                ),
                "apgd-dlr": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="dlr",
                    n_restarts=1,
                    device=self.device,
                ),
                "fab": FABEnsemble(self.model, norm=self.norm, device=self.device),
                "square": SquareAttack(
                    self.model, eps=self.eps, n_queries=5000, device=self.device
                )
                if self.norm == "linf"
                else None,
            }
        elif self.version == "plus":
            self.attacks = {
                "apgd-ce": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="ce",
                    n_restarts=5,
                    device=self.device,
                ),
                "apgd-dlr": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="dlr",
                    n_restarts=5,
                    device=self.device,
                ),
                "apgd-md": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="md",
                    n_restarts=1,
                    device=self.device,
                ),
                "fab": FABEnsemble(self.model, norm=self.norm, device=self.device),
                "square": SquareAttack(
                    self.model, eps=self.eps, n_queries=10000, device=self.device
                )
                if self.norm == "linf"
                else None,
            }
        elif self.version == "rand":
            self.attacks = {
                "apgd-ce": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="ce",
                    n_restarts=10,
                    device=self.device,
                ),
                "apgd-dlr": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="dlr",
                    n_restarts=10,
                    device=self.device,
                ),
                "fab": FABEnsemble(self.model, norm=self.norm, device=self.device),
            }
        else:
            raise ValueError("version must be one of: standard, plus, rand")

        self.attacks = {k: v for k, v in self.attacks.items() if v is not None}

    def run(self, x: torch.Tensor, y: torch.Tensor, verbose: bool = True) -> tuple:
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = x.size(0)

        attacks = self.attacks
        if x.ndim != 4:
            attacks = {k: v for k, v in self.attacks.items() if k.startswith("apgd")}
            if verbose:
                print(
                    "[AutoAttack] Non-image input detected; "
                    "running APGD-only subset for compatibility."
                )

        is_adversarial = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        x_adv = x.clone()

        asr_per_attack = {}
        samples_per_attack = {}

        for attack_name, attack in attacks.items():
            if verbose:
                print(f"\n[AutoAttack] Running {attack_name}...")
                print(f"  Currently adversarial: {is_adversarial.sum()}/{batch_size}")

            remaining_mask = ~is_adversarial
            if not remaining_mask.any():
                if verbose:
                    print(f"  All samples already adversarial, skipping {attack_name}")
                asr_per_attack[attack_name] = 0.0
                samples_per_attack[attack_name] = 0
                continue

            x_remaining = x_adv[remaining_mask]
            y_remaining = y[remaining_mask]

            if attack_name == "square":
                x_attacked, _stats = attack(x_remaining, y_remaining, verbose=False)
            else:
                adv_out = attack(x_remaining, y_remaining)
                x_attacked = adv_out[0] if isinstance(adv_out, (tuple, list)) else adv_out

            with torch.no_grad():
                preds = self.model(x_attacked).argmax(1)
                newly_adversarial = preds != y_remaining

                remaining_indices = remaining_mask.nonzero(as_tuple=False).squeeze(1)
                is_adversarial[remaining_indices[newly_adversarial]] = True
                x_adv[remaining_indices[newly_adversarial]] = x_attacked[newly_adversarial]

                asr_per_attack[attack_name] = newly_adversarial.float().mean().item()
                samples_per_attack[attack_name] = newly_adversarial.sum().item()

            if verbose:
                print(
                    f"  {attack_name} ASR: {asr_per_attack[attack_name]*100:.1f}% "
                    f"({samples_per_attack[attack_name]} new adversarial)"
                )

        total_asr = is_adversarial.float().mean().item()
        robust_accuracy = 1.0 - total_asr

        metrics = {
            "robust_accuracy": robust_accuracy,
            "asr": total_asr,
            "asr_per_attack": asr_per_attack,
            "samples_adversarial_per_attack": samples_per_attack,
            "total_adversarial": is_adversarial.sum().item(),
        }

        if verbose:
            print("\n[AutoAttack] Final Results:")
            print(f"  Total ASR: {total_asr*100:.1f}%")
            print(f"  Robust Accuracy: {robust_accuracy*100:.1f}%")
            print(f"  Adversarial samples: {is_adversarial.sum()}/{batch_size}")

        return x_adv, metrics

    def run_standard_eval(self, test_loader: torch.utils.data.DataLoader, n_examples: int = 10000) -> Dict:
        self.model.eval()

        total_samples = 0
        total_correct_clean = 0
        total_correct_robust = 0

        all_metrics = {"asr_per_attack": {name: [] for name in self.attacks.keys()}}

        for x_batch, y_batch in test_loader:
            if total_samples >= n_examples:
                break

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                preds_clean = self.model(x_batch).argmax(1)
                correct_clean = preds_clean == y_batch
                x_correct = x_batch[correct_clean]
                y_correct = y_batch[correct_clean]

            if x_correct.size(0) == 0:
                continue

            _, metrics = self.run(x_correct, y_correct, verbose=False)

            total_samples += x_batch.size(0)
            total_correct_clean += correct_clean.sum().item()
            total_correct_robust += correct_clean.sum().item() * metrics["robust_accuracy"]

            for attack_name, asr in metrics["asr_per_attack"].items():
                all_metrics["asr_per_attack"][attack_name].append(asr)

            print(
                f"Processed {total_samples}/{n_examples} samples | "
                f"Clean: {total_correct_clean/total_samples*100:.1f}% | "
                f"Robust: {total_correct_robust/total_correct_clean*100:.1f}%"
            )

        clean_accuracy = total_correct_clean / total_samples if total_samples > 0 else 0.0
        robust_accuracy = (
            total_correct_robust / total_correct_clean if total_correct_clean > 0 else 0.0
        )

        final_metrics = {
            "clean_accuracy": clean_accuracy,
            "robust_accuracy": robust_accuracy,
            "total_samples": total_samples,
            "avg_asr_per_attack": {
                name: float(np.mean(scores)) if scores else 0.0
                for name, scores in all_metrics["asr_per_attack"].items()
            },
        }

        return final_metrics
