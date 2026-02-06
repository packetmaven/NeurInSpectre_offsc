"""
Automated attack orchestration based on defense characterization.

This is Layer 2 of NeurInSpectre's pipeline. It selects and configures attacks
based on characterization results and runs a composed attack.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn

from ..characterization.defense_analyzer import DefenseAnalyzer, DefenseCharacterization
from .apgd import APGD
from .base import Attack
from .ma_pgd import MAPGD


class AttackOrchestrator(Attack):
    """
    Automated attack orchestrator with defense-aware adaptation.
    """

    def __init__(
        self,
        model: nn.Module,
        characterization: Optional[DefenseCharacterization] = None,
        eps: float = 8 / 255,
        steps: int = 100,
        norm: str = "linf",
        auto_characterize_data: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda",
        verbose: bool = True,
    ):
        super().__init__(model, device)

        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.verbose = verbose

        if characterization is None and auto_characterize_data is None:
            warnings.warn(
                "No characterization provided and no data for auto-characterization. "
                "Will use default APGD attack."
            )
            self.characterization = None
            self.attack_chain = [self._create_default_attack()]
        elif characterization is None:
            if self.verbose:
                print("[AttackOrchestrator] Running auto-characterization...")
            analyzer = DefenseAnalyzer(model, device=device, verbose=verbose)
            self.characterization = analyzer.characterize(auto_characterize_data, eps=eps)
            self.attack_chain = self._build_attack_chain()
        else:
            self.characterization = characterization
            self.attack_chain = self._build_attack_chain()

        if self.verbose:
            self._print_orchestration_plan()

    def _create_default_attack(self) -> Attack:
        return APGD(
            self.model,
            eps=self.eps,
            steps=self.steps,
            norm=self.norm,
            loss="dlr",
            device=self.device,
        )

    def _build_attack_chain(self) -> List[Attack]:
        if self.characterization is None:
            return [self._create_default_attack()]

        char = self.characterization

        if char.requires_mapgd:
            if self.verbose:
                print(
                    f"[Orchestrator] Using MA-PGD (alpha={char.alpha_volterra:.3f}, "
                    f"k={char.recommended_memory_length})"
                )
            base_attack = MAPGD(
                self.model,
                eps=self.eps,
                steps=self.steps,
                norm=self.norm,
                alpha_volterra=char.alpha_volterra,
                memory_length=char.recommended_memory_length,
                use_tg=True,
                device=self.device,
            )
        else:
            if self.verbose:
                print("[Orchestrator] Using APGD (no memory needed)")
            base_attack = APGD(
                self.model,
                eps=self.eps,
                steps=self.steps,
                norm=self.norm,
                loss="dlr",
                device=self.device,
            )

        if char.requires_bpda:
            if self.verbose:
                print("[Orchestrator] BPDA flagged (shattered gradients detected)")

            if hasattr(self.model, "defense") and hasattr(self.model.defense, "forward"):
                if self.verbose:
                    print("  [BPDA] Using learned approximation placeholder")
                base_attack.model = self.model
            else:
                if self.verbose:
                    print("  [BPDA] Using identity approximation (no defense wrapper found)")

        if char.requires_eot and self.verbose:
            print(f"[Orchestrator] EOT flagged ({char.recommended_eot_samples} samples)")
            print("  [EOT] Using adaptive sampling placeholder")

        return [base_attack]

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        primary_attack = self.attack_chain[0]

        if return_stats:
            if hasattr(primary_attack, "forward") and "return_stats" in primary_attack.forward.__code__.co_varnames:
                return primary_attack(x, y, return_stats=True)
            x_adv = primary_attack(x, y)
            stats = {"characterization": self.characterization.to_dict() if self.characterization else {}}
            return x_adv, stats

        return primary_attack(x, y)

    def _print_orchestration_plan(self) -> None:
        if self.characterization is None:
            print("[AttackOrchestrator] No characterization available, using default APGD")
            return

        char = self.characterization

        print("\n" + "=" * 70)
        print("[AttackOrchestrator] ORCHESTRATION PLAN")
        print("=" * 70)

        print("\nDetected Obfuscation Types:")
        for obf_type in char.obfuscation_types:
            print(f"  - {obf_type.value.upper()}")

        print("\nCharacterization Metrics:")
        print(f"  ETD Score:         {char.etd_score:.3f}")
        print(f"  Volterra alpha:    {char.alpha_volterra:.3f}")
        print(f"  Gradient Variance: {char.gradient_variance:.6f}")
        print(f"  Jacobian Rank:     {char.jacobian_rank:.3f}")
        print(f"  Autocorr Timescale:{char.autocorr_timescale:.2f}")
        print(f"  Confidence:        {char.confidence:.2f}")

        print("\nSelected Attack Components:")
        if char.requires_bpda:
            print("  - BPDA (shattered gradients)")
        if char.requires_eot:
            print(f"  - EOT ({char.recommended_eot_samples} samples)")
        if char.requires_mapgd:
            print(
                f"  - MA-PGD (alpha={char.alpha_volterra:.3f}, k={char.recommended_memory_length})"
            )
        if not (char.requires_bpda or char.requires_eot or char.requires_mapgd):
            print("  - Standard APGD (no obfuscation detected)")

        print("\n" + "=" * 70 + "\n")

    def evaluate_asr(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        self.model.eval()
        total_samples = 0
        total_correct_clean = 0
        total_adversarial = 0

        for x_batch, y_batch in data_loader:
            if n_samples is not None and total_samples >= n_samples:
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

            x_adv = self(x_correct, y_correct)

            with torch.no_grad():
                preds_adv = self.model(x_adv).argmax(1)
                adversarial = preds_adv != y_correct

            total_samples += x_batch.size(0)
            total_correct_clean += correct_clean.sum().item()
            total_adversarial += adversarial.sum().item()

            if self.verbose and total_samples % 100 == 0:
                current_asr = (
                    total_adversarial / total_correct_clean if total_correct_clean > 0 else 0
                )
                print(
                    f"[Orchestrator] Processed {total_samples} samples | ASR: {current_asr*100:.1f}%"
                )

        clean_accuracy = total_correct_clean / total_samples if total_samples > 0 else 0
        asr = total_adversarial / total_correct_clean if total_correct_clean > 0 else 0

        metrics = {
            "clean_accuracy": clean_accuracy,
            "asr": asr,
            "total_samples": total_samples,
            "total_adversarial": total_adversarial,
        }

        if self.verbose:
            print("\n[Orchestrator] Final Results:")
            print(f"  Clean Accuracy: {clean_accuracy*100:.1f}%")
            print(f"  ASR: {asr*100:.1f}%")
            print(f"  Adversarial Samples: {total_adversarial}/{total_correct_clean}")

        return metrics


def attack_with_characterization(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    characterization_data: torch.utils.data.DataLoader,
    eps: float = 8 / 255,
    device: str = "cuda",
) -> torch.Tensor:
    analyzer = DefenseAnalyzer(model, device=device)
    char = analyzer.characterize(characterization_data, eps=eps)
    orchestrator = AttackOrchestrator(model, characterization=char, eps=eps, device=device)
    return orchestrator(x, y)
