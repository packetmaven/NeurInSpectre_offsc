"""
Tier 2: Head-to-head comparisons vs ART/Foolbox.

This module intentionally provides baseline *methods* (framework adapters) without
shipping any "expected" baseline numbers in-repo.

Primary goal: run comparable attacks under identical eps/norm/budget and emit
machine-readable results (ASR + runtime + clean/robust acc).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..attacks.base_interface import AttackResult
from ..attacks.factory import AttackFactory


def _default_step_size(*, eps: float, steps: int, norm: str) -> float:
    """Heuristic used across frameworks for budget parity."""
    steps = int(max(int(steps), 1))
    norm_key = str(norm).lower().replace("_", "")
    if norm_key in {"linf", "inf"}:
        return float(2.5 * float(eps) / float(steps))
    if norm_key in {"l2", "2"}:
        return float(2.0 * float(eps) / float(steps) ** 0.5)
    return float(float(eps) / 10.0)


def _require_torch_device_compat(framework: str, device: str) -> None:
    """
    ART historically assumes cpu/cuda. On Apple Silicon, users often run on MPS;
    we fail early with a clear message instead of silently producing wrong results.
    """
    if str(device).lower() == "mps" and framework.lower().startswith("art"):
        raise RuntimeError("ART baseline does not support device='mps'. Use --device cpu.")


def _infer_num_classes(model: nn.Module, x: torch.Tensor) -> int:
    with torch.no_grad():
        logits = model(x[:1])
    if logits.ndim == 1:
        return 2
    if logits.ndim == 2 and int(logits.size(1)) == 1:
        return 2
    if logits.ndim >= 2:
        return int(logits.size(1))
    return 0


@dataclass
class FrameworkResult:
    framework: str
    attack: str
    runtime_seconds: float
    clean_accuracy: float
    robust_accuracy: float
    asr: float
    n_samples: int
    n_attackable: int
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "attack": self.attack,
            "runtime_seconds": float(self.runtime_seconds),
            "clean_accuracy": float(self.clean_accuracy),
            "robust_accuracy": float(self.robust_accuracy),
            "attack_success_rate": float(self.asr),
            "n_samples": int(self.n_samples),
            "n_attackable": int(self.n_attackable),
            "extra": dict(self.extra),
        }


class _ARTAutoPGDRunner:
    """
    Adapter running ART's AutoProjectedGradientDescent (APGD) on a torch model.

    Notes:
    - This is intentionally "off-the-shelf" ART usage (no BPDA hooks). On
      non-differentiable defenses, expect weak ASR; that's part of the Tier 2
      comparison story.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        eps: float,
        steps: int,
        norm: str,
        eps_step: Optional[float],
        restarts: int,
        batch_size: int,
        device: str,
        loss_type: Optional[str] = "difference_logits_ratio",
    ) -> None:
        _require_torch_device_compat("art_apgd", device)
        self.model = model
        self.eps = float(eps)
        self.steps = int(steps)
        self.norm = str(norm)
        self.eps_step = float(eps_step) if eps_step is not None else _default_step_size(eps=eps, steps=steps, norm=norm)
        self.restarts = int(restarts)
        self.batch_size = int(batch_size)
        self.device = str(device)
        self.loss_type = loss_type

        self._attack = None
        self._classifier = None

    def _lazy_init(self, x: torch.Tensor) -> None:
        if self._attack is not None:
            return

        try:
            from art.attacks.evasion import AutoProjectedGradientDescent
            from art.estimators.classification.pytorch import PyTorchClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ART is required for framework comparisons.\n"
                "Install it with:\n"
                "  pip install -e \".[frameworks]\""
            ) from exc

        # ART expects a (gpu|cpu) device_type.
        device_type = "cpu" if str(self.device).lower() == "cpu" else "gpu"
        if str(self.device).lower() not in {"cpu", "cuda"}:
            # Keep behavior explicit; ART's device routing is not MPS-aware.
            device_type = "cpu"

        nb_classes = _infer_num_classes(self.model, x)
        if nb_classes <= 0:
            raise ValueError("Unable to infer num_classes from model output.")

        loss = torch.nn.CrossEntropyLoss()
        # Optimizer is not used for attack generation, but PyTorchClassifier accepts it.
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self._classifier = PyTorchClassifier(
            model=self.model,
            loss=loss,
            optimizer=opt,
            input_shape=tuple(int(d) for d in x.shape[1:]),
            nb_classes=int(nb_classes),
            channels_first=True,
            clip_values=(0.0, 1.0),
            device_type=device_type,
        )

        norm = np.inf if str(self.norm).lower() in {"linf", "inf"} else 2
        self._attack = AutoProjectedGradientDescent(
            estimator=self._classifier,
            norm=norm,
            eps=float(self.eps),
            eps_step=float(self.eps_step),
            max_iter=int(self.steps),
            targeted=False,
            nb_random_init=int(self.restarts),
            batch_size=int(self.batch_size),
            loss_type=self.loss_type,
            verbose=False,
        )

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        self._lazy_init(x)
        assert self._attack is not None

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        x_adv_np = self._attack.generate(x=x_np, y=y_np)
        x_adv = torch.from_numpy(x_adv_np).to(device=x.device, dtype=x.dtype)
        return AttackResult(x_adv=x_adv)


class _FoolboxPGDRunner:
    """
    Adapter running Foolbox PGD (Linf/L2).

    Foolbox does not ship APGD in its core API; we use its native PGD
    implementation and label it accordingly in results.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        eps: float,
        steps: int,
        norm: str,
        abs_stepsize: Optional[float],
        restarts: int,
        device: str,
    ) -> None:
        self.model = model
        self.eps = float(eps)
        self.steps = int(steps)
        self.norm = str(norm)
        self.abs_stepsize = float(abs_stepsize) if abs_stepsize is not None else _default_step_size(eps=eps, steps=steps, norm=norm)
        self.restarts = int(restarts)
        self.device = str(device)

    @staticmethod
    def _maybe_raw(t: Any) -> Any:
        return getattr(t, "raw", t)

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        try:
            import foolbox as fb
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Foolbox is required for framework comparisons.\n"
                "Install it with:\n"
                "  pip install -e \".[frameworks]\""
            ) from exc

        bounds = (0.0, 1.0)
        fmodel = fb.PyTorchModel(self.model, bounds=bounds)

        norm_key = str(self.norm).lower().replace("_", "")
        if norm_key in {"linf", "inf"}:
            attack = fb.attacks.LinfPGD(
                steps=int(self.steps),
                abs_stepsize=float(self.abs_stepsize),
                random_start=True,
            )
        elif norm_key in {"l2", "2"}:
            attack = fb.attacks.L2PGD(
                steps=int(self.steps),
                abs_stepsize=float(self.abs_stepsize),
                random_start=True,
            )
        else:
            raise ValueError(f"Unsupported norm for Foolbox runner: {self.norm!r}")

        best_adv = x.detach().clone()
        best_success = torch.zeros(int(x.size(0)), dtype=torch.bool, device=x.device)

        # Multiple restarts: keep first-found success per sample.
        for _ in range(max(int(self.restarts), 1)):
            raw, clipped, is_adv = attack(fmodel, x, y, epsilons=float(self.eps))
            clipped_t = self._maybe_raw(clipped)
            is_adv_t = self._maybe_raw(is_adv)
            if not isinstance(clipped_t, torch.Tensor):
                clipped_t = torch.as_tensor(clipped_t, device=x.device, dtype=x.dtype)
            if not isinstance(is_adv_t, torch.Tensor):
                is_adv_t = torch.as_tensor(is_adv_t, device=x.device)
            is_adv_mask = is_adv_t.to(device=x.device).bool().view(-1)

            update = (~best_success) & is_adv_mask
            if update.any():
                best_adv[update] = clipped_t.to(device=x.device, dtype=x.dtype)[update]
                best_success[update] = True

            if bool(best_success.all()):
                break

        return AttackResult(x_adv=best_adv)


def run_framework_head_to_head(
    *,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    eval_model_factory: Any,
    # attack config
    eps: float,
    norm: str,
    steps: int,
    restarts: int,
    batch_size: int,
    num_samples: Optional[int],
    device: str,
    eps_step: Optional[float] = None,
    fail_fast: bool = True,
    frameworks: Sequence[str] = ("neurinspectre", "art_apgd", "foolbox_pgd"),
    neurinspectre_raw_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, FrameworkResult]:
    """
    Run head-to-head comparison on identical (eps, norm, steps, restarts).

    eval_model_factory: callable(framework_name) -> (eval_model, defense_model_or_none)
    """
    results: Dict[str, FrameworkResult] = {}
    raw_cfg = dict(neurinspectre_raw_config or {})

    for fw in frameworks:
        fw_key = str(fw).lower().strip()
        eval_model, defense_model = eval_model_factory(fw_key)
        eval_model.eval()

        if fw_key == "neurinspectre":
            runner = AttackFactory.create_attack(
                "neurinspectre",
                eval_model,
                defense=defense_model,
                characterization_loader=data_loader,
                device=str(device),
                config={
                    "epsilon": float(eps),
                    "norm": str(norm),
                    "steps": int(steps),
                    "n_restarts": int(restarts),
                    "batch_size": int(batch_size),
                    # Keep comparisons apples-to-apples unless user overrides.
                    "step_size": float(raw_cfg.get("step_size", _default_step_size(eps=eps, steps=steps, norm=norm))),
                    "volterra_mode": str(raw_cfg.get("volterra_mode", "auto")),
                    "characterization_samples": int(raw_cfg.get("characterization_samples", 50)),
                },
            )
            attack_name = "neurinspectre_routed"
        elif fw_key in {"art", "art_apgd", "art-apgd"}:
            runner = _ARTAutoPGDRunner(
                eval_model,
                eps=float(eps),
                steps=int(steps),
                norm=str(norm),
                eps_step=eps_step,
                restarts=int(restarts),
                batch_size=int(batch_size),
                device=str(device),
            )
            attack_name = "apgd"
        elif fw_key in {"foolbox", "foolbox_pgd", "foolbox-pgd"}:
            runner = _FoolboxPGDRunner(
                eval_model,
                eps=float(eps),
                steps=int(steps),
                norm=str(norm),
                abs_stepsize=eps_step,
                restarts=int(restarts),
                device=str(device),
            )
            attack_name = "pgd"
        else:
            raise ValueError(f"Unknown framework key: {fw!r}")

        start = time.perf_counter()
        try:
            # Local copy of evaluate loop (no CLI dependencies).
            total = 0
            clean_correct = 0
            attackable = 0
            adv_misclassified = 0

            for x, y in data_loader:
                if num_samples is not None and total >= int(num_samples):
                    break
                x = x.to(device)
                y = y.to(device)
                if num_samples is not None:
                    remaining = int(num_samples) - int(total)
                    if remaining <= 0:
                        break
                    if int(x.size(0)) > remaining:
                        x = x[:remaining]
                        y = y[:remaining]

                with torch.no_grad():
                    logits = eval_model(x)
                    preds = logits.argmax(dim=1) if logits.ndim >= 2 else (logits > 0).long()
                correct_mask = preds == y
                clean_correct += int(correct_mask.sum().item())
                total += int(x.size(0))

                if not bool(correct_mask.any()):
                    continue

                x_attack = x[correct_mask]
                y_attack = y[correct_mask]
                attackable += int(x_attack.size(0))

                res = runner.run(x_attack, y_attack)
                x_adv = res.x_adv
                with torch.no_grad():
                    adv_logits = eval_model(x_adv)
                    adv_preds = adv_logits.argmax(dim=1) if adv_logits.ndim >= 2 else (adv_logits > 0).long()
                adv_misclassified += int((adv_preds != y_attack).sum().item())

            clean_acc = float(clean_correct / total) if total > 0 else 0.0
            robust_acc = float((clean_correct - adv_misclassified) / total) if total > 0 else 0.0
            asr = float(adv_misclassified / clean_correct) if clean_correct > 0 else 0.0

            results[fw_key] = FrameworkResult(
                framework=fw_key,
                attack=attack_name,
                runtime_seconds=float(time.perf_counter() - start),
                clean_accuracy=float(clean_acc),
                robust_accuracy=float(robust_acc),
                asr=float(asr),
                n_samples=int(total),
                n_attackable=int(attackable),
                extra={
                    "eps": float(eps),
                    "norm": str(norm),
                    "steps": int(steps),
                    "restarts": int(restarts),
                    "eps_step": float(eps_step) if eps_step is not None else None,
                },
            )
        except Exception as exc:
            if fail_fast:
                raise
            results[fw_key] = FrameworkResult(
                framework=fw_key,
                attack=attack_name,
                runtime_seconds=float(time.perf_counter() - start),
                clean_accuracy=0.0,
                robust_accuracy=0.0,
                asr=0.0,
                n_samples=0,
                n_attackable=0,
                extra={"error": str(exc)},
            )

    return results

