"""Attack evaluation utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class AttackEvaluator:
    """Simple evaluator for attack success rates and sanity metrics."""

    def __init__(self, model, device: str = "cuda"):
        device = str(device)
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        elif device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            device = "cpu"
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def _grad_stats(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        x = x.detach().clone().requires_grad_(True)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        grad = x.grad
        flat = grad.view(grad.size(0), -1)
        norms = flat.norm(p=2, dim=1)
        nan_count = torch.isnan(grad).sum().item()
        inf_count = torch.isinf(grad).sum().item()
        return {
            "grad_norm_mean": float(norms.mean().item()),
            "grad_norm_max": float(norms.max().item()),
            "grad_nan_count": float(nan_count),
            "grad_inf_count": float(inf_count),
        }

    def evaluate_single_batch(
        self,
        attack,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        compute_grad_stats: bool = True,
    ) -> dict[str, Any]:
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            clean_logits = self.model(x)
            clean_preds = clean_logits.argmax(1)
            clean_acc = (clean_preds == y).float().mean().item()

        adv = None
        if hasattr(attack, "run") and callable(getattr(attack, "run")):
            try:
                result = attack.run(self.model, x, y)
                adv = result.x_adv if hasattr(result, "x_adv") else result
            except TypeError:
                try:
                    result = attack.run(x, y)
                    adv = result.x_adv if hasattr(result, "x_adv") else result
                except TypeError:
                    adv = None
        if adv is None:
            adv_out = attack(x, y)
            adv = adv_out[0] if isinstance(adv_out, (tuple, list)) else adv_out
        with torch.no_grad():
            adv_logits = self.model(adv)
            adv_preds = adv_logits.argmax(1)
            adv_acc = (adv_preds == y).float().mean().item()

        attack_success_rate = float((adv_preds != y).float().mean().item())
        out = {
            "clean_accuracy": float(clean_acc),
            "adversarial_accuracy": float(adv_acc),
            "attack_success_rate": attack_success_rate,
        }
        if compute_grad_stats:
            out.update(self._grad_stats(adv, y))
        return out

    def evaluate(
        self,
        attack,
        dataloader,
        *,
        max_batches: int | None = None,
        compute_grad_stats: bool = True,
    ) -> dict[str, Any]:
        total = 0
        clean_correct = 0
        adv_correct = 0
        asr_sum = 0.0
        grad_stats_accum = {"grad_norm_mean": 0.0, "grad_norm_max": 0.0, "grad_nan_count": 0.0, "grad_inf_count": 0.0}
        batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            res = self.evaluate_single_batch(attack, x, y, compute_grad_stats=compute_grad_stats)
            bsz = int(x.size(0))
            total += bsz
            clean_correct += int(res["clean_accuracy"] * bsz)
            adv_correct += int(res["adversarial_accuracy"] * bsz)
            asr_sum += float(res["attack_success_rate"]) * bsz
            if compute_grad_stats:
                for k in grad_stats_accum:
                    grad_stats_accum[k] += float(res[k]) * bsz
            batches += 1

        if total == 0:
            return {"error": "No samples evaluated."}

        out = {
            "clean_accuracy": clean_correct / total,
            "adversarial_accuracy": adv_correct / total,
            "attack_success_rate": asr_sum / total,
            "batches": batches,
            "samples": total,
        }
        if compute_grad_stats:
            out.update({k: v / total for k, v in grad_stats_accum.items()})
        return out
