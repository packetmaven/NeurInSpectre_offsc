"""
Comprehensive evaluation metrics for adversarial robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:  # pragma: no cover - dependency guarded
    from scipy import stats
except Exception as exc:  # pragma: no cover
    stats = None
    logger.warning("scipy not available for statistical metrics: %s", exc)


def compute_attack_success_rate(
    predictions: torch.Tensor,
    true_labels: torch.Tensor,
    success_mask: Optional[torch.Tensor] = None,
    originally_correct: Optional[torch.Tensor] = None,
) -> float:
    if success_mask is not None:
        success = success_mask.bool()
        if originally_correct is not None:
            denom = float(originally_correct.float().sum().item())
            if denom == 0:
                return 0.0
            return float((success & originally_correct.bool()).float().sum().item() / denom)
        return float(success.float().mean().item()) if success.numel() > 0 else 0.0

    preds = predictions.detach().view(-1)
    labels = true_labels.detach().view(-1)
    if preds.numel() == 0:
        return 0.0
    misclassified = preds != labels
    if originally_correct is not None:
        denom = float(originally_correct.float().sum().item())
        if denom == 0:
            return 0.0
        return float((misclassified & originally_correct.bool()).float().sum().item() / denom)
    return float(misclassified.float().mean().item())


def compute_robust_accuracy(predictions: torch.Tensor, true_labels: torch.Tensor) -> float:
    preds = predictions.detach().view(-1)
    labels = true_labels.detach().view(-1)
    if preds.numel() == 0:
        return 0.0
    return float((preds == labels).float().mean().item())


def compute_clean_accuracy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(len(y))
    return float(correct / total) if total > 0 else 0.0


def compute_perturbation_metrics(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    norm: str = "Linf",
) -> Dict[str, float]:
    delta = x_adv - x_clean
    batch_size = delta.shape[0]
    if batch_size == 0:
        return {}
    delta_flat = delta.view(batch_size, -1)

    linf = delta_flat.abs().max(dim=1)[0]
    l2 = delta_flat.norm(p=2, dim=1)
    l1 = delta_flat.norm(p=1, dim=1)
    l0 = (delta_flat.abs() > 1e-8).float().sum(dim=1)

    metrics = {
        "linf_mean": float(linf.mean().item()),
        "linf_max": float(linf.max().item()),
        "linf_min": float(linf.min().item()),
        "linf_std": float(linf.std().item()),
        "l2_mean": float(l2.mean().item()),
        "l2_max": float(l2.max().item()),
        "l2_min": float(l2.min().item()),
        "l2_std": float(l2.std().item()),
        "l1_mean": float(l1.mean().item()),
        "l0_mean": float(l0.mean().item()),
    }

    norm = str(norm)
    if norm.lower() in {"linf", "l_inf"}:
        metrics["primary_norm_mean"] = metrics["linf_mean"]
        metrics["primary_norm_max"] = metrics["linf_max"]
    elif norm.lower() == "l2":
        metrics["primary_norm_mean"] = metrics["l2_mean"]
        metrics["primary_norm_max"] = metrics["l2_max"]
    elif norm.lower() == "l1":
        metrics["primary_norm_mean"] = metrics["l1_mean"]
    elif norm.lower() == "l0":
        metrics["primary_norm_mean"] = metrics["l0_mean"]
    return metrics


def compute_query_efficiency(queries_per_sample: List[int], success_mask: torch.Tensor) -> Dict[str, float]:
    if len(queries_per_sample) == 0:
        return {}
    queries = np.array(queries_per_sample, dtype=float)
    success = success_mask.detach().cpu().numpy().astype(bool)

    metrics = {
        "total_queries": int(queries.sum()),
        "mean_queries": float(queries.mean()),
        "median_queries": float(np.median(queries)),
        "mean_queries_successful": float(queries[success].mean()) if success.any() else 0.0,
        "median_queries_successful": float(np.median(queries[success])) if success.any() else 0.0,
    }
    return metrics


def compute_confidence_interval(
    values: Union[torch.Tensor, np.ndarray, List[float]],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    if stats is None:
        raise ImportError("scipy is required for confidence intervals")
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    elif isinstance(values, list):
        values = np.array(values, dtype=float)
    n = int(len(values))
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean
    se = float(stats.sem(values))
    h = se * float(stats.t.ppf((1 + confidence) / 2, n - 1))
    return mean, mean - h, mean + h


def compute_statistical_significance(
    values1: Union[torch.Tensor, np.ndarray],
    values2: Union[torch.Tensor, np.ndarray],
    test: str = "wilcoxon",
) -> Tuple[float, float]:
    if stats is None:
        raise ImportError("scipy is required for statistical tests")
    if isinstance(values1, torch.Tensor):
        values1 = values1.detach().cpu().numpy()
    if isinstance(values2, torch.Tensor):
        values2 = values2.detach().cpu().numpy()

    if test == "wilcoxon":
        stat, p = stats.wilcoxon(values1, values2)
    elif test == "ttest":
        stat, p = stats.ttest_rel(values1, values2)
    elif test == "mannwhitney":
        stat, p = stats.mannwhitneyu(values1, values2)
    else:
        raise ValueError(f"Unknown test: {test}")
    return float(stat), float(p)


def compute_certified_accuracy(
    certified_radii: torch.Tensor,
    predictions: torch.Tensor,
    true_labels: torch.Tensor,
    radius_threshold: float,
) -> float:
    if certified_radii.numel() == 0:
        return 0.0
    correct = predictions == true_labels
    certified = certified_radii >= float(radius_threshold)
    certified_correct = correct & certified
    return float(certified_correct.float().mean().item())


def compute_certified_accuracy_curve(
    certified_radii: torch.Tensor,
    predictions: torch.Tensor,
    true_labels: torch.Tensor,
    radii: Optional[List[float]] = None,
) -> Dict[float, float]:
    radii = radii or [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    curve = {}
    for r in radii:
        curve[r] = compute_certified_accuracy(certified_radii, predictions, true_labels, r)
    return curve


@dataclass
class EvaluationMetrics:
    defense_name: str
    domain: str
    clean_accuracy: float = 0.0
    pgd_asr: float = 0.0
    autoattack_asr: float = 0.0
    neurinspectre_asr: float = 0.0
    pgd_robust_acc: float = 0.0
    autoattack_robust_acc: float = 0.0
    neurinspectre_robust_acc: float = 0.0
    claimed_robust_acc: float = 0.0
    actual_robust_acc: float = 0.0
    mean_perturbation: float = 0.0
    queries_per_sample: float = 0.0
    evaluation_time_seconds: float = 0.0
    neurinspectre_asr_ci: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, float]:
        return {
            "defense_name": self.defense_name,
            "domain": self.domain,
            "clean_accuracy": self.clean_accuracy,
            "pgd_asr": self.pgd_asr,
            "autoattack_asr": self.autoattack_asr,
            "neurinspectre_asr": self.neurinspectre_asr,
            "pgd_robust_acc": self.pgd_robust_acc,
            "autoattack_robust_acc": self.autoattack_robust_acc,
            "neurinspectre_robust_acc": self.neurinspectre_robust_acc,
            "claimed_robust_acc": self.claimed_robust_acc,
            "actual_robust_acc": self.actual_robust_acc,
            "mean_perturbation": self.mean_perturbation,
            "queries_per_sample": self.queries_per_sample,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "neurinspectre_asr_ci_lower": self.neurinspectre_asr_ci[0],
            "neurinspectre_asr_ci_upper": self.neurinspectre_asr_ci[1],
        }

    def to_table_row(self) -> str:
        return (
            f"| {self.defense_name:20s} | "
            f"{self.pgd_asr*100:5.1f}% | "
            f"{self.autoattack_asr*100:5.1f}% | "
            f"{self.neurinspectre_asr*100:5.1f}% | "
            f"{self.claimed_robust_acc*100:5.1f}% |"
        )


def aggregate_batch_metrics(batch_results: List[Dict[str, float]]) -> Dict[str, float]:
    if not batch_results:
        return {"total_samples": 0}
    total_samples = sum(r.get("batch_size", 0) for r in batch_results)
    aggregated = {"total_samples": total_samples}
    for key in ["asr", "robust_acc", "loss"]:
        if key in batch_results[0]:
            weighted_sum = sum(r[key] * r["batch_size"] for r in batch_results)
            aggregated[key] = weighted_sum / max(total_samples, 1)
    for key in ["successful_attacks", "total_queries"]:
        if key in batch_results[0]:
            aggregated[key] = sum(r[key] for r in batch_results)
    return aggregated


def generate_table1(results: Dict[str, EvaluationMetrics], output_path: Optional[str] = None) -> str:
    header = (
        "| Defense              | PGD   | AA    | Ours  | Claimed RA |\n"
        "|----------------------|-------|-------|-------|------------|\n"
    )
    domains = ["vision", "malware", "av_perception"]
    domain_names = {
        "vision": "Content Moderation",
        "malware": "Malware Detection",
        "av_perception": "AV Perception",
    }
    rows = []
    for domain in domains:
        rows.append(f"| **{domain_names.get(domain, domain)}** | | | | |")
        domain_defenses = [(n, m) for n, m in results.items() if m.domain == domain]
        for _name, metrics in domain_defenses:
            rows.append(metrics.to_table_row())

    all_metrics = list(results.values())
    if all_metrics:
        avg_pgd = np.mean([m.pgd_asr for m in all_metrics])
        avg_aa = np.mean([m.autoattack_asr for m in all_metrics])
        avg_ours = np.mean([m.neurinspectre_asr for m in all_metrics])
        avg_claimed = np.mean([m.claimed_robust_acc for m in all_metrics])
    else:
        avg_pgd = avg_aa = avg_ours = avg_claimed = 0.0

    rows.append("|----------------------|-------|-------|-------|------------|")
    rows.append(
        f"| **Average**          | "
        f"{avg_pgd*100:5.1f}% | "
        f"{avg_aa*100:5.1f}% | "
        f"{avg_ours*100:5.1f}% | "
        f"{avg_claimed*100:5.1f}% |"
    )

    table = header + "\n".join(rows)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(table)
        logger.info("Table 1 saved to %s", output_path)
    return table


def generate_latex_table1(results: Dict[str, EvaluationMetrics], output_path: Optional[str] = None) -> str:
    latex = r"""
\begin{table}[t]
\caption{Attack success rate (\%) against evaluated defenses. $\epsilon = 8/255$ for $L_\infty$ attacks. Higher is better (stronger attack).}
\label{tab:main_results}
\centering
\small
\begin{tabular}{l|ccc|c}
\toprule
\textbf{Defense} & \textbf{PGD} & \textbf{AA} & \textbf{Ours} & \textbf{Claimed RA} \\
\midrule
"""
    domains = {
        "vision": "Content Moderation",
        "malware": "Malware Detection",
        "av_perception": "AV Perception",
    }
    for domain, domain_name in domains.items():
        latex += f"\\multicolumn{{5}}{{l}}{{\\textit{{{domain_name}}}}} \\\\\n"
        domain_defenses = [(n, m) for n, m in results.items() if m.domain == domain]
        for _name, m in domain_defenses:
            latex += (
                f"{m.defense_name} & "
                f"{m.pgd_asr*100:.1f} & "
                f"{m.autoattack_asr*100:.1f} & "
                f"\\textbf{{{m.neurinspectre_asr*100:.1f}}} & "
                f"{m.claimed_robust_acc*100:.0f} \\\\\n"
            )
    all_m = list(results.values())
    latex += r"\midrule" + "\n"
    if all_m:
        latex += (
            f"\\textbf{{Average}} & "
            f"{np.mean([m.pgd_asr for m in all_m])*100:.1f} & "
            f"{np.mean([m.autoattack_asr for m in all_m])*100:.1f} & "
            f"\\textbf{{{np.mean([m.neurinspectre_asr for m in all_m])*100:.1f}}} & "
            f"{np.mean([m.claimed_robust_acc for m in all_m])*100:.1f} \\\\\n"
        )
    else:
        latex += "\\textbf{Average} & 0.0 & 0.0 & \\textbf{0.0} & 0.0 \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(latex)
        logger.info("LaTeX Table 1 saved to %s", output_path)
    return latex
