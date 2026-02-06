"""
Table 1 reproduction runner using real datasets and models.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import torch
import yaml

from ..attacks import AttackFactory
from ..characterization.defense_analyzer import DefenseAnalyzer
from ..defenses import DefenseFactory
from ..evaluation.cache import DatasetCache, AttackCheckpoint
from ..evaluation.metrics import (
    compute_attack_success_rate,
    compute_confidence_interval,
    compute_query_efficiency,
    compute_robust_accuracy,
)
from ..evaluation.datasets import DatasetFactory
from ..models import ModelFactory


@dataclass
class DefenseResult:
    defense_name: str
    domain: str
    dataset: str
    model: str
    clean_accuracy: float
    pgd_asr: float
    autoattack_asr: float
    neurinspectre_asr: float
    claimed_robust_accuracy: float
    n_samples_tested: int
    obfuscation_types: List[str]
    requires_bpda: bool
    requires_eot: bool
    requires_mapgd: bool
    pgd_robust_accuracy: float = 0.0
    autoattack_robust_accuracy: float = 0.0
    neurinspectre_robust_accuracy: float = 0.0
    pgd_asr_ci: Tuple[float, float] = (0.0, 0.0)
    autoattack_asr_ci: Tuple[float, float] = (0.0, 0.0)
    neurinspectre_asr_ci: Tuple[float, float] = (0.0, 0.0)
    attack_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Table1Reproducer:
    """
    Reproduce Table 1 using real datasets and real model checkpoints.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        results_dir: str = "./results/table1",
        allow_missing: bool = False,
    ):
        if config is None:
            if not config_path:
                raise ValueError("config_path is required when config is not provided")
            self.config_path = Path(config_path)
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config not found: {self.config_path}")
            self.config = self._load_config()
        else:
            self.config_path = Path(config_path) if config_path else None
            self.config = config
        self.device = self._resolve_device(device)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.allow_missing = bool(allow_missing)

        self.results: List[DefenseResult] = []
        self.dataset_cache = self._init_dataset_cache()
        self.attack_checkpoint = self._init_attack_checkpoint()

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path is None:
            raise ValueError("config_path not set")
        with open(self.config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _resolve_device(self, device: str) -> str:
        device = str(device)
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return "cpu"
        return device

    def _init_dataset_cache(self) -> Optional[DatasetCache]:
        cache_cfg = self.config.get("cache", {})
        if not cache_cfg.get("enable_dataset_cache", False):
            return None
        cache_dir = Path(cache_cfg.get("cache_dir", "./cache")) / "datasets"
        return DatasetCache(
            cache_dir=cache_dir,
            max_memory_gb=float(cache_cfg.get("dataset_cache_gb", 16.0)),
            eviction_threshold=float(cache_cfg.get("dataset_cache_threshold", 0.9)),
        )

    def _init_attack_checkpoint(self) -> Optional[AttackCheckpoint]:
        cache_cfg = self.config.get("cache", {})
        if not cache_cfg.get("enable_attack_checkpoint", False):
            return None
        checkpoint_dir = Path(cache_cfg.get("checkpoint_dir", "./checkpoints/table1"))
        return AttackCheckpoint(checkpoint_dir=checkpoint_dir)

    def run_evaluation(self) -> List[DefenseResult]:
        defenses = self.config.get("defenses", {})
        if not defenses:
            raise ValueError("Config missing defenses section.")

        for defense_name, defense_cfg in defenses.items():
            try:
                result = self._evaluate_defense(defense_name, defense_cfg)
                self.results.append(result)
            except Exception as exc:
                if self.allow_missing:
                    print(f"[WARN] Skipping {defense_name}: {exc}")
                    continue
                raise

        return self.results

    def _evaluate_defense(self, defense_name: str, defense_cfg: Dict[str, Any]) -> DefenseResult:
        self._set_seed(int(self.config.get("seed", 42)))

        dataset_name = str(defense_cfg["dataset"])
        domain = str(defense_cfg.get("domain", "unknown"))

        loader, x_test, y_test = self._load_dataset(dataset_name)
        model_cfg = defense_cfg.get("model", {})
        model = self._load_model(model_cfg, dataset_name, domain)
        defended_model = DefenseFactory.create_defense(
            defense_name=defense_name,
            base_model=model,
            params=defense_cfg.get("params", {}),
        )

        x_correct, y_correct, clean_acc = self._collect_correct_samples(defended_model, loader)
        n_correct = int(x_correct.size(0))
        if n_correct == 0:
            return DefenseResult(
                defense_name=defense_name,
                domain=domain,
                dataset=dataset_name,
                model=self._model_label(model_cfg),
                clean_accuracy=float(clean_acc),
                pgd_asr=0.0,
                autoattack_asr=0.0,
                neurinspectre_asr=0.0,
                pgd_robust_accuracy=0.0,
                autoattack_robust_accuracy=0.0,
                neurinspectre_robust_accuracy=0.0,
                pgd_asr_ci=(0.0, 0.0),
                autoattack_asr_ci=(0.0, 0.0),
                neurinspectre_asr_ci=(0.0, 0.0),
                claimed_robust_accuracy=float(defense_cfg.get("claimed_robust_accuracy", 0.0)),
                n_samples_tested=0,
                obfuscation_types=[],
                requires_bpda=False,
                requires_eot=False,
                requires_mapgd=False,
                attack_metrics={},
            )

        characterization = self._characterize_defense(defended_model, loader)

        attacks_cfg = self.config.get("attacks", {})
        pgd_summary = self._run_attack("pgd", defended_model, x_correct, y_correct, attacks_cfg.get("pgd", {}))
        aa_summary = self._run_attack(
            "autoattack", defended_model, x_correct, y_correct, attacks_cfg.get("autoattack", {})
        )
        nis_summary = self._run_attack(
            "neurinspectre",
            defended_model,
            x_correct,
            y_correct,
            attacks_cfg.get("neurinspectre", {}),
            characterization=characterization,
            loader=loader,
        )

        return DefenseResult(
            defense_name=defense_name,
            domain=domain,
            dataset=dataset_name,
            model=self._model_label(model_cfg),
            clean_accuracy=float(clean_acc),
            pgd_asr=float(pgd_summary["asr"]),
            autoattack_asr=float(aa_summary["asr"]),
            neurinspectre_asr=float(nis_summary["asr"]),
            pgd_robust_accuracy=float(pgd_summary["robust_accuracy"]),
            autoattack_robust_accuracy=float(aa_summary["robust_accuracy"]),
            neurinspectre_robust_accuracy=float(nis_summary["robust_accuracy"]),
            pgd_asr_ci=tuple(pgd_summary.get("asr_ci", (0.0, 0.0))),
            autoattack_asr_ci=tuple(aa_summary.get("asr_ci", (0.0, 0.0))),
            neurinspectre_asr_ci=tuple(nis_summary.get("asr_ci", (0.0, 0.0))),
            claimed_robust_accuracy=float(defense_cfg.get("claimed_robust_accuracy", 0.0)),
            n_samples_tested=n_correct,
            obfuscation_types=[t.value for t in characterization.obfuscation_types],
            requires_bpda=bool(characterization.requires_bpda),
            requires_eot=bool(characterization.requires_eot),
            requires_mapgd=bool(characterization.requires_mapgd),
            attack_metrics={
                "pgd": pgd_summary,
                "autoattack": aa_summary,
                "neurinspectre": nis_summary,
            },
        )

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_dataset(self, dataset_name: str):
        dataset_cfg = self.config.get("datasets", {}).get(dataset_name, {})
        cache_key = self._dataset_cache_key(dataset_name, dataset_cfg)
        if self.dataset_cache and self.dataset_cache.has(cache_key):
            cached = self.dataset_cache.get(cache_key)
            if cached is not None:
                return cached

        kwargs = dict(dataset_cfg)
        # Avoid pin_memory warnings on non-CUDA devices.
        if "pin_memory" not in kwargs:
            kwargs["pin_memory"] = bool(self.device == "cuda" and torch.cuda.is_available())
        loader, x_test, y_test = DatasetFactory.get_dataset(dataset_name, **kwargs)
        if self.dataset_cache:
            self.dataset_cache.put(cache_key, (loader, x_test, y_test))
        return loader, x_test, y_test

    def _dataset_cache_key(self, name: str, cfg: Dict[str, Any]) -> str:
        key_parts = [name]
        for k in sorted(cfg.keys()):
            key_parts.append(f"{k}={cfg[k]}")
        return "|".join(key_parts)

    def _load_model(self, model_cfg: Any, dataset_name: str, domain: str):
        if isinstance(model_cfg, str):
            model_cfg = {"model_name": model_cfg}
        model_name = model_cfg.get("model_name") or model_cfg.get("architecture") or model_cfg.get("name")
        training_type = model_cfg.get("training_type", "standard")
        extra_cfg = dict(model_cfg)
        # Avoid passing duplicate keywords that are already provided explicitly.
        for key in ("model_name", "architecture", "name", "training_type", "dataset", "domain", "device"):
            extra_cfg.pop(key, None)
        return ModelFactory.load_model(
            domain=domain,
            model_name=model_name,
            training_type=training_type,
            dataset=model_cfg.get("dataset", dataset_name),
            device=self.device,
            **extra_cfg,
        )

    def _model_label(self, model_cfg: Any) -> str:
        if isinstance(model_cfg, str):
            return model_cfg
        return str(model_cfg.get("model_name") or model_cfg.get("architecture") or model_cfg.get("name"))

    def _collect_correct_samples(self, model, loader):
        model.eval()
        total = 0
        correct_total = 0
        x_correct = []
        y_correct = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            with torch.no_grad():
                preds = model(x_batch).argmax(1)
            mask = preds == y_batch
            total += int(x_batch.size(0))
            correct_total += int(mask.sum().item())
            if mask.any():
                x_correct.append(x_batch[mask])
                y_correct.append(y_batch[mask])
        if x_correct:
            x_correct = torch.cat(x_correct, dim=0)
            y_correct = torch.cat(y_correct, dim=0)
        else:
            x_correct = torch.empty((0,), device=self.device)
            y_correct = torch.empty((0,), dtype=torch.long, device=self.device)
        clean_acc = correct_total / total if total > 0 else 0.0
        return x_correct, y_correct, clean_acc

    def _characterize_defense(self, model, loader):
        attack_cfg = self.config.get("attacks", {}).get("neurinspectre", {})
        n_samples = int(attack_cfg.get("characterization_samples", 50))
        analyzer = DefenseAnalyzer(model, n_samples=n_samples, device=self.device, verbose=False)
        return analyzer.characterize(loader, eps=float(attack_cfg.get("eps", 8 / 255)))

    def _run_attack(
        self,
        attack_type: str,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        cfg: Dict[str, Any],
        *,
        characterization=None,
        loader=None,
    ) -> Dict[str, Any]:
        checkpoint_key = f"{attack_type}:{self._attack_config_signature(cfg)}"
        if self.attack_checkpoint and self.attack_checkpoint.exists(checkpoint_key):
            saved = self.attack_checkpoint.load(checkpoint_key)
            preds = saved.get("predictions")
            success = saved.get("success_mask")
            summary = saved.get("summary")
            if summary is not None:
                return summary
            y_cpu = y.detach().cpu()
            asr = compute_attack_success_rate(preds, y_cpu, success_mask=success)
            robust_acc = compute_robust_accuracy(preds, y_cpu)
            asr_ci = (0.0, 0.0)
            if success is not None and success.numel() > 1:
                mean, lower, upper = compute_confidence_interval(success.float())
                asr_ci = (lower, upper)
            return {
                "asr": float(asr),
                "robust_accuracy": float(robust_acc),
                "asr_ci": asr_ci,
                "perturbation": {},
                "query_efficiency": {},
                "n_samples": int(y_cpu.numel()),
            }

        runner = AttackFactory.create_attack(
            attack_type,
            model,
            config=cfg,
            characterization=characterization,
            characterization_loader=loader,
            defense=model,
            device=self.device,
        )

        batch_size = int(self.config.get("attack_batch_size", 50))
        total = int(x.size(0))
        all_preds = []
        all_success = []
        linf_vals = []
        l2_vals = []
        l1_vals = []
        l0_vals = []
        query_counts = []
        for i in range(0, total, batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            res = runner.run(x_batch, y_batch)
            all_preds.append(res.predictions.detach().cpu())
            all_success.append(res.success_mask.detach().cpu())
            if res.x_adv is not None:
                delta = (res.x_adv.detach() - x_batch).view(x_batch.size(0), -1)
                linf_vals.append(delta.abs().max(dim=1)[0].cpu().numpy())
                l2_vals.append(delta.norm(p=2, dim=1).cpu().numpy())
                l1_vals.append(delta.norm(p=1, dim=1).cpu().numpy())
                l0_vals.append((delta.abs() > 1e-8).float().sum(dim=1).cpu().numpy())
            if isinstance(res.metadata, dict) and res.metadata.get("queries_used") is not None:
                queries = res.metadata.get("queries_used")
                if isinstance(queries, np.ndarray):
                    query_counts.append(queries)
                else:
                    query_counts.append(np.array(queries))

        preds = torch.cat(all_preds, dim=0)
        success_mask = torch.cat(all_success, dim=0)
        y_cpu = y.detach().cpu()
        asr = compute_attack_success_rate(preds, y_cpu, success_mask=success_mask)
        robust_acc = compute_robust_accuracy(preds, y_cpu)
        asr_ci = (0.0, 0.0)
        if success_mask.numel() > 1:
            mean, lower, upper = compute_confidence_interval(success_mask.float())
            asr_ci = (lower, upper)

        perturbation = {}
        if linf_vals:
            linf = np.concatenate(linf_vals)
            l2 = np.concatenate(l2_vals) if l2_vals else np.array([])
            l1 = np.concatenate(l1_vals) if l1_vals else np.array([])
            l0 = np.concatenate(l0_vals) if l0_vals else np.array([])
            perturbation = {
                "linf_mean": float(linf.mean()),
                "linf_max": float(linf.max()),
                "linf_min": float(linf.min()),
                "linf_std": float(linf.std()),
                "l2_mean": float(l2.mean()) if l2.size > 0 else 0.0,
                "l2_max": float(l2.max()) if l2.size > 0 else 0.0,
                "l2_min": float(l2.min()) if l2.size > 0 else 0.0,
                "l2_std": float(l2.std()) if l2.size > 0 else 0.0,
                "l1_mean": float(l1.mean()) if l1.size > 0 else 0.0,
                "l0_mean": float(l0.mean()) if l0.size > 0 else 0.0,
            }

        query_efficiency = {}
        if query_counts:
            queries_flat = np.concatenate(query_counts).astype(int).tolist()
            query_efficiency = compute_query_efficiency(queries_flat, success_mask)

        summary = {
            "asr": float(asr),
            "robust_accuracy": float(robust_acc),
            "asr_ci": asr_ci,
            "perturbation": perturbation,
            "query_efficiency": query_efficiency,
            "n_samples": int(total),
        }
        if self.attack_checkpoint:
            self.attack_checkpoint.save(
                checkpoint_key,
                {
                    "predictions": preds,
                    "success_mask": success_mask,
                    "summary": summary,
                },
            )
        return summary

    def _attack_config_signature(self, cfg: Dict[str, Any]) -> str:
        parts = []
        for k in sorted(cfg.keys()):
            parts.append(f"{k}={cfg[k]}")
        return "|".join(parts)

    def save_results(self, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(self._results_to_csv())

        json_path = output_path.with_suffix(".json")
        payload = {"results": [r.to_dict() for r in self.results]}
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def plot_results(self, output_path: str) -> None:
        import matplotlib.pyplot as plt

        defenses = [r.defense_name for r in self.results]
        pgd_asrs = [r.pgd_asr * 100 for r in self.results]
        aa_asrs = [r.autoattack_asr * 100 for r in self.results]
        ours_asrs = [r.neurinspectre_asr * 100 for r in self.results]

        x = np.arange(len(defenses))
        width = 0.25
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width, pgd_asrs, width, label="PGD", color="lightcoral")
        ax.bar(x, aa_asrs, width, label="AutoAttack", color="lightskyblue")
        ax.bar(x + width, ours_asrs, width, label="NeurInSpectre (Ours)", color="lightgreen")
        ax.set_xlabel("Defense")
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title("Table 1 Reproduction: ASR Across 12 Defenses")
        ax.set_xticks(x)
        ax.set_xticklabels(defenses, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def print_table(self) -> None:
        if not self.results:
            return
        print("\nTable 1: Attack Success Rates")
        print(f"{'Defense':<30} {'PGD':>8} {'AA':>8} {'Ours':>8} {'Claimed':>10}")
        for r in self.results:
            print(
                f"{r.defense_name:<30} "
                f"{r.pgd_asr*100:>7.1f}% "
                f"{r.autoattack_asr*100:>7.1f}% "
                f"{r.neurinspectre_asr*100:>7.1f}% "
                f"{r.claimed_robust_accuracy*100:>9.0f}%"
            )

    def _results_to_csv(self) -> str:
        header = [
            "defense_name",
            "domain",
            "dataset",
            "model",
            "clean_accuracy",
            "pgd_asr",
            "autoattack_asr",
            "neurinspectre_asr",
            "pgd_robust_accuracy",
            "autoattack_robust_accuracy",
            "neurinspectre_robust_accuracy",
            "pgd_asr_ci_lower",
            "pgd_asr_ci_upper",
            "autoattack_asr_ci_lower",
            "autoattack_asr_ci_upper",
            "neurinspectre_asr_ci_lower",
            "neurinspectre_asr_ci_upper",
            "claimed_robust_accuracy",
            "n_samples_tested",
            "obfuscation_types",
            "requires_bpda",
            "requires_eot",
            "requires_mapgd",
        ]
        rows = [",".join(header)]
        for r in self.results:
            rows.append(
                ",".join(
                    [
                        r.defense_name,
                        r.domain,
                        r.dataset,
                        r.model,
                        f"{r.clean_accuracy:.6f}",
                        f"{r.pgd_asr:.6f}",
                        f"{r.autoattack_asr:.6f}",
                        f"{r.neurinspectre_asr:.6f}",
                                f"{r.pgd_robust_accuracy:.6f}",
                                f"{r.autoattack_robust_accuracy:.6f}",
                                f"{r.neurinspectre_robust_accuracy:.6f}",
                                f"{r.pgd_asr_ci[0]:.6f}",
                                f"{r.pgd_asr_ci[1]:.6f}",
                                f"{r.autoattack_asr_ci[0]:.6f}",
                                f"{r.autoattack_asr_ci[1]:.6f}",
                                f"{r.neurinspectre_asr_ci[0]:.6f}",
                                f"{r.neurinspectre_asr_ci[1]:.6f}",
                        f"{r.claimed_robust_accuracy:.6f}",
                        str(r.n_samples_tested),
                        "|".join(r.obfuscation_types),
                        str(r.requires_bpda),
                        str(r.requires_eot),
                        str(r.requires_mapgd),
                    ]
                )
            )
        return "\n".join(rows)
