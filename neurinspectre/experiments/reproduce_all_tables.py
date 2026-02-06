"""
Master script to reproduce Table 1 plus all paper-claim validations.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict
import json

from .reproduce_table1 import Table1Reproducer
from .reproduce_paper_claims import PaperClaimValidator


class ComprehensiveReproduction:
    """
    Reproduces Table 1 and validates all other paper claims.

    Args:
        device: Computation device
        n_seeds: Number of random seeds for validations
        results_dir: Directory to save outputs
        table1_config: Path to Table 1 YAML config
        allow_missing: Skip missing datasets/models
        fast_mode: Use short/fast configurations for claims
        synthetic_mode: Use synthetic data for validations
        plot: Enable plotting (Table 1 + figure)
        verbose: Print progress
    """

    def __init__(
        self,
        device: str = "cpu",
        n_seeds: int = 2,
        results_dir: str = "./comprehensive_results",
        table1_config: str = "experiments/configs/table1_config.yaml",
        allow_missing: bool = False,
        fast_mode: bool = True,
        synthetic_mode: bool = True,
        plot: bool = False,
        verbose: bool = True,
    ):
        self.device = str(device)
        self.n_seeds = int(n_seeds)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.table1_config = str(table1_config)
        self.allow_missing = bool(allow_missing)
        self.fast_mode = bool(fast_mode)
        self.synthetic_mode = bool(synthetic_mode)
        self.plot = bool(plot)
        self.verbose = bool(verbose)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.all_results: Dict[str, Dict] = {}

    def reproduce_everything(self) -> Dict[str, Dict]:
        if self.verbose:
            print("\n" + "=" * 80)
            print("COMPREHENSIVE NEURINSPECTRE REPRODUCTION")
            print("=" * 80)
            print(f"Results directory: {self.run_dir}")
            print(f"Device: {self.device}")
            print(f"Fast mode (claims): {self.fast_mode}")
            print(f"Synthetic mode (claims): {self.synthetic_mode}")
            print("=" * 80 + "\n")

        self.all_results["table1"] = self._reproduce_table1()
        self.all_results["claims"] = self._reproduce_claims()
        self._save_master_results()
        return self.all_results

    def _reproduce_table1(self) -> Dict:
        table1_dir = self.run_dir / "table1"
        reproducer = Table1Reproducer(
            config_path=self.table1_config,
            results_dir=str(table1_dir),
            device=self.device,
            allow_missing=self.allow_missing,
        )
        results = reproducer.run_evaluation()
        reproducer.print_table()
        if self.plot:
            reproducer.plot_results(str(table1_dir / "table1_results.png"))
        reproducer.save_results(str(table1_dir / "table1_results.csv"))
        avg_neurinspectre_asr = 0.0
        if results:
            avg_neurinspectre_asr = sum(r.neurinspectre_asr for r in results) / len(results)
        return {
            "completed": True,
            "n_defenses": len(results),
            "avg_neurinspectre_asr": avg_neurinspectre_asr,
        }

    def _reproduce_claims(self) -> Dict:
        validator = PaperClaimValidator(
            device=self.device,
            n_seeds=self.n_seeds,
            results_dir=str(self.run_dir / "claims"),
            verbose=self.verbose,
            fast_mode=self.fast_mode,
            synthetic_mode=self.synthetic_mode,
            plot=self.plot,
        )
        results = validator.validate_all()
        return {"completed": True, "results": results}

    def _save_master_results(self) -> None:
        master_file = self.run_dir / "master_results.json"
        payload = {
            "timestamp": self.timestamp,
            "device": self.device,
            "n_seeds": self.n_seeds,
            "fast_mode": self.fast_mode,
            "synthetic_mode": self.synthetic_mode,
            "results": self.all_results,
        }
        with open(master_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nMaster results saved to: {master_file}")

    def generate_report(self) -> None:
        report_path = self.run_dir / "report.md"
        table1 = self.all_results.get("table1", {})
        claims = self.all_results.get("claims", {}).get("results", {})

        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("# NeurInSpectre Reproduction Report\n\n")
            handle.write(f"**Generated:** {self.timestamp}\n\n")
            handle.write(f"**Device:** {self.device}\n\n")
            handle.write(f"**Fast mode:** {self.fast_mode}\n\n")
            handle.write(f"**Synthetic mode:** {self.synthetic_mode}\n\n")
            handle.write("---\n\n")

            handle.write("## Table 1: Main Results\n\n")
            if table1.get("completed"):
                handle.write("✅ **Completed**\n\n")
                handle.write(f"- Defenses evaluated: {table1.get('n_defenses', 0)}\n")
                avg_asr = float(table1.get("avg_neurinspectre_asr", 0.0))
                handle.write(f"- Avg NeurInSpectre ASR: {avg_asr * 100:.1f}%\n")
            else:
                handle.write("❌ **Not completed**\n\n")

            handle.write("\n## Claim Validations (Figure 1 / Tables 2-8)\n\n")
            if claims:
                for name, passed in claims.items():
                    status = "✅ PASSED" if passed else "❌ FAILED"
                    handle.write(f"- {name}: {status}\n")
            else:
                handle.write("No claim validation results found.\n")

        print(f"\nReport generated: {report_path}")


def reproduce_all_tables(
    device: str = "cpu",
    n_seeds: int = 2,
    table1_config: str = "experiments/configs/table1_config.yaml",
    allow_missing: bool = False,
    fast_mode: bool = True,
    synthetic_mode: bool = True,
    plot: bool = False,
) -> Dict[str, Dict]:
    """
    Convenience wrapper for the full reproduction suite.
    """
    reproduction = ComprehensiveReproduction(
        device=device,
        n_seeds=n_seeds,
        table1_config=table1_config,
        allow_missing=allow_missing,
        fast_mode=fast_mode,
        synthetic_mode=synthetic_mode,
        plot=plot,
        verbose=True,
    )
    results = reproduction.reproduce_everything()
    reproduction.generate_report()
    return results
