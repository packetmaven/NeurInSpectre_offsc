"""
NeurInSpectre CLI - Command-line interface for adaptive adversarial evaluation.

Usage:
    neurinspectre attack --model resnet50.pth --defense jpeg --epsilon 0.03
    neurinspectre characterize --model resnet50.pth --defense jpeg
    neurinspectre evaluate --config eval.yaml

Cross-ref: Paper Section 4 "Implementation"
Cross-ref: WOOT 2026 submission

Author: [Redacted]
Date: 2026-02-05
Version: 2.0
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

try:
    import rich_click as click
except Exception:  # pragma: no cover - fallback
    import click

logger = logging.getLogger(__name__)

_CLICK_COMMANDS = {
    "attack",
    "characterize",
    "defense-analyzer",
    "doctor",
    "evaluate",
    "table2",
    "table2-smoke",
    "compare",
    "config",
}


@click.group()
@click.version_option(version="2.0.0", prog_name="neurinspectre")
@click.option("--verbose", "-v", count=True, help="Increase verbosity (-v, -vv, -vvv)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output (errors only)")
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: bool) -> None:
    """
    NeurInSpectre - Adaptive Adversarial Attack Framework

    Automatically characterizes gradient obfuscation defenses and
    synthesizes adaptive attacks to bypass them.

    \b
    Examples:
        # Run adaptive attack
        neurinspectre attack --model model.pth --defense jpeg --epsilon 0.03

        # Characterize defense
        neurinspectre characterize --model model.pth --defense jpeg

        # Full evaluation suite (Paper Section 5)
        neurinspectre evaluate --config evaluation.yaml

    Cross-ref: Paper Section 3 "NEURINSPECTRE Framework"
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif verbose == 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose >= 3:
        logging.getLogger("neurinspectre").setLevel(logging.DEBUG)
        logging.getLogger("torch").setLevel(logging.DEBUG)

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command("attack")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to model file (.pth, .pt, or .onnx)",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Choice(
        ["cifar10", "cifar100", "imagenet", "imagenet100", "ember", "nuscenes", "custom"]
    ),
    help="Dataset name",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="Path to custom dataset (required if dataset=custom)",
)
@click.option(
    "--labels-path",
    type=click.Path(exists=True),
    help="Path to nuScenes label map JSON (required for dataset=nuscenes)",
)
@click.option(
    "--nuscenes-version",
    type=str,
    default="v1.0-mini",
    show_default=True,
    help="nuScenes version string (e.g., v1.0-mini, v1.0-trainval)",
)
@click.option(
    "--defense",
    type=click.Choice(
        [
            "none",
            "jpeg",
            "bitdepth",
            "randsmooth",
            "thermometer",
            "distillation",
            "ensemble",
            "feature_squeezing",
            "gradient_regularization",
            "at_transform",
            "spatial_smoothing",
            "random_pad_crop",
            "certified_defense",
            "custom",
        ]
    ),
    default="none",
    help="Defense mechanism",
)
@click.option(
    "--defense-config",
    type=click.Path(exists=True),
    help="Path to defense configuration YAML",
)
@click.option(
    "--attack-type",
    type=click.Choice(
        ["neurinspectre", "pgd", "apgd", "autoattack", "square", "fab", "bpda", "eot"]
    ),
    default="neurinspectre",
    help="Attack algorithm (default: neurinspectre adaptive)",
)
@click.option(
    "--epsilon",
    "-e",
    type=float,
    default=8 / 255,
    help="Perturbation budget (default: 8/255 for Linf)",
)
@click.option(
    "--norm",
    type=click.Choice(["Linf", "L2", "L1"]),
    default="Linf",
    help="Lp norm for perturbation budget",
)
@click.option(
    "--iterations",
    "-n",
    type=int,
    default=100,
    help="Number of attack iterations",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=128,
    help="Batch size for evaluation",
)
@click.option(
    "--num-samples",
    type=int,
    default=1000,
    help="Number of samples to attack",
)
@click.option(
    "--targeted/--untargeted",
    default=False,
    help="Targeted vs untargeted attack",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="attack_results.json",
    help="Output file for results",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics (scan-friendly output)",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--save-adversarials",
    type=click.Path(),
    help="Save adversarial examples to directory",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="cuda",
    help="Device for computation",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.pass_context
def attack_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Run adaptive adversarial attack against defended model.

    This command implements Paper Algorithm 2 "NEURINSPECTRE Adaptive Attack".

    \b
    Examples:
        # Basic adaptive attack
        neurinspectre attack -m model.pth -d cifar10 --defense jpeg -e 0.03

        # Full configuration
        neurinspectre attack \\
            --model resnet50.pth \\
            --dataset imagenet \\
            --defense randsmooth \\
            --defense-config smooth_config.yaml \\
            --epsilon 0.5 --norm L2 \\
            --iterations 100 \\
            --output results.json

        # Compare multiple attacks
        for attack in neurinspectre pgd apgd autoattack; do
            neurinspectre attack -m model.pth -d cifar10 \\
                --attack-type $attack -o ${attack}_results.json
        done

    Cross-ref: Paper Section 3.2 "Phase 2: Adaptive Attack Synthesis"
    Cross-ref: Paper Table 1 "Attack success rate against evaluated defenses"
    """
    from .attack_cmd import run_attack

    run_attack(ctx, **kwargs)


@cli.command("characterize")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to model file",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Choice(
        ["cifar10", "cifar100", "imagenet", "imagenet100", "ember", "nuscenes", "custom"]
    ),
    help="Dataset name",
)
@click.option("--data-path", type=click.Path(exists=True), help="Path to custom dataset")
@click.option(
    "--labels-path",
    type=click.Path(exists=True),
    help="Path to nuScenes label map JSON (required for dataset=nuscenes)",
)
@click.option(
    "--nuscenes-version",
    type=str,
    default="v1.0-mini",
    show_default=True,
    help="nuScenes version string (e.g., v1.0-mini, v1.0-trainval)",
)
@click.option(
    "--defense",
    type=click.Choice(
        [
            "none",
            "jpeg",
            "bitdepth",
            "randsmooth",
            "thermometer",
            "distillation",
            "ensemble",
            "feature_squeezing",
            "gradient_regularization",
            "at_transform",
            "spatial_smoothing",
            "random_pad_crop",
            "certified_defense",
            "custom",
        ]
    ),
    default="none",
    help="Defense mechanism to characterize",
)
@click.option(
    "--defense-config",
    type=click.Path(exists=True),
    help="Path to defense configuration",
)
@click.option(
    "--use-bpda-approx",
    is_flag=True,
    help="Use BPDA approximation during characterization (for non-differentiable defenses)",
)
@click.option(
    "--krylov-order",
    "-k",
    type=int,
    default=20,
    help="Krylov subspace order (default: 20)",
)
@click.option(
    "--num-samples",
    type=int,
    default=100,
    help="Number of samples for characterization",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="characterization.json",
    help="Output file for characterization results",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualizations (eigenvalue spectrum, etc.)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="cuda",
    help="Device for computation",
)
@click.pass_context
def characterize_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Characterize defense obfuscation type.

    Implements Paper Algorithm 1 "Spectral Gradient Analysis".
    Identifies shattered/stochastic/vanishing/exploding gradients.

    \b
    Examples:
        # Characterize defense
        neurinspectre characterize -m model.pth -d cifar10 --defense jpeg

        # With visualization
        neurinspectre characterize -m model.pth -d cifar10 \\
            --defense randsmooth --visualize

        # Custom Krylov order
        neurinspectre characterize -m model.pth -d cifar10 \\
            --defense thermometer -k 50

    Cross-ref: Paper Section 3.1 "Phase 1: Defense Characterization"
    Cross-ref: Paper Equation 9 "Krylov subspace"
    """
    from .characterize_cmd import run_characterization

    run_characterization(ctx, **kwargs)


cli.add_command(characterize_cmd, name="defense-analyzer")


@cli.command("compare")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["attacks", "defenses", "runs", "baseline", "characterization"]),
    default="attacks",
    help="Comparison mode",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.argument("input_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON path",
)
@click.option(
    "--sort-by",
    default="asr",
    type=click.Choice(["asr", "confidence", "etd", "alpha", "variance"]),
    help="Sort key for comparison outputs",
)
@click.option(
    "--threshold",
    type=float,
    default=2.0,
    help="Significance threshold in percentage points",
)
@click.option(
    "--expected-asr-path",
    "--expected-asr",
    type=click.Path(exists=True),
    help="Expected ASR baseline YAML/JSON for --mode baseline (kept out of repo)",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.pass_context
def compare_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Compare attacks, defenses, runs, or baselines side-by-side.

    \b
    Examples:
        # Attack comparison (Table 1 style)
        neurinspectre compare --mode attacks eval_results/summary.json

        # Defense ranking per attack
        neurinspectre compare --mode defenses eval_results/summary.json

        # Regression comparison between runs
        neurinspectre compare --mode runs run_a/summary.json run_b/summary.json --threshold 3.0

        # External baseline comparison (expected ASR kept out of repo)
        neurinspectre compare --mode baseline eval_results/summary.json \\
            --expected-asr-path /path/to/expected_asr.yaml

        # Characterization signal comparison
        neurinspectre compare --mode characterization char_results/*.json --sort-by alpha

    """
    from .compare_cmd import run_compare

    run_compare(ctx, **kwargs)


@cli.command("evaluate")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="YAML configuration file for evaluation",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="evaluation_results",
    help="Output directory for all results",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--defenses",
    multiple=True,
    help="Specific defenses to evaluate (default: all in config)",
)
@click.option(
    "--attacks",
    multiple=True,
    help="Specific attacks to run (default: all)",
)
@click.option(
    "--parallel",
    "-j",
    type=int,
    default=1,
    help="Number of parallel workers",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume interrupted evaluation",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="cuda",
    help="Device for computation",
)
@click.pass_context
def evaluate_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Run full evaluation suite (Paper Section 5).

    Evaluates multiple defenses with ensemble of attacks:
    - APGD-CE
    - APGD-DLR
    - NEURINSPECTRE Adaptive
    - Square Attack (query-based)

    \b
    Example config (evaluation.yaml):
        defenses:
          - name: jpeg_compression
            type: jpeg
            quality: 75
          - name: randomized_smoothing
            type: randsmooth
            sigma: 0.25

        attacks:
          - neurinspectre
          - apgd
          - autoattack

        datasets:
          cifar10:
            path: ./data/cifar10
            num_samples: 1000

        perturbation:
          epsilon: 0.03
          norm: Linf

    \b
    Examples:
        # Full evaluation
        neurinspectre evaluate --config evaluation.yaml

        # Specific defenses only
        neurinspectre evaluate -c eval.yaml --defenses jpeg randsmooth

        # Parallel execution
        neurinspectre evaluate -c eval.yaml -j 4

        # Resume interrupted
        neurinspectre evaluate -c eval.yaml --resume

    Cross-ref: Paper Section 5 "Evaluation"
    Cross-ref: Paper Table 1 "Attack success rate against evaluated defenses"
    """
    from .evaluate_cmd import run_evaluation

    run_evaluation(ctx, **kwargs)


@cli.command("table2")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="YAML configuration file for Table 2 evaluation",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results/table2",
    help="Output directory for all results",
)
@click.option(
    "--strict-real-data/--no-strict-real-data",
    default=True,
    help="Require only real datasets and local assets",
)
@click.option(
    "--strict-dataset-budgets/--no-strict-dataset-budgets",
    default=True,
    help="Require and enforce dataset-specific attack budgets",
)
@click.option(
    "--allow-missing",
    is_flag=True,
    help="Allow missing local assets in strict checks",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--defenses",
    multiple=True,
    help="Specific defenses to evaluate (default: all in config)",
)
@click.option(
    "--attacks",
    multiple=True,
    help="Specific attacks to run (default: all)",
)
@click.option(
    "--parallel",
    "-j",
    type=int,
    default=1,
    help="Number of parallel workers",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume interrupted evaluation",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="auto",
    help="Device for computation",
)
@click.pass_context
def table2_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Run Table 2 pipeline with strict real-data checks.

    This command normalizes Table 2 config variants, enforces real-dataset
    constraints, and then executes the standard evaluation matrix runner.
    """
    from .table2_cmd import run_table2

    run_table2(ctx, **kwargs)


@cli.command("table2-smoke")
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results/table2_smoke_real",
    help="Output directory for all results",
)
@click.option(
    "--data-root",
    type=click.Path(),
    default="./data",
    help="Base directory used to discover datasets",
)
@click.option(
    "--models-root",
    type=click.Path(),
    default="./models",
    help="Base directory used to discover model artifacts",
)
@click.option(
    "--pgd-steps",
    type=int,
    default=10,
    show_default=True,
    help="PGD steps for smoke run",
)
@click.option(
    "--neurinspectre-steps",
    type=int,
    default=10,
    show_default=True,
    help="NeurInSpectre attack steps for smoke run",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--defenses",
    multiple=True,
    help="Specific defenses to evaluate (default: all discovered)",
)
@click.option(
    "--attacks",
    multiple=True,
    help="Specific attacks to run (default: all)",
)
@click.option(
    "--parallel",
    "-j",
    type=int,
    default=1,
    help="Number of parallel workers",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume interrupted evaluation",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="auto",
    help="Device for computation",
)
@click.pass_context
def table2_smoke_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Run a small, real-data Table2 smoke matrix.

    Discovers which datasets/models are available locally, generates a minimal
    config, then runs the standard `table2` pipeline with strict real-data
    checks (validity + integrity gates enabled).
    """

    from .table2_smoke_cmd import run_table2_smoke

    run_table2_smoke(ctx, **kwargs)


@cli.command("doctor")
@click.option("--json-output", type=click.Path(), help="Write environment report JSON to path")
@click.option("--as-json", is_flag=True, help="Print environment report JSON to stdout")
@click.option(
    "--models-dir",
    type=click.Path(),
    default="models",
    show_default=True,
    help="Models directory to scan for stub metadata",
)
@click.option("--check-models/--no-check-models", default=True, help="Scan models dir for stub markers")
@click.pass_context
def doctor_cli_cmd(ctx: click.Context, **kwargs) -> None:
    """Environment + dependency sanity checks (no network)."""
    from .doctor_cmd import run_doctor

    run_doctor(ctx, **kwargs)


@cli.command("config")
@click.argument("config_type", type=click.Choice(["attack", "defense", "evaluation"]))
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
def config_cmd(config_type: str, output: str | None) -> None:
    """
    Generate example configuration files.

    \b
    Examples:
        # Generate attack config
        neurinspectre config attack > attack.yaml

        # Generate full evaluation config
        neurinspectre config evaluation -o evaluation.yaml
    """
    from .config import generate_example_config

    config_str = generate_example_config(config_type)

    if output:
        Path(output).write_text(config_str)
        click.echo(f"Configuration written to {output}")
    else:
        click.echo(config_str)


def main() -> None:
    """Main CLI entry point"""
    try:
        argv = sys.argv[1:]
        if argv and argv[0] not in _CLICK_COMMANDS and argv[0] not in {"-h", "--help", "--version"}:
            enable_legacy = str(os.environ.get("NEURINSPECTRE_ENABLE_LEGACY_CLI", "")).lower() in {
                "1",
                "true",
                "yes",
            }
            if enable_legacy:
                from .__main__ import main as legacy_main

                legacy_main()
                return
            print(
                f"[ERROR] Unknown command: {argv[0]!r}\n"
                "Legacy CLI fallback is disabled by default.\n"
                "Run `neurinspectre --help` to see supported commands.",
                file=sys.stderr,
            )
            sys.exit(2)
        cli(obj={})
    except Exception as e:  # pragma: no cover - CLI error handling
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
