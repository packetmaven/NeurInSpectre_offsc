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
    "evaluate",
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
    type=click.Choice(["cifar10", "cifar100", "imagenet", "custom"]),
    help="Dataset name",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="Path to custom dataset (required if dataset=custom)",
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
    type=click.Choice(["cifar10", "cifar100", "imagenet", "custom"]),
    help="Dataset name",
)
@click.option("--data-path", type=click.Path(exists=True), help="Path to custom dataset")
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

        # Paper baseline comparison
        neurinspectre compare --mode baseline eval_results/summary.json

        # Characterization signal comparison
        neurinspectre compare --mode characterization char_results/*.json --sort-by alpha

    Cross-ref: Paper Section 5.2.2 "Baseline comparison"
    Cross-ref: Paper Table 1 "Attack success rate against evaluated defenses"
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
            from .__main__ import main as legacy_main

            legacy_main()
            return
        cli(obj={})
    except Exception as e:  # pragma: no cover - CLI error handling
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
