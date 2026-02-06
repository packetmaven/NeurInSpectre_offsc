"""
Rich CLI output formatting for NeurInSpectre offensive workflows.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def build_console(no_color: bool = False, force_color: bool = False) -> Console:
    if no_color:
        return Console(color_system=None)
    if force_color:
        return Console(color_system="truecolor", force_terminal=True)
    if os.getenv("NO_COLOR") or os.getenv("TERM", "").lower() == "dumb":
        return Console(color_system=None)
    return Console()


def _summary_panel(
    title: str,
    rows: Sequence[Tuple[str, str]],
    severity: Optional[Tuple[str, str]] = None,
) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(style="white")
    if severity:
        label, color = severity
        table.add_row("Severity", f"[bold {color}]{label}[/bold {color}]")
    for key, value in rows:
        table.add_row(str(key), str(value))
    border = severity[1] if severity else "cyan"
    return Panel(table, title=title, border_style=border, box=box.ROUNDED)


def _print_text_summary(
    console: Console,
    title: str,
    rows: Sequence[Tuple[str, str]],
    severity: Optional[Tuple[str, str]] = None,
) -> None:
    console.print(title)
    if severity:
        console.print(f"Severity: {severity[0]}")
    for key, value in rows:
        console.print(f"{key}: {value}")


def _risk_level(asr: float) -> Tuple[str, str]:
    if asr >= 0.7:
        return "CRITICAL", "red"
    if asr >= 0.5:
        return "HIGH", "bright_red"
    if asr >= 0.3:
        return "MEDIUM", "yellow"
    if asr >= 0.1:
        return "LOW", "blue"
    return "MINIMAL", "green"


def _asr_interpretation(asr: float) -> str:
    if asr >= 0.9:
        return "Defense appears ineffective against this attack"
    if asr >= 0.7:
        return "Defense provides weak protection"
    if asr >= 0.5:
        return "Defense provides moderate protection"
    if asr >= 0.3:
        return "Defense provides good protection"
    return "Defense appears robust against this attack"


def _obfuscation_severity(obf_types: Sequence[str], confidence: Optional[float]) -> Tuple[str, str]:
    if not obf_types:
        label = "NONE"
        color = "green"
    else:
        lowered = {t.lower() for t in obf_types}
        if "vanishing" in lowered or "shattered" in lowered:
            label = "SEVERE"
            color = "red"
        elif "stochastic" in lowered:
            label = "HIGH"
            color = "yellow"
        else:
            label = "ELEVATED"
            color = "yellow"
    if confidence is not None and float(confidence) < 0.4:
        label = f"{label} (LOW CONF)"
        color = "blue"
    return label, color


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _recommendation(asr: float, defense: str) -> str:
    defense_name = str(defense)
    no_defense = defense_name.lower() in {"none", "no_defense", "nodefense", "baseline"}
    note = ""
    if no_defense:
        note = "NOTE: defense=none means no protection was applied.\n"

    if asr >= 0.7:
        if no_defense:
            return (
                "URGENT ACTION REQUIRED\n"
                f"{note}"
                "- High ASR means the model is easy to break under this threat model.\n"
                "- Add a defense or switch to a robust model before deployment.\n"
                "- Validate with AutoAttack baseline and adaptive attacks.\n"
                "Priority: CRITICAL (days to remediate)."
            )
        return (
            "URGENT ACTION REQUIRED\n"
            f"{note}"
            f"- {defense_name} is highly vulnerable to adaptive attacks.\n"
            "- Review defense configuration and increase robustness.\n"
            "- Validate with AutoAttack baseline and adaptive attacks.\n"
            "Priority: CRITICAL (days to remediate)."
        )
    if asr >= 0.5:
        if no_defense:
            return (
                "IMPROVEMENTS NEEDED\n"
                f"{note}"
                "- Moderate ASR indicates baseline vulnerability under this threat model.\n"
                "- Add a defense or increase robustness.\n"
                "Priority: HIGH (1-2 weeks)."
            )
        return (
            "IMPROVEMENTS NEEDED\n"
            f"{note}"
            f"- {defense_name} shows moderate vulnerability.\n"
            "- Tune defense hyperparameters and add complementary defenses.\n"
            "Priority: HIGH (1-2 weeks)."
        )
    if asr >= 0.3:
        if no_defense:
            return (
                "MONITORING RECOMMENDED\n"
                f"{note}"
                "- This attack did not succeed often, but no defense was applied.\n"
                "- Increase attack strength or test with additional attacks.\n"
                "Priority: MEDIUM."
            )
        return (
            "MONITORING RECOMMENDED\n"
            f"{note}"
            f"- {defense_name} provides reasonable protection.\n"
            "- Continue periodic evaluation and document baseline.\n"
            "Priority: MEDIUM."
        )
    if no_defense:
        return (
            "DEFENSE EFFECTIVE\n"
            f"{note}"
            "- This attack did not succeed, but no defense was applied.\n"
            "- Verify with stronger attacks before concluding robustness.\n"
            "Priority: LOW (monitoring only)."
        )
    return (
        "DEFENSE EFFECTIVE\n"
        f"{note}"
        f"- {defense_name} appears robust against this attack.\n"
        "- Evaluate against additional attack types and transfer settings.\n"
        "Priority: LOW (monitoring only)."
    )


def render_attack_report(
    console: Console,
    *,
    summary: Dict[str, Any],
    meta: Dict[str, Any],
    output_path: str,
    verbosity: int = 0,
    report_format: str = "rich",
    brief: bool = False,
    summary_only: bool = False,
) -> None:
    asr = float(summary.get("attack_success_rate", 0.0))
    robust_acc = float(summary.get("robust_accuracy", 0.0))
    clean_acc = float(summary.get("clean_accuracy", 0.0))
    drop = clean_acc - robust_acc
    risk_label, risk_color = _risk_level(asr)
    threat = f"{meta.get('norm')} eps={_format_float(meta.get('epsilon'))}"
    summary_rows = [
        ("Attack", str(meta.get("attack", "attack"))),
        ("Defense", str(meta.get("defense", "none"))),
        ("Dataset", str(meta.get("dataset", "unknown"))),
        ("Threat", threat),
        ("ASR", _format_pct(asr)),
        ("Robust Acc", _format_pct(robust_acc)),
        ("Clean Acc", _format_pct(clean_acc)),
        ("Accuracy Drop", _format_pct(drop)),
    ]

    if report_format == "text":
        _print_text_summary(console, "Executive Summary", summary_rows, (risk_label, risk_color))
        if not brief and not summary_only:
            console.print(
                f"Recommendation: {_recommendation(asr, str(meta.get('defense'))).splitlines()[0]}"
            )
        console.print(f"Results saved to: {output_path}")
        return

    console.rule(f"Attack Evaluation: {str(meta.get('attack', 'attack')).upper()}")
    console.print(_summary_panel("Executive Summary", summary_rows, (risk_label, risk_color)))
    if summary_only:
        console.print(f"Results saved to: {output_path}")
        return

    table = Table(
        title="Attack Metrics Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan", width=24)
    table.add_column("Value", style="white", width=14)
    if not brief:
        table.add_column("Interpretation", style="dim", width=48)
    defense_name = str(meta.get("defense", ""))
    no_defense = defense_name.lower() in {"none", "no_defense", "nodefense", "baseline"}
    asr_note = _asr_interpretation(asr)
    if no_defense:
        asr_note = "No defense selected; this is baseline vulnerability."
    if brief:
        table.add_row("Attack Success Rate", _format_pct(asr))
        table.add_row("Robust Accuracy", _format_pct(robust_acc))
        table.add_row("Clean Accuracy", _format_pct(clean_acc))
    else:
        table.add_row("Attack Success Rate", _format_pct(asr), asr_note)
        table.add_row("Robust Accuracy", _format_pct(robust_acc), "Accuracy under attack")
        table.add_row("Clean Accuracy", _format_pct(clean_acc), "Baseline accuracy on clean data")
    drop_color = "red" if drop >= 0.3 else "yellow" if drop >= 0.1 else "green"
    if brief:
        table.add_row("Accuracy Drop", f"[{drop_color}]{_format_pct(drop)}[/{drop_color}]")
    else:
        table.add_row(
            "Accuracy Drop",
            f"[{drop_color}]{_format_pct(drop)}[/{drop_color}]",
            "Defense robustness indicator",
        )
    console.print(table)

    if not brief:
        console.print(Panel(_recommendation(asr, defense_name), title="Recommendation", border_style="green"))

    if verbosity >= 1 and not brief:
        config = Table(title="Attack Configuration", box=box.SIMPLE, show_header=False)
        config.add_column("Parameter", style="cyan")
        config.add_column("Value", style="yellow")
        config.add_row("Attack", str(meta.get("attack")))
        config.add_row("Defense", str(meta.get("defense")))
        config.add_row("Dataset", str(meta.get("dataset")))
        config.add_row("Threat Model", f"{meta.get('norm')} eps={_format_float(meta.get('epsilon'))}")
        config.add_row("Iterations", str(meta.get("iterations")))
        config.add_row("Targeted", "yes" if meta.get("targeted") else "no")
        console.print(config)

        perturb = summary.get("perturbation", {}) or {}
        pert_table = Table(title="Perturbation Statistics", box=box.SIMPLE, show_header=False)
        pert_table.add_column("Metric", style="cyan")
        pert_table.add_column("Value", style="yellow")
        pert_table.add_row("Norm", str(meta.get("norm")))
        pert_table.add_row("Mean", _format_float(perturb.get("primary_norm_mean") or perturb.get("linf_mean")))
        pert_table.add_row("Max", _format_float(perturb.get("primary_norm_max") or perturb.get("linf_max")))
        pert_table.add_row("Budget (eps)", _format_float(meta.get("epsilon")))
        console.print(pert_table)

        query = summary.get("query_efficiency", {}) or {}
        if query:
            query_table = Table(title="Query Cost", box=box.SIMPLE, show_header=False)
            query_table.add_column("Metric", style="cyan")
            query_table.add_column("Value", style="yellow")
            query_table.add_row("Mean Queries", _format_float(query.get("mean_queries")))
            query_table.add_row("Median Queries", _format_float(query.get("median_queries")))
            query_table.add_row("Mean Iterations", _format_float(query.get("mean_iterations")))
            console.print(query_table)

    console.print(f"Results saved to: {output_path}")


def render_characterization_report(
    console: Console,
    *,
    characterization: Dict[str, Any],
    defense: str,
    dataset: str,
    output_path: str,
    verbosity: int = 0,
    report_format: str = "rich",
    brief: bool = False,
    summary_only: bool = False,
) -> None:
    obf_types = characterization.get("obfuscation_types") or []
    obf_str = ", ".join(obf_types) if obf_types else "none"
    confidence = characterization.get("confidence")
    severity_label, severity_color = _obfuscation_severity(obf_types, confidence)

    bypass = []
    if characterization.get("requires_bpda"):
        bypass.append("BPDA")
    if characterization.get("requires_eot"):
        bypass.append("EOT")
    if characterization.get("requires_mapgd"):
        bypass.append("MA-PGD")
    bypass_str = ", ".join(bypass) if bypass else "none"

    summary_rows = [
        ("Defense", defense),
        ("Dataset", dataset),
        ("Obfuscation", obf_str),
        ("Confidence", _format_float(confidence)),
        ("Bypass", bypass_str),
    ]

    if report_format == "text":
        _print_text_summary(console, "Executive Summary", summary_rows, (severity_label, severity_color))
        console.print(f"Results saved to: {output_path}")
        return

    console.rule("Defense Characterization")
    console.print(_summary_panel("Executive Summary", summary_rows, (severity_label, severity_color)))
    if summary_only:
        console.print(f"Results saved to: {output_path}")
        return

    strategy = Table(title="Recommended Bypass Strategy", box=box.ROUNDED, show_header=True)
    strategy.add_column("Technique", style="cyan")
    strategy.add_column("Configuration", style="yellow")
    strategy.add_column("Rationale", style="dim")
    if bypass_str == "none":
        strategy.add_row("Standard PGD", "default parameters", "No strong obfuscation signals detected")
    if "BPDA" in bypass:
        strategy.add_row("BPDA", "differentiable approx", "Handles non-differentiable defenses")
    if "EOT" in bypass:
        strategy.add_row(
            "EOT",
            f"samples={characterization.get('recommended_eot_samples', 1)}",
            "Handles stochastic defenses",
        )
    if "MA-PGD" in bypass:
        strategy.add_row(
            "MA-PGD",
            f"memory={characterization.get('recommended_memory_length', 1)}",
            "Handles temporal dynamics",
        )
    if not brief:
        console.print(strategy)

    if verbosity >= 1 and not brief:
        signals = Table(title="Characterization Signals", box=box.SIMPLE, show_header=False)
        signals.add_column("Signal", style="cyan", width=26)
        signals.add_column("Value", style="yellow", width=16)
        signals.add_row("ETD", _format_float(characterization.get("etd_score")))
        signals.add_row("Alpha (Volterra)", _format_float(characterization.get("alpha_volterra")))
        signals.add_row("Gradient Variance", _format_float(characterization.get("gradient_variance")))
        signals.add_row("Jacobian Rank", _format_float(characterization.get("jacobian_rank")))
        signals.add_row("Autocorr Timescale", _format_float(characterization.get("autocorr_timescale")))
        signals.add_row("Confidence", _format_float(confidence))
        console.print(signals)

    console.print(f"Results saved to: {output_path}")


def render_evaluation_report(
    console: Console,
    *,
    results: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: str,
    report_format: str = "rich",
    brief: bool = False,
    summary_only: bool = False,
) -> None:
    threat = config.get("perturbation", {})
    epsilon = threat.get("epsilon", 8 / 255)
    norm = threat.get("norm", "Linf")
    attack_names: Iterable[str] = []
    if results:
        attack_names = (results[0].get("attacks", {}) or {}).keys()
    attack_names = list(attack_names)

    defense_worst: list[Tuple[str, float]] = []
    for res in results:
        defense = str(res.get("defense", res.get("type", "defense")))
        attacks = res.get("attacks", {}) or {}
        worst = 0.0
        for attack in attack_names:
            worst = max(worst, float(attacks.get(attack, {}).get("attack_success_rate", 0.0)))
        defense_worst.append((defense, worst))

    overall_worst = max((w for _d, w in defense_worst), default=0.0)
    best_defense = "n/a"
    worst_defense = "n/a"
    best_worst = 0.0
    worst_worst = 0.0
    if defense_worst:
        best_defense, best_worst = min(defense_worst, key=lambda item: item[1])
        worst_defense, worst_worst = max(defense_worst, key=lambda item: item[1])

    severity = _risk_level(overall_worst)
    summary_rows = [
        ("Threat", f"{norm} eps={_format_float(epsilon)}"),
        ("Defenses", str(len(results))),
        ("Attacks", str(len(attack_names))),
        ("Worst ASR", _format_pct(overall_worst)),
        ("Best Defense", f"{best_defense} ({_format_pct(best_worst)})" if defense_worst else "n/a"),
        ("Worst Defense", f"{worst_defense} ({_format_pct(worst_worst)})" if defense_worst else "n/a"),
    ]

    if report_format == "text":
        _print_text_summary(console, "Executive Summary", summary_rows, severity)
        console.print(f"Results saved to: {output_dir}")
        return

    console.rule("Evaluation Suite Results")
    console.print(_summary_panel("Executive Summary", summary_rows, severity))
    if summary_only:
        console.print(f"Results saved to: {output_dir}/summary.json")
        return

    table = Table(
        title="Attack Success Rate (ASR) by Defense",
        box=box.HEAVY_HEAD,
        header_style="bold cyan",
    )
    table.add_column("Defense", style="white", width=18)
    if not brief:
        for attack in attack_names:
            table.add_column(str(attack), justify="center", width=10)
    table.add_column("Worst", justify="center", width=10)
    table.add_column("Severity", justify="center", width=10)

    for res in results:
        defense = str(res.get("defense", res.get("type", "defense")))
        attacks = res.get("attacks", {}) or {}
        row = [defense]
        worst = 0.0
        for attack in attack_names:
            asr = float(attacks.get(attack, {}).get("attack_success_rate", 0.0))
            worst = max(worst, asr)
            if not brief:
                color = "red" if asr >= 0.7 else "yellow" if asr >= 0.5 else "green"
                row.append(f"[{color}]{_format_pct(asr)}[/{color}]")
        worst_color = "red" if worst >= 0.7 else "yellow" if worst >= 0.5 else "green"
        row.append(f"[bold {worst_color}]{_format_pct(worst)}[/bold {worst_color}]")
        severity_label, severity_color = _risk_level(worst)
        row.append(f"[bold {severity_color}]{severity_label}[/bold {severity_color}]")
        table.add_row(*row)

    console.print(table)
    console.print(f"Results saved to: {output_dir}/summary.json")
