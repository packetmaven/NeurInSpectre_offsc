"""
Comparison command implementation for NeurInSpectre CLI (Enhancement #5).

Provides side-by-side comparison of NeurInSpectre results against
AutoAttack baselines and across evaluation runs, enabling:
  - Attack-vs-attack comparison on same defense (Paper Table 1 columns)
  - Defense-vs-defense comparison under same attack
  - Run-vs-run diff for regression testing
  - Baseline comparison against AutoAttack/RobustBench (Paper Section 5.2.2)
  - Characterization signal comparison across defenses

PAPER ALIGNMENT:
- Paper Table 1: NeurInSpectre vs PGD vs AutoAttack comparison
- Paper Section 5.2.2: "outperforms AutoAttack by 29.5 pp on average"
- Paper Table 2: Ablation component comparison
- Paper Table 3: Detection capability comparison across methods

ENHANCEMENT #5:
  `neurinspectre compare` provides structured, actionable side-by-side
  analysis for AI red teamers and security researchers performing
  authorized adversarial evaluation of ML systems.

Version: 2.0.1 (WOOT 2026 submission aligned)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .exporters import export_compare_json, export_compare_sarif
from .formatters import build_console
from .utils import save_json

logger = logging.getLogger(__name__)

COMPARE_MODES = ("attacks", "defenses", "runs", "baseline", "characterization")

PAPER_TABLE1_DEFENSE_ASR = {
    "jpegcompression": {"pgd": 0.124, "autoattack": 0.673, "neurinspectre": 0.982},
    "bitdepthreduction": {"pgd": 0.087, "autoattack": 0.718, "neurinspectre": 0.974},
    "randsmoothing": {"pgd": 0.312, "autoattack": 0.584, "neurinspectre": 0.893},
    "ensemblediversity": {"pgd": 0.246, "autoattack": 0.621, "neurinspectre": 0.917},
    "featuresqueezing": {"pgd": 0.153, "autoattack": 0.694, "neurinspectre": 0.968},
    "gradientreg": {"pgd": 0.412, "autoattack": 0.732, "neurinspectre": 0.884},
    "defdistillation": {"pgd": 0.062, "autoattack": 0.521, "neurinspectre": 0.946},
    "attransform": {"pgd": 0.387, "autoattack": 0.719, "neurinspectre": 0.921},
    "spatialsmoothing": {"pgd": 0.221, "autoattack": 0.658, "neurinspectre": 0.953},
    "randompadcrop": {"pgd": 0.184, "autoattack": 0.592, "neurinspectre": 0.937},
    "thermometerenc": {"pgd": 0.098, "autoattack": 0.486, "neurinspectre": 0.961},
    "certifieddefense": {"pgd": 0.523, "autoattack": 0.784, "neurinspectre": 0.978},
}


def run_compare(ctx: click.Context, **kwargs: Any) -> None:
    cmd_verbose = int(kwargs.get("verbose", 0) or 0)
    if ctx is not None:
        ctx.obj = ctx.obj or {}
        if cmd_verbose:
            ctx.obj["verbose"] = max(int(ctx.obj.get("verbose", 0)), cmd_verbose)

    mode = str(kwargs.get("mode", "attacks")).lower()
    if mode not in COMPARE_MODES:
        raise click.ClickException(f"Unknown compare mode: {mode}")

    input_files = list(kwargs.get("input_files") or [])
    if not input_files:
        raise click.ClickException("Provide at least one input JSON file.")

    sort_by = str(kwargs.get("sort_by", "asr"))
    threshold_pp = float(kwargs.get("threshold", 2.0))
    report_format = str(kwargs.get("report_format", "rich"))
    no_color = bool(kwargs.get("no_color", False))
    force_color = bool(kwargs.get("color", False))
    console = build_console(no_color=no_color, force_color=force_color)

    datasets = _load_results(input_files)

    if mode == "attacks":
        payload = _compare_attacks(datasets)
    elif mode == "defenses":
        payload = _compare_defenses(datasets, sort_by=sort_by)
    elif mode == "runs":
        payload = _compare_runs(datasets, threshold_pp=threshold_pp)
    elif mode == "baseline":
        payload = _compare_baseline(datasets)
    else:
        payload = _compare_characterization(datasets, sort_by=sort_by)

    payload["mode"] = mode
    payload["inputs"] = input_files

    output_path = kwargs.get("output")
    if output_path:
        save_json(payload, Path(output_path))
        click.echo(f"Comparison written to {output_path}")

    if report_format:
        _render_comparison_report(console, payload, report_format=report_format)

    json_output = kwargs.get("json_output")
    if json_output:
        export_compare_json(payload, json_output)
        click.echo(f"JSON report written to {json_output}")

    sarif_output = kwargs.get("sarif_output")
    if sarif_output:
        export_compare_sarif(payload, sarif_output)
        click.echo(f"SARIF report written to {sarif_output}")


def _compare_attacks(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = _extract_pairs(datasets)
    defenses = sorted({d for d, _a, _m in pairs})
    attacks = sorted({a for _d, a, _m in pairs})
    matrix: Dict[str, Dict[str, float]] = {d: {} for d in defenses}
    for defense, attack, metrics in pairs:
        matrix[defense][attack] = float(metrics.get("attack_success_rate", 0.0))
    return {"defenses": defenses, "attacks": attacks, "matrix": matrix}


def _compare_defenses(datasets: List[Dict[str, Any]], *, sort_by: str) -> Dict[str, Any]:
    pairs = _extract_pairs(datasets)
    attacks = sorted({a for _d, a, _m in pairs})
    ranking: Dict[str, List[Tuple[str, float]]] = {}
    for attack in attacks:
        rows = [
            (defense, float(metrics.get("attack_success_rate", 0.0)))
            for defense, attack_name, metrics in pairs
            if attack_name == attack
        ]
        rows.sort(key=lambda item: item[1], reverse=True)
        ranking[attack] = rows
    return {"attacks": attacks, "ranking": ranking, "sort_by": sort_by}


def _compare_runs(datasets: List[Dict[str, Any]], *, threshold_pp: float) -> Dict[str, Any]:
    if len(datasets) < 2:
        raise click.ClickException("Run comparison requires at least two summary files.")

    base = datasets[0]
    base_pairs = _pair_map(_extract_pairs([base]))
    comparisons = []
    findings = []
    for other in datasets[1:]:
        other_pairs = _pair_map(_extract_pairs([other]))
        for key in sorted(set(base_pairs) | set(other_pairs)):
            base_asr = base_pairs.get(key, 0.0)
            other_asr = other_pairs.get(key, 0.0)
            delta = other_asr - base_asr
            significant = abs(delta) >= threshold_pp / 100.0
            comparisons.append(
                {
                    "defense": key[0],
                    "attack": key[1],
                    "base_asr": base_asr,
                    "other_asr": other_asr,
                    "delta": delta,
                    "significant": significant,
                }
            )
            if significant:
                findings.append(
                    {
                        "level": "warning" if abs(delta) < 0.3 else "error",
                        "message": f"{key[0]} vs {key[1]} delta={delta:.3f} (threshold={threshold_pp:.1f}pp)",
                    }
                )
    return {"comparisons": comparisons, "threshold_pp": threshold_pp, "findings": findings}


def _compare_baseline(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = _extract_pairs(datasets)
    comparisons = []
    findings = []
    for defense, attack, metrics in pairs:
        key = _normalize_defense_key(defense)
        baseline = PAPER_TABLE1_DEFENSE_ASR.get(key, {})
        if not baseline:
            continue
        base_asr = float(baseline.get(attack, 0.0))
        obs_asr = float(metrics.get("attack_success_rate", 0.0))
        delta = obs_asr - base_asr
        comparisons.append(
            {
                "defense": defense,
                "attack": attack,
                "baseline_asr": base_asr,
                "observed_asr": obs_asr,
                "delta": delta,
            }
        )
        if abs(delta) >= 0.02:
            findings.append(
                {
                    "level": "warning" if abs(delta) < 0.3 else "error",
                    "message": f"{defense} vs {attack} baseline delta={delta:.3f}",
                }
            )
    return {"comparisons": comparisons, "findings": findings}


def _compare_characterization(datasets: List[Dict[str, Any]], *, sort_by: str) -> Dict[str, Any]:
    records = []
    for data in datasets:
        for entry in _extract_characterizations(data):
            records.append(entry)
    key_map = {
        "confidence": "confidence",
        "etd": "etd_score",
        "alpha": "alpha_volterra",
        "variance": "gradient_variance",
        "asr": "asr",
    }
    sort_key = key_map.get(sort_by, "confidence")
    records.sort(key=lambda item: float(item.get(sort_key, 0.0)), reverse=True)
    return {"records": records, "sort_by": sort_by}


def _render_comparison_report(console: Console, payload: Dict[str, Any], *, report_format: str) -> None:
    if report_format == "text":
        _render_text_report(console, payload)
        return

    mode = payload.get("mode")
    if mode == "attacks":
        _render_attacks_comparison(console, payload)
    elif mode == "defenses":
        _render_defenses_comparison(console, payload)
    elif mode == "runs":
        _render_runs_comparison(console, payload)
    elif mode == "baseline":
        _render_baseline_comparison(console, payload)
    elif mode == "characterization":
        _render_characterization_comparison(console, payload)


def _render_attacks_comparison(console: Console, payload: Dict[str, Any]) -> None:
    defenses = payload.get("defenses", [])
    attacks = payload.get("attacks", [])
    matrix = payload.get("matrix", {})
    console.rule("Attack Comparison (Table 1 format)")
    table = Table(box=box.HEAVY_HEAD, header_style="bold cyan")
    table.add_column("Defense", style="white", width=18)
    for attack in attacks:
        table.add_column(str(attack), justify="center", width=10)
    table.add_column("Worst", justify="center", width=10)
    for defense in defenses:
        row = [defense]
        worst = 0.0
        for attack in attacks:
            asr = float(matrix.get(defense, {}).get(attack, 0.0))
            worst = max(worst, asr)
            color = "red" if asr >= 0.7 else "yellow" if asr >= 0.5 else "green"
            row.append(f"[{color}]{_fmt(asr)}[/{color}]")
        worst_color = "red" if worst >= 0.7 else "yellow" if worst >= 0.5 else "green"
        row.append(f"[bold {worst_color}]{_fmt(worst)}[/bold {worst_color}]")
        table.add_row(*row)
    console.print(table)


def _render_defenses_comparison(console: Console, payload: Dict[str, Any]) -> None:
    rankings = payload.get("ranking", {}) or {}
    console.rule("Defense Comparison (weakest first)")
    for attack, rows in rankings.items():
        table = Table(title=f"Attack: {attack}", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Defense", style="white", width=20)
        table.add_column("ASR", justify="center", width=10)
        for defense, asr in rows:
            color = "red" if asr >= 0.7 else "yellow" if asr >= 0.5 else "green"
            table.add_row(defense, f"[{color}]{_fmt(asr)}[/{color}]")
        console.print(table)


def _render_runs_comparison(console: Console, payload: Dict[str, Any]) -> None:
    comparisons = payload.get("comparisons", []) or []
    threshold_pp = float(payload.get("threshold_pp", 2.0))
    console.rule("Run Comparison (regression)")
    table = Table(box=box.HEAVY_HEAD, header_style="bold cyan")
    table.add_column("Defense", style="white", width=18)
    table.add_column("Attack", style="white", width=14)
    table.add_column("Base ASR", justify="center", width=10)
    table.add_column("Other ASR", justify="center", width=10)
    table.add_column("Delta", justify="center", width=10)
    table.add_column("Sig", justify="center", width=6)
    for row in comparisons:
        delta = float(row.get("delta", 0.0))
        color = "red" if abs(delta) >= threshold_pp / 100.0 else "green"
        table.add_row(
            str(row.get("defense")),
            str(row.get("attack")),
            _fmt(row.get("base_asr")),
            _fmt(row.get("other_asr")),
            f"[{color}]{delta:+.3f}[/{color}]",
            "yes" if row.get("significant") else "no",
        )
    console.print(table)


def _render_baseline_comparison(console: Console, payload: Dict[str, Any]) -> None:
    comparisons = payload.get("comparisons", []) or []
    console.rule("Paper Table 1 Baseline Comparison")
    table = Table(box=box.HEAVY_HEAD, header_style="bold cyan")
    table.add_column("Defense", style="white", width=18)
    table.add_column("Attack", style="white", width=14)
    table.add_column("Baseline", justify="center", width=10)
    table.add_column("Observed", justify="center", width=10)
    table.add_column("Delta", justify="center", width=10)
    for row in comparisons:
        delta = float(row.get("delta", 0.0))
        color = "red" if abs(delta) >= 0.02 else "green"
        table.add_row(
            str(row.get("defense")),
            str(row.get("attack")),
            _fmt(row.get("baseline_asr")),
            _fmt(row.get("observed_asr")),
            f"[{color}]{delta:+.3f}[/{color}]",
        )
    console.print(table)


def _render_characterization_comparison(console: Console, payload: Dict[str, Any]) -> None:
    records = payload.get("records", []) or []
    console.rule("Characterization Signal Comparison")
    table = Table(box=box.HEAVY_HEAD, header_style="bold cyan")
    table.add_column("Defense", style="white", width=18)
    table.add_column("Obfuscation", style="white", width=22)
    table.add_column("Confidence", justify="center", width=10)
    table.add_column("ETD", justify="center", width=8)
    table.add_column("Alpha", justify="center", width=8)
    table.add_column("Variance", justify="center", width=10)
    table.add_column("Jacobian", justify="center", width=9)
    table.add_column("Timescale", justify="center", width=10)
    for row in records:
        table.add_row(
            str(row.get("defense")),
            str(row.get("obfuscation", "none")),
            _fmt(row.get("confidence")),
            _fmt(row.get("etd_score")),
            _fmt(row.get("alpha_volterra")),
            _fmt(row.get("gradient_variance")),
            _fmt(row.get("jacobian_rank")),
            _fmt(row.get("autocorr_timescale")),
        )
    console.print(table)


def _render_text_report(console: Console, payload: Dict[str, Any]) -> None:
    mode = payload.get("mode", "compare")
    console.print(f"Compare mode: {mode}")
    if mode == "runs":
        console.print(f"Pairs compared: {len(payload.get('comparisons', []))}")
    if mode == "baseline":
        console.print(f"Baseline pairs: {len(payload.get('comparisons', []))}")
    if mode == "characterization":
        console.print(f"Records: {len(payload.get('records', []))}")


def _load_results(paths: List[str]) -> List[Dict[str, Any]]:
    results = []
    for path in paths:
        data = _load_json(Path(path))
        results.append(_unwrap_report(data))
    return results


def _unwrap_report(data: Dict[str, Any]) -> Dict[str, Any]:
    if "report" in data and isinstance(data["report"], dict):
        return data["report"]
    return data


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_pairs(datasets: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    pairs: List[Tuple[str, str, Dict[str, Any]]] = []
    for data in datasets:
        if "results" in data and isinstance(data["results"], list):
            for res in data.get("results", []) or []:
                pairs.extend(_extract_pairs([res]))
            continue
        if "attacks" in data and isinstance(data.get("attacks"), dict):
            defense = str(data.get("defense", data.get("type", "defense")))
            for attack_name, metrics in data.get("attacks", {}).items():
                pairs.append((defense, str(attack_name), dict(metrics)))
            continue
        if "attack_success_rate" in data and "attack" in data:
            defense = str(data.get("defense", data.get("dataset", "defense")))
            attack = str(data.get("attack"))
            pairs.append((defense, attack, data))
    return pairs


def _extract_characterizations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    if "results" in data and isinstance(data["results"], list):
        for res in data.get("results", []) or []:
            records.extend(_extract_characterizations(res))
        return records
    if "characterization" in data and isinstance(data.get("characterization"), dict):
        char = data.get("characterization") or {}
        records.append(
            {
                "defense": str(data.get("defense", data.get("type", "defense"))),
                "obfuscation": ", ".join(char.get("obfuscation_types", []) or []),
                "confidence": char.get("confidence"),
                "etd_score": char.get("etd_score"),
                "alpha_volterra": char.get("alpha_volterra"),
                "gradient_variance": char.get("gradient_variance"),
                "jacobian_rank": char.get("jacobian_rank"),
                "autocorr_timescale": char.get("autocorr_timescale"),
                "asr": data.get("attack_success_rate"),
            }
        )
    return records


def _pair_map(pairs: List[Tuple[str, str, Dict[str, Any]]]) -> Dict[Tuple[str, str], float]:
    return {(d, a): float(m.get("attack_success_rate", 0.0)) for d, a, m in pairs}


def _normalize_defense_key(defense: str) -> str:
    key = str(defense).lower().replace("_", "").replace("-", "").replace(" ", "")
    alias = {
        "jpeg": "jpegcompression",
        "jpegcompression": "jpegcompression",
        "bitdepth": "bitdepthreduction",
        "bitdepthreduction": "bitdepthreduction",
        "bitdepthreductiondefense": "bitdepthreduction",
        "randomizedsmoothing": "randsmoothing",
        "randsmooth": "randsmoothing",
        "randsmoothing": "randsmoothing",
        "ensemblediversity": "ensemblediversity",
        "featuresqueezing": "featuresqueezing",
        "feature_squeezing": "featuresqueezing",
        "gradientregularization": "gradientreg",
        "gradientreg": "gradientreg",
        "defensivedistillation": "defdistillation",
        "distillation": "defdistillation",
        "attransform": "attransform",
        "spatialsmoothing": "spatialsmoothing",
        "randompadcrop": "randompadcrop",
        "thermometerencoding": "thermometerenc",
        "thermometerenc": "thermometerenc",
        "certifieddefense": "certifieddefense",
    }
    return alias.get(key, key)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)
