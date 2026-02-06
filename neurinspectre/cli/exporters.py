"""
Export NeurInSpectre findings to structured formats.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def export_attack_json(report: Dict[str, Any], path: str | Path) -> None:
    payload = {
        "metadata": _metadata(),
        "type": "attack",
        "report": report,
    }
    _write_json(payload, path)


def export_characterization_json(report: Dict[str, Any], path: str | Path) -> None:
    payload = {
        "metadata": _metadata(),
        "type": "characterization",
        "report": report,
    }
    _write_json(payload, path)


def export_evaluation_json(report: Dict[str, Any], path: str | Path) -> None:
    payload = {
        "metadata": _metadata(),
        "type": "evaluation",
        "report": report,
    }
    _write_json(payload, path)


def export_compare_json(report: Dict[str, Any], path: str | Path) -> None:
    payload = {
        "metadata": _metadata(),
        "type": "compare",
        "report": report,
    }
    _write_json(payload, path)


def export_attack_sarif(report: Dict[str, Any], path: str | Path) -> None:
    attack = report.get("attack", "attack")
    defense = report.get("defense", "defense")
    asr = float(report.get("attack_success_rate", 0.0))
    characterization = report.get("characterization", {}) or {}
    obf_types = characterization.get("obfuscation_types") or []

    results = []
    if asr >= 0.7:
        results.append(
            _sarif_result(
                "high-attack-success",
                "error",
                f"Attack success rate {asr:.2f} indicates weak defense ({defense}) against {attack}.",
            )
        )
    elif asr >= 0.5:
        results.append(
            _sarif_result(
                "moderate-attack-success",
                "warning",
                f"Attack success rate {asr:.2f} indicates moderate risk ({defense}) against {attack}.",
            )
        )
    if obf_types:
        results.append(
            _sarif_result(
                "gradient-obfuscation-detected",
                "warning",
                f"Gradient obfuscation detected: {', '.join(obf_types)}.",
            )
        )

    sarif = _sarif_document(results)
    _write_json(sarif, path)


def export_characterization_sarif(report: Dict[str, Any], path: str | Path) -> None:
    obf_types = report.get("obfuscation_types") or []
    confidence = float(report.get("confidence", 0.0))
    results = []
    if obf_types:
        results.append(
            _sarif_result(
                "gradient-obfuscation-detected",
                "warning",
                f"Detected obfuscation: {', '.join(obf_types)} (confidence={confidence:.2f}).",
            )
        )
    sarif = _sarif_document(results)
    _write_json(sarif, path)


def export_evaluation_sarif(report: Dict[str, Any], path: str | Path) -> None:
    results = []
    pairs = report.get("pairs", []) or []
    for defense, attack, asr, drop in pairs:
        if asr >= 0.7 or drop >= 0.3:
            results.append(
                _sarif_result(
                    "evaluation-risk-pair",
                    "warning" if asr < 0.7 else "error",
                    f"{defense} vs {attack}: ASR={asr:.2f}, drop={drop:.2f}",
                )
            )
    sarif = _sarif_document(results)
    _write_json(sarif, path)


def export_compare_sarif(report: Dict[str, Any], path: str | Path) -> None:
    results = []
    findings = report.get("findings", []) or []
    for item in findings:
        level = str(item.get("level", "warning"))
        message = str(item.get("message", "comparison finding"))
        results.append(_sarif_result("comparison-delta", level, message))
    sarif = _sarif_document(results)
    _write_json(sarif, path)


def _metadata() -> Dict[str, Any]:
    return {
        "tool": "NeurInSpectre",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _sarif_document(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "NeurInSpectre", "version": "2.0.0"}},
                "results": results,
            }
        ],
    }


def _sarif_result(rule_id: str, level: str, message: str) -> Dict[str, Any]:
    return {
        "ruleId": rule_id,
        "level": level,
        "message": {"text": message},
    }


def _write_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
