"""
MITRE ATLAS CLI (offline STIX).

This Click command mirrors the legacy argparse `mitre-atlas` tooling but is
integrated into the main `neurinspectre` CLI for artifact-grade reproducibility.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import click

from ..mitre_atlas.registry import (
    list_atlas_tactics,
    list_atlas_techniques,
    load_stix_atlas_bundle,
    tactic_by_phase_name,
    technique_index,
)
from ..mitre_atlas.coverage import (
    format_module_coverage_markdown,
    module_coverage,
    neurinspectre_package_root,
    scan_markdown_label_mismatches,
    scan_tree_for_atlas_placeholders,
    validate_atlas_ids,
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_or_echo(text: str, out: Optional[str]) -> None:
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        click.echo(str(p))
    else:
        click.echo(text)


@click.group("mitre-atlas")
def mitre_atlas_cmd() -> None:
    """MITRE ATLAS catalog + coverage tools (offline STIX)."""


@mitre_atlas_cmd.command("list")
@click.argument("kind", type=click.Choice(["tactics", "techniques"]))
@click.option("--format", "fmt", type=click.Choice(["table", "json", "markdown"]), default="table")
@click.option("--tactic", default=None, help="Filter techniques by tactic id (AML.TA####) or tactic name")
@click.option("--no-subtechniques", is_flag=True, help="Exclude sub-techniques (AML.Txxxx.yyy)")
def list_cmd(kind: str, fmt: str, tactic: Optional[str], no_subtechniques: bool) -> None:
    bundle = load_stix_atlas_bundle()

    if kind == "tactics":
        tactics = list_atlas_tactics(bundle)
        if fmt == "json":
            _write_or_echo(json.dumps([asdict(t) for t in tactics], indent=2), out=None)
            return
        if fmt == "markdown":
            _write_or_echo("\n".join(f"- {t.tactic_id} {t.name}" for t in tactics), out=None)
            return
        # table
        click.echo("tactic_id\tname\tphase_name")
        for t in tactics:
            click.echo(f"{t.tactic_id}\t{t.name}\t{t.phase_name}")
        return

    # techniques
    techs = list_atlas_techniques(bundle)

    if tactic:
        q = str(tactic).strip().lower()
        phase_to_tactic = tactic_by_phase_name(bundle)
        tactics = list_atlas_tactics(bundle)
        want_tactic_id = None
        if q.startswith("aml.ta"):
            want_tactic_id = q
        else:
            for t in tactics:
                if t.name.lower() == q:
                    want_tactic_id = t.tactic_id.lower()
                    break

        if want_tactic_id:
            def _has_tactic(tech) -> bool:
                for ph in tech.tactic_phase_names:
                    tac = phase_to_tactic.get(ph)
                    if tac and tac.tactic_id.lower() == want_tactic_id:
                        return True
                return False

            techs = [t for t in techs if _has_tactic(t)]

    if no_subtechniques:
        techs = [t for t in techs if (("." not in t.technique_id) and (not t.is_subtechnique))]

    if fmt == "json":
        _write_or_echo(json.dumps([asdict(t) for t in techs], indent=2), out=None)
        return
    if fmt == "markdown":
        _write_or_echo("\n".join(f"- {t.technique_id} {t.name}" for t in techs), out=None)
        return

    phase_to_tactic = tactic_by_phase_name(bundle)
    click.echo("technique_id\tname\ttactics")
    for t in techs:
        tnames = []
        for ph in t.tactic_phase_names:
            tac = phase_to_tactic.get(ph)
            if tac and tac.name not in tnames:
                tnames.append(tac.name)
        click.echo(f"{t.technique_id}\t{t.name}\t{', '.join(tnames)}")


@mitre_atlas_cmd.command("show")
@click.argument("atlas_id")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def show_cmd(atlas_id: str, fmt: str) -> None:
    bundle = load_stix_atlas_bundle()
    atlas_id = str(atlas_id).strip()

    tactics = list_atlas_tactics(bundle)
    tidx = technique_index(bundle)
    phase_to_tactic = tactic_by_phase_name(bundle)

    obj: Dict[str, Any]
    if atlas_id.startswith("AML.TA"):
        t = next((x for x in tactics if x.tactic_id == atlas_id), None)
        if t is None:
            raise click.ClickException(f"Unknown tactic id: {atlas_id}")
        obj = {
            "id": t.tactic_id,
            "type": "tactic",
            "name": t.name,
            "phase_name": t.phase_name,
            "description": t.description,
        }
    else:
        tech = tidx.get(atlas_id)
        if tech is None:
            raise click.ClickException(f"Unknown technique id: {atlas_id}")
        tnames = []
        tids = []
        for ph in tech.tactic_phase_names:
            tac = phase_to_tactic.get(ph)
            if tac and tac.tactic_id not in tids:
                tids.append(tac.tactic_id)
                tnames.append(tac.name)
        obj = {
            "id": tech.technique_id,
            "type": "technique",
            "name": tech.name,
            "tactic_ids": tids,
            "tactics": tnames,
            "tactic_phase_names": tech.tactic_phase_names,
            "url": tech.url,
            "is_subtechnique": tech.is_subtechnique,
            "description": tech.description,
        }

    if fmt == "json":
        click.echo(json.dumps(obj, indent=2))
        return

    # text
    click.echo(f"{obj['id']} ({obj['type']})")
    click.echo(f"Name: {obj.get('name')}")
    if obj.get("type") == "tactic":
        click.echo(f"Phase: {obj.get('phase_name')}")
    else:
        click.echo(f"Tactics: {', '.join(obj.get('tactics') or [])}")
        if obj.get("url"):
            click.echo(f"URL: {obj.get('url')}")
    desc = (obj.get("description") or "").strip()
    if desc:
        click.echo("")
        click.echo(desc)


@mitre_atlas_cmd.command("coverage")
@click.option("--scope", type=click.Choice(["code", "docs", "all"]), default="code")
@click.option("--format", "fmt", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--out", default=None, help="Write output to file instead of stdout")
def coverage_cmd(scope: str, fmt: str, out: Optional[str]) -> None:
    rows = module_coverage(scope=str(scope))
    if fmt == "json":
        _write_or_echo(json.dumps([asdict(r) for r in rows], indent=2), out=out)
        return
    _write_or_echo(format_module_coverage_markdown(rows), out=out)


@mitre_atlas_cmd.command("validate")
@click.option("--scope", type=click.Choice(["code", "docs", "all"]), default="all")
@click.option("--format", "fmt", type=click.Choice(["json", "text"]), default="json")
@click.option("--out", default=None, help="Write output JSON/text to file instead of stdout")
@click.option("--strict", is_flag=True, help="Exit non-zero if unknown AML.* IDs are found")
def validate_cmd(scope: str, fmt: str, out: Optional[str], strict: bool) -> None:
    rows = module_coverage(scope=str(scope))
    referenced = [mid for r in rows for mid in (r.atlas_ids or [])]
    res = validate_atlas_ids(referenced)

    pkg_root = neurinspectre_package_root()
    repo_root = pkg_root.parent
    label_mismatches = scan_markdown_label_mismatches(repo_root) if scope in {"docs", "all"} else []
    placeholders = scan_tree_for_atlas_placeholders(repo_root) if scope in {"docs", "all"} else []

    payload: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "scope": str(scope),
        "validation": asdict(res),
        "label_mismatches": [asdict(m) for m in label_mismatches],
        "placeholders": [asdict(h) for h in placeholders],
    }

    if fmt == "text":
        txt = [
            f"MITRE ATLAS validation (scope={scope})",
            f"Catalog: tactics={res.tactic_count} techniques={res.technique_count}",
            f"Referenced IDs: total={res.referenced_ids_total} unique={res.referenced_ids_unique}",
            f"Unknown IDs: {len(res.unknown_ids)}",
        ]
        if res.unknown_ids:
            txt.append("Unknown:")
            txt.extend(f"- {x}" for x in res.unknown_ids)
        if label_mismatches:
            txt.append(f"Label mismatches in markdown: {len(label_mismatches)}")
        if placeholders:
            txt.append(f"ATLAS placeholders found: {len(placeholders)}")
        _write_or_echo("\n".join(txt), out=out)
    else:
        _write_or_echo(json.dumps(payload, indent=2), out=out)

    if strict and res.unknown_ids:
        raise SystemExit(2)

