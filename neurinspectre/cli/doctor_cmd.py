"""
`neurinspectre doctor` - lightweight environment sanity checks.

This command is intentionally conservative: it does not attempt network access or
download assets. It helps artifact evaluators confirm that optional dependencies
are installed and that local model metadata is present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click

from .metadata import collect_env_metadata, collect_git_metadata


def _scan_stub_models(models_dir: Path) -> Tuple[int, List[str]]:
    """
    Scan `models_dir` for `.meta.json` files that mark `is_stub=true`.
    """

    stubbed: List[str] = []
    try:
        for meta_path in sorted(models_dir.glob("*.meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if bool(meta.get("is_stub", False)) or bool(meta.get("stub", False)) or bool(meta.get("placeholder", False)):
                stubbed.append(meta_path.name)
    except Exception:
        return 0, []
    return len(stubbed), stubbed


def run_doctor(ctx: click.Context, **kwargs: Any) -> None:
    json_output = kwargs.get("json_output")
    as_json = bool(kwargs.get("as_json", False))
    models_dir = Path(str(kwargs.get("models_dir", "models")))
    check_models = bool(kwargs.get("check_models", True))

    payload: Dict[str, Any] = {
        "env": collect_env_metadata(),
        "git": collect_git_metadata(),
    }

    if check_models and models_dir.exists() and models_dir.is_dir():
        stub_count, stub_names = _scan_stub_models(models_dir)
        payload["models"] = {
            "models_dir": str(models_dir),
            "stub_meta_count": int(stub_count),
            "stub_meta_files": list(stub_names),
        }
    else:
        payload["models"] = {
            "models_dir": str(models_dir),
            "stub_meta_count": None,
            "stub_meta_files": [],
            "note": "models directory not found or model scanning disabled",
        }

    if json_output:
        out_p = Path(str(json_output))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        click.echo(f"Wrote {out_p}")

    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    # Human-readable minimal output
    py = (payload.get("env") or {}).get("python") or {}
    plat = (payload.get("env") or {}).get("platform") or {}
    torch = (payload.get("env") or {}).get("torch") or {}
    opt = (payload.get("env") or {}).get("optional_modules") or {}
    git = payload.get("git") or {}
    models = payload.get("models") or {}

    click.echo(f"Python: {py.get('version')} ({py.get('executable')})")
    click.echo(f"Platform: {plat.get('system')} {plat.get('release')} {plat.get('machine')}")
    if isinstance(torch, dict) and torch.get("available") is False:
        click.echo("Torch: not available")
    else:
        click.echo(
            "Torch: "
            f"{torch.get('version')} cuda={torch.get('cuda_available')} mps={torch.get('mps_available')}"
        )
    click.echo(f"Optional deps: ember={opt.get('ember')} nuscenes={opt.get('nuscenes')} onnxruntime={opt.get('onnxruntime')}")

    if isinstance(git, dict) and git.get("available"):
        click.echo(f"Git: commit={git.get('commit')} dirty={git.get('dirty')} branch={git.get('branch')}")
    else:
        click.echo("Git: not a repo (or git unavailable)")

    click.echo(f"Models: dir={models.get('models_dir')} stub_meta_count={models.get('stub_meta_count')}")

