#!/usr/bin/env python3
"""
Monitor and chain full Table 8 attack stages.

Behavior:
1) Ensure AutoAttack reaches 12/12 defenses (restart with --resume if interrupted).
2) Then ensure NeurInSpectre reaches 12/12 defenses (restart with --resume if interrupted).
3) Keep logging progress snapshots to monitor_table8_v2.log.

This script is intentionally conservative and idempotent: it never deletes outputs
and only launches resume-capable commands when a stage is incomplete and not running.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_defense_jsons(out_dir: Path) -> List[Tuple[Path, dict]]:
    rows: List[Tuple[Path, dict]] = []
    for p in sorted(out_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("attacks"), dict):
            rows.append((p, obj))
    return rows


def _count_stage_done(out_dir: Path, attack_name: str) -> Tuple[int, int, List[str]]:
    rows = _read_defense_jsons(out_dir)
    done = 0
    missing: List[str] = []
    for path, obj in rows:
        attacks = obj.get("attacks") or {}
        if isinstance(attacks, dict) and attack_name in attacks:
            done += 1
        else:
            missing.append(path.name)
    return done, len(rows), missing


def _list_attack_worker_pids(config_name: str, out_rel: str, attack_name: str) -> List[int]:
    """
    Find active `neurinspectre table2` workers for a specific attack stage.
    """

    try:
        out = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True)
    except Exception:
        return []

    pids: List[int] = []
    needle1 = "neurinspectre table2"
    needle2 = f"--attacks {attack_name}"
    needle3 = f"-c {config_name}"
    needle4 = f"-o {out_rel}"

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_s, cmd = line.split(None, 1)
        except ValueError:
            continue
        if needle1 not in cmd:
            continue
        if needle2 not in cmd or needle3 not in cmd or needle4 not in cmd:
            continue
        # Exclude obvious wrappers/searches, if any.
        if "python3 - <<'PY'" in cmd or "rg " in cmd:
            continue
        try:
            pids.append(int(pid_s))
        except Exception:
            pass
    return sorted(set(pids))


def _launch_stage(
    *,
    repo: Path,
    out_rel: str,
    config_name: str,
    attack_name: str,
    stage_log_name: str,
) -> int:
    cmd = [
        str(repo / ".venv-neurinspectre" / "bin" / "neurinspectre"),
        "table2",
        "-c",
        config_name,
        "-o",
        out_rel,
        "--attacks",
        attack_name,
        "--resume",
        "--strict-real-data",
        "--strict-dataset-budgets",
        "--no-report",
        "--no-progress",
        "--device",
        "auto",
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    log_path = repo / out_rel / stage_log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        handle.write(f"\n\n# ---- launched by monitor at {_now()} ----\n".encode("utf-8"))
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return int(proc.pid)


def _run_stage(
    *,
    repo: Path,
    out_rel: str,
    config_name: str,
    attack_name: str,
    expected_defenses: int,
    poll_seconds: int,
    monitor_log: Path,
    stage_log_name: str,
) -> None:
    target = int(expected_defenses)
    while True:
        done, total, missing = _count_stage_done(repo / out_rel, attack_name)
        pids = _list_attack_worker_pids(config_name, out_rel, attack_name)
        with monitor_log.open("a", encoding="utf-8") as f:
            f.write(
                f"[{_now()}] stage={attack_name} done={done}/{target} "
                f"(visible_defense_files={total}) running_pids={pids}\n"
            )

        if done >= target and not pids:
            with monitor_log.open("a", encoding="utf-8") as f:
                f.write(f"[{_now()}] stage={attack_name} complete\n")
            return

        if done < target and not pids:
            with monitor_log.open("a", encoding="utf-8") as f:
                f.write(
                    f"[{_now()}] stage={attack_name} interrupted/incomplete; "
                    f"launching --resume (missing={missing})\n"
                )
            pid = _launch_stage(
                repo=repo,
                out_rel=out_rel,
                config_name=config_name,
                attack_name=attack_name,
                stage_log_name=stage_log_name,
            )
            with monitor_log.open("a", encoding="utf-8") as f:
                f.write(f"[{_now()}] stage={attack_name} launched pid={pid}\n")
            time.sleep(10)

        time.sleep(max(10, int(poll_seconds)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Chain AutoAttack -> NeurInSpectre for Table 8")
    ap.add_argument("--repo", type=Path, default=Path("."), help="Repo root")
    ap.add_argument("--config", type=str, default="table2_config.yaml", help="Table2 config filename")
    ap.add_argument("--out-rel", type=str, default="results/table8_run_v2", help="Output dir relative to repo")
    ap.add_argument("--expected-defenses", type=int, default=12, help="Expected number of defense rows")
    ap.add_argument("--poll-seconds", type=int, default=300, help="Monitor poll interval seconds")
    args = ap.parse_args()

    repo = args.repo.expanduser().resolve()
    out_dir = repo / args.out_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor_log = out_dir / "monitor_table8_v2.log"

    with monitor_log.open("a", encoding="utf-8") as f:
        f.write(f"[{_now()}] monitor start (autoattack -> neurinspectre)\n")

    _run_stage(
        repo=repo,
        out_rel=args.out_rel,
        config_name=args.config,
        attack_name="autoattack",
        expected_defenses=int(args.expected_defenses),
        poll_seconds=int(args.poll_seconds),
        monitor_log=monitor_log,
        stage_log_name="stage_autoattack_full.log",
    )

    _run_stage(
        repo=repo,
        out_rel=args.out_rel,
        config_name=args.config,
        attack_name="neurinspectre",
        expected_defenses=int(args.expected_defenses),
        poll_seconds=int(args.poll_seconds),
        monitor_log=monitor_log,
        stage_log_name="stage_neurinspectre_full.log",
    )

    attacks_seen = set()
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for row in summary.get("results", []) or []:
                if isinstance(row, dict) and isinstance(row.get("attacks"), dict):
                    attacks_seen.update((row.get("attacks") or {}).keys())
        except Exception:
            pass

    with monitor_log.open("a", encoding="utf-8") as f:
        f.write(f"[{_now()}] finished: summary_attacks_seen={sorted(attacks_seen)}\n")


if __name__ == "__main__":
    main()

