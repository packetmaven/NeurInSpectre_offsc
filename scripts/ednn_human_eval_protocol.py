#!/usr/bin/env python3
"""
EDNN Human Evaluation Protocol + Inter-Rater Agreement (Cohen's kappa).

Issue 10 (EDNN Underdeveloped):
  - Provide a formal, reproducible human-evaluation protocol artifact generator
  - Provide a scoring utility that computes Cohen's kappa (with sample sizes)

This script is intentionally "data-first": it does not generate synthetic text.
You must provide a real text-pairs file (JSONL) derived from real artifacts.

Input JSONL schema (one JSON object per line):
  Required keys (configurable via flags):
    - item_id (unique stable identifier)
    - original_text
    - candidate_text     (e.g., adversarial / reconstructed / modified)

  Optional keys (kept in the admin file; can be hidden from raters):
    - algorithm / recipe / attack / model / dataset / split / source_id / etc.

Outputs (prepare):
  - protocol.json
  - tasks_admin.jsonl            (full metadata + blinding map)
  - rater_<name>.csv             (annotation sheet template)

Outputs (score):
  - kappa_report.json            (per-column kappa + confusion matrix + N used)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            ln = line.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception as exc:
                raise ValueError(f"Invalid JSON on line {i}: {path}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {i} is not an object: {path}")
            yield obj


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _canonicalize_label(val: Any) -> Optional[str]:
    """
    Canonicalize common categorical labels for agreement.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s_low = s.lower()

    # Boolean-ish
    if s_low in {"y", "yes", "true", "1"}:
        return "yes"
    if s_low in {"n", "no", "false", "0"}:
        return "no"
    if s_low in {"unsure", "unknown", "na", "n/a"}:
        return "unsure"

    # Ordinal 1..5 (stored as strings for CSV)
    if s_low in {"1", "2", "3", "4", "5"}:
        return s_low

    # Fallback: keep as normalized token.
    return s_low


def cohen_kappa(
    rater_a: Sequence[Optional[str]],
    rater_b: Sequence[Optional[str]],
    *,
    allowed_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Cohen's kappa on paired categorical ratings.

    Returns a JSON-serializable dict with:
      - n_total, n_used, n_missing_pairs
      - labels (category set)
      - confusion (dict-of-dicts counts)
      - p_observed, p_expected, kappa
      - degenerate_marginals (when 1 - p_expected ~ 0)
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length.")

    pairs: List[Tuple[str, str]] = []
    n_total = int(len(rater_a))
    n_missing = 0
    for a, b in zip(rater_a, rater_b):
        ca = _canonicalize_label(a)
        cb = _canonicalize_label(b)
        if ca is None or cb is None:
            n_missing += 1
            continue
        pairs.append((ca, cb))

    n_used = int(len(pairs))
    if n_used == 0:
        return {
            "n_total": n_total,
            "n_used": 0,
            "n_missing_pairs": n_missing,
            "labels": [],
            "confusion": {},
            "p_observed": None,
            "p_expected": None,
            "kappa": None,
            "degenerate_marginals": False,
        }

    if allowed_labels is not None:
        labels = [str(x) for x in allowed_labels]
    else:
        labels = sorted({x for ab in pairs for x in ab})

    idx = {lab: i for i, lab in enumerate(labels)}
    k = int(len(labels))
    mat = np.zeros((k, k), dtype=np.int64)
    for a, b in pairs:
        if a not in idx or b not in idx:
            # If allowed_labels is set, unseen labels are ignored (protocol violation).
            continue
        mat[idx[a], idx[b]] += 1

    n_eff = int(mat.sum())
    if n_eff == 0:
        return {
            "n_total": n_total,
            "n_used": 0,
            "n_missing_pairs": n_missing,
            "labels": labels,
            "confusion": {},
            "p_observed": None,
            "p_expected": None,
            "kappa": None,
            "degenerate_marginals": False,
        }

    p_o = float(np.trace(mat) / n_eff)
    row_marg = mat.sum(axis=1).astype(np.float64) / float(n_eff)
    col_marg = mat.sum(axis=0).astype(np.float64) / float(n_eff)
    p_e = float(np.sum(row_marg * col_marg))

    denom = float(1.0 - p_e)
    degenerate = abs(denom) <= 1e-12
    if degenerate:
        # If marginals are degenerate, kappa isn't informative; follow a safe convention:
        # - if agreement is perfect -> kappa = 1
        # - else -> kappa = 0 (cannot exceed chance because chance is 1.0)
        kappa = 1.0 if (1.0 - p_o) <= 1e-12 else 0.0
    else:
        kappa = float((p_o - p_e) / denom)

    confusion: Dict[str, Dict[str, int]] = {}
    for i, la in enumerate(labels):
        confusion[la] = {}
        for j, lb in enumerate(labels):
            confusion[la][lb] = int(mat[i, j])

    return {
        "n_total": n_total,
        "n_used": int(n_eff),
        "n_missing_pairs": n_missing + (n_used - n_eff),
        "labels": labels,
        "confusion": confusion,
        "p_observed": p_o,
        "p_expected": p_e,
        "kappa": float(kappa),
        "degenerate_marginals": bool(degenerate),
    }


def _bootstrap_kappa_ci(
    pairs: List[Tuple[str, str]],
    *,
    labels: Sequence[str],
    trials: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Bootstrap percentile CI for kappa. Uses paired resampling.
    """
    if trials <= 0 or len(pairs) == 0:
        return {"enabled": False}

    rng = np.random.default_rng(int(seed))
    n = int(len(pairs))
    kappas: List[float] = []

    # Pre-split to arrays for fast sampling.
    a = np.array([p[0] for p in pairs], dtype=object)
    b = np.array([p[1] for p in pairs], dtype=object)

    for _ in range(int(trials)):
        idx = rng.integers(0, n, size=n)
        res = cohen_kappa(a[idx].tolist(), b[idx].tolist(), allowed_labels=list(labels))
        k = res.get("kappa")
        if k is None or not math.isfinite(float(k)):
            continue
        kappas.append(float(k))

    if not kappas:
        return {"enabled": True, "trials": int(trials), "n_success": 0, "ci95": None}

    ks = np.sort(np.asarray(kappas, dtype=np.float64))
    lo = float(np.quantile(ks, 0.025))
    hi = float(np.quantile(ks, 0.975))
    return {
        "enabled": True,
        "trials": int(trials),
        "n_success": int(len(kappas)),
        "ci95": [lo, hi],
    }


def _prepare_tasks(
    *,
    input_jsonl: Path,
    output_dir: Path,
    raters: Sequence[str],
    seed: int,
    sample_size: Optional[int],
    stratify_key: Optional[str],
    blind_pairs: bool,
    id_key: str,
    original_key: str,
    candidate_key: str,
) -> None:
    rows_in: List[Dict[str, Any]] = list(_iter_jsonl(input_jsonl))
    if not rows_in:
        raise SystemExit("[human-eval] Empty input JSONL.")

    # Validate + normalize.
    seen_ids: set[str] = set()
    items: List[Dict[str, Any]] = []
    for obj in rows_in:
        if id_key not in obj:
            raise SystemExit(f"[human-eval] Missing id_key={id_key!r} in one or more rows.")
        if original_key not in obj or candidate_key not in obj:
            raise SystemExit(
                f"[human-eval] Missing required keys: {original_key!r} and/or {candidate_key!r}."
            )
        item_id = str(obj[id_key]).strip()
        if not item_id:
            raise SystemExit("[human-eval] Empty item_id encountered.")
        if item_id in seen_ids:
            raise SystemExit(f"[human-eval] Duplicate item_id: {item_id}")
        seen_ids.add(item_id)

        orig = str(obj[original_key])
        cand = str(obj[candidate_key])
        items.append(
            {
                "item_id": item_id,
                "original_text": orig,
                "candidate_text": cand,
                "meta": {k: v for k, v in obj.items() if k not in {id_key, original_key, candidate_key}},
            }
        )

    rng = np.random.default_rng(int(seed))

    # Optional stratified sampling.
    if sample_size is not None:
        n_req = int(sample_size)
        if n_req <= 0:
            raise SystemExit("[human-eval] sample_size must be > 0.")
        if n_req > len(items):
            n_req = len(items)

        if stratify_key:
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for it in items:
                g = str((it.get("meta") or {}).get(stratify_key, "unknown")).strip().lower() or "unknown"
                groups.setdefault(g, []).append(it)
            group_keys = sorted(groups.keys())
            # Deterministic shuffle within each group.
            for g in group_keys:
                rng.shuffle(groups[g])

            base = n_req // max(1, len(group_keys))
            rem = n_req - base * len(group_keys)
            picked: List[Dict[str, Any]] = []
            for g in group_keys:
                take = min(base, len(groups[g]))
                picked.extend(groups[g][:take])
            # Distribute remainder in a round-robin.
            gi = 0
            while len(picked) < n_req and group_keys:
                g = group_keys[gi % len(group_keys)]
                already = sum(1 for it in picked if str((it.get("meta") or {}).get(stratify_key, "unknown")).strip().lower() == g)
                if already < len(groups[g]):
                    picked.append(groups[g][already])
                gi += 1
                if gi > 10_000:
                    break
            items = picked[:n_req]
        else:
            idx = rng.permutation(len(items))[:n_req]
            items = [items[int(i)] for i in idx]

    # Shuffle final task order.
    rng.shuffle(items)

    # Build admin tasks + rater view.
    tasks_admin_path = output_dir / "tasks_admin.jsonl"
    tasks_admin_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_admin_path.open("w", encoding="utf-8") as f_admin:
        for it in items:
            item_id = it["item_id"]
            orig = it["original_text"]
            cand = it["candidate_text"]

            if blind_pairs:
                if bool(rng.integers(0, 2) == 0):
                    text_a, text_b = orig, cand
                    a_is_original = True
                else:
                    text_a, text_b = cand, orig
                    a_is_original = False
            else:
                text_a, text_b = orig, cand
                a_is_original = True

            admin_obj = {
                "item_id": item_id,
                "text_a": text_a,
                "text_b": text_b,
                "a_is_original": bool(a_is_original),
                "original_sha256": _sha256_text(orig),
                "candidate_sha256": _sha256_text(cand),
                "meta": dict(it.get("meta") or {}),
            }
            f_admin.write(json.dumps(admin_obj, ensure_ascii=False) + "\n")

    protocol = {
        "protocol_id": "ednn_human_eval_v1",
        "created_at": _now_iso(),
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "seed": int(seed),
        "blind_pairs": bool(blind_pairs),
        "id_column": "item_id",
        "columns": {
            "meaning_preserved": {
                "type": "categorical",
                "labels": ["yes", "no", "unsure"],
                "prompt": "Do Text A and Text B preserve the same meaning (semantic equivalence)?",
            },
            "label_preserved": {
                "type": "categorical",
                "labels": ["yes", "no", "unsure"],
                "prompt": "Does the intended label/task outcome remain valid after the modification?",
            },
            "fluency": {
                "type": "categorical",
                "labels": ["1", "2", "3", "4", "5"],
                "prompt": "Fluency/grammaticality of the modified text (1=bad, 5=excellent).",
            },
            "notes": {
                "type": "free_text",
                "labels": None,
                "prompt": "Optional notes (not used for kappa).",
            },
        },
    }
    _write_json(output_dir / "protocol.json", protocol)

    # Emit rater sheets (CSV). Keep it simple: identical sheets for each rater.
    header = [
        "item_id",
        "text_a",
        "text_b",
        "meaning_preserved",
        "label_preserved",
        "fluency",
        "notes",
    ]
    for rater in raters:
        out_csv = output_dir / f"rater_{str(rater)}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f_csv:
            w = csv.DictWriter(f_csv, fieldnames=header)
            w.writeheader()
            # Re-read admin JSONL to ensure exact ordering.
            for obj in _iter_jsonl(tasks_admin_path):
                w.writerow(
                    {
                        "item_id": obj["item_id"],
                        "text_a": obj["text_a"],
                        "text_b": obj["text_b"],
                        "meaning_preserved": "",
                        "label_preserved": "",
                        "fluency": "",
                        "notes": "",
                    }
                )


def _read_rater_csv(path: Path, *, id_column: str) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or id_column not in set(r.fieldnames):
            raise ValueError(f"Missing id_column={id_column!r} in {path}")
        out: Dict[str, Dict[str, Any]] = {}
        for row in r:
            item_id = str(row.get(id_column, "")).strip()
            if not item_id:
                continue
            if item_id in out:
                raise ValueError(f"Duplicate item_id in rater CSV: {item_id} ({path})")
            out[item_id] = dict(row)
        return out


def _score_kappa(
    *,
    rater_a_path: Path,
    rater_b_path: Path,
    protocol_path: Optional[Path],
    output_path: Path,
    columns: Optional[Sequence[str]],
    id_column: str,
    bootstrap_trials: int,
    bootstrap_seed: int,
) -> None:
    protocol = None
    if protocol_path is not None and protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    ra = _read_rater_csv(rater_a_path, id_column=id_column)
    rb = _read_rater_csv(rater_b_path, id_column=id_column)

    ids = sorted(set(ra.keys()) & set(rb.keys()))
    if not ids:
        raise SystemExit("[human-eval] No overlapping item_id values between raters.")

    if columns is None:
        if protocol and isinstance(protocol.get("columns"), dict):
            columns = [c for c in protocol["columns"].keys() if c != "notes"]
        else:
            # Safe default.
            columns = ["meaning_preserved", "label_preserved", "fluency"]
    columns = [str(c) for c in columns]

    report: Dict[str, Any] = {
        "run": {
            "started_at": _now_iso(),
            "cwd": os.getcwd(),
            "rater_a": str(rater_a_path),
            "rater_b": str(rater_b_path),
            "protocol": str(protocol_path) if protocol_path else None,
            "id_column": str(id_column),
            "columns": list(columns),
            "bootstrap_trials": int(bootstrap_trials),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "overlap": {
            "n_overlap": int(len(ids)),
            "n_rater_a": int(len(ra)),
            "n_rater_b": int(len(rb)),
        },
        "per_column": {},
    }

    for col in columns:
        a_vals = [ra[i].get(col) for i in ids]
        b_vals = [rb[i].get(col) for i in ids]
        allowed = None
        if protocol and isinstance(protocol.get("columns"), dict):
            spec = protocol["columns"].get(col)
            if isinstance(spec, dict) and spec.get("type") == "categorical" and spec.get("labels"):
                allowed = [str(x) for x in list(spec["labels"])]
        base = cohen_kappa(a_vals, b_vals, allowed_labels=allowed)

        # Bootstrap CI uses the *paired* canonicalized labels.
        pairs: List[Tuple[str, str]] = []
        for av, bv in zip(a_vals, b_vals):
            ca = _canonicalize_label(av)
            cb = _canonicalize_label(bv)
            if ca is None or cb is None:
                continue
            pairs.append((ca, cb))
        ci = _bootstrap_kappa_ci(
            pairs,
            labels=base.get("labels") or (allowed or []),
            trials=int(bootstrap_trials),
            seed=int(bootstrap_seed),
        )
        base["bootstrap_ci"] = ci

        report["per_column"][col] = base

    _write_json(output_path, report)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="EDNN human evaluation protocol + Cohen's kappa scoring")
    sub = ap.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Generate rater sheets + admin mapping from JSONL")
    prep.add_argument("--input-jsonl", required=True, help="JSONL containing real text pairs")
    prep.add_argument("--output-dir", default="_cli_runs/ednn_human_eval", help="Output directory")
    prep.add_argument("--raters", nargs="+", default=["r1", "r2"], help="Rater identifiers (default: r1 r2)")
    prep.add_argument("--seed", type=int, default=0, help="Shuffle/sample seed (default: 0)")
    prep.add_argument("--sample-size", type=int, default=0, help="If >0: sample this many items")
    prep.add_argument("--stratify-key", type=str, default="", help="Optional metadata key to stratify sampling on")
    prep.add_argument("--blind-pairs", action="store_true", help="Blind which text is original vs candidate (A/B)")
    prep.add_argument("--id-key", type=str, default="item_id", help="JSONL key for item id")
    prep.add_argument("--original-key", type=str, default="original_text", help="JSONL key for original text")
    prep.add_argument("--candidate-key", type=str, default="candidate_text", help="JSONL key for candidate text")

    score = sub.add_parser("score", help="Compute Cohen's kappa from two completed rater CSVs")
    score.add_argument("--rater-a", required=True, help="CSV path for rater A")
    score.add_argument("--rater-b", required=True, help="CSV path for rater B")
    score.add_argument("--protocol", type=str, default="", help="protocol.json path (optional)")
    score.add_argument("--output", type=str, default="_cli_runs/ednn_human_eval/kappa_report.json", help="Output JSON")
    score.add_argument("--id-column", type=str, default="item_id", help="Item id column name (default: item_id)")
    score.add_argument("--columns", nargs="*", default=[], help="Columns to score (default: from protocol.json)")
    score.add_argument("--bootstrap-trials", type=int, default=1000, help="Bootstrap trials for 95% CI (default: 1000)")
    score.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed (default: 0)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "prepare":
        input_jsonl = Path(str(args.input_jsonl)).expanduser().resolve()
        out_dir = Path(str(args.output_dir)).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_size = int(args.sample_size) if int(args.sample_size) > 0 else None
        stratify_key = str(args.stratify_key).strip() or None
        raters = [str(r).strip() for r in (args.raters or []) if str(r).strip()]
        if len(raters) < 2:
            raise SystemExit("[human-eval] Provide at least 2 raters (Cohen's kappa requires two).")
        _prepare_tasks(
            input_jsonl=input_jsonl,
            output_dir=out_dir,
            raters=raters,
            seed=int(args.seed),
            sample_size=sample_size,
            stratify_key=stratify_key,
            blind_pairs=bool(args.blind_pairs),
            id_key=str(args.id_key),
            original_key=str(args.original_key),
            candidate_key=str(args.candidate_key),
        )
        print(f"[human-eval] Wrote: {out_dir / 'protocol.json'}")
        print(f"[human-eval] Wrote: {out_dir / 'tasks_admin.jsonl'}")
        for r in raters:
            print(f"[human-eval] Wrote: {out_dir / f'rater_{r}.csv'}")
        return 0

    if args.cmd == "score":
        rater_a = Path(str(args.rater_a)).expanduser().resolve()
        rater_b = Path(str(args.rater_b)).expanduser().resolve()
        protocol = Path(str(args.protocol)).expanduser().resolve() if str(args.protocol).strip() else None
        out_path = Path(str(args.output)).expanduser().resolve()
        cols = [str(c) for c in (args.columns or []) if str(c).strip()]
        _score_kappa(
            rater_a_path=rater_a,
            rater_b_path=rater_b,
            protocol_path=protocol,
            output_path=out_path,
            columns=cols or None,
            id_column=str(args.id_column),
            bootstrap_trials=int(args.bootstrap_trials),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        print(f"[human-eval] Wrote: {out_path}")
        return 0

    raise SystemExit("Unknown command.")


if __name__ == "__main__":
    raise SystemExit(main())

