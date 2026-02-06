"""
CLI entrypoint for Table 1 evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from .table1_reproducer import Table1Reproducer


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    datasets = dict(config.get("datasets", {}))

    if args.n_samples is not None:
        for name, cfg in datasets.items():
            cfg = dict(cfg)
            cfg["n_samples"] = int(args.n_samples)
            datasets[name] = cfg

    if args.batch_size is not None:
        for name, cfg in datasets.items():
            cfg = dict(cfg)
            cfg["batch_size"] = int(args.batch_size)
            datasets[name] = cfg
        config["attack_batch_size"] = int(args.batch_size)

    if args.defenses:
        selected = set(args.defenses)
        defenses = config.get("defenses", {})
        config["defenses"] = {k: v for k, v in defenses.items() if k in selected}

    if args.no_cache:
        cache_cfg = dict(config.get("cache", {}))
        cache_cfg["enable_dataset_cache"] = False
        cache_cfg["enable_attack_checkpoint"] = False
        config["cache"] = cache_cfg

    config["datasets"] = datasets
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeurInSpectre Table 1 evaluator")
    parser.add_argument(
        "--config",
        default="experiments/configs/table1_config.yaml",
        help="Path to Table 1 YAML config",
    )
    parser.add_argument("--output-dir", default="results/table1", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=None, help="Override samples per dataset")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device")
    parser.add_argument(
        "--defenses",
        nargs="+",
        default=None,
        help="Subset of defenses to evaluate (space-separated)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset/attack caching")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing datasets/models")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = _load_config(args.config)
    config = _apply_overrides(config, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reproducer = Table1Reproducer(
        config_path=args.config,
        config=config,
        device=args.device,
        results_dir=str(out_dir),
        allow_missing=bool(args.allow_missing),
    )
    reproducer.run_evaluation()
    reproducer.print_table()
    reproducer.save_results(str(out_dir / "table1_results.csv"))
    if not args.no_plot:
        reproducer.plot_results(str(out_dir / "table1_results.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
