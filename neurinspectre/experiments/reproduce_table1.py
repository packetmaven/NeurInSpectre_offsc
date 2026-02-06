"""
Table 1 reproduction entrypoint (real datasets/models).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..evaluation.table1_reproducer import DefenseResult, Table1Reproducer


def reproduce_table1(
    config_path: str = "experiments/configs/table1_config.yaml",
    *,
    device: str = "auto",
    results_dir: str = "./results/table1",
    allow_missing: bool = False,
) -> Table1Reproducer:
    reproducer = Table1Reproducer(
        config_path=config_path,
        device=device,
        results_dir=results_dir,
        allow_missing=allow_missing,
    )
    reproducer.run_evaluation()
    return reproducer


__all__ = ["DefenseResult", "Table1Reproducer", "reproduce_table1"]
