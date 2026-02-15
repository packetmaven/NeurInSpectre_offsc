"""Evaluation utilities (Table 1 reproduction)."""

from __future__ import annotations

from typing import Any

__all__ = ["Table1Reproducer"]


def __getattr__(name: str) -> Any:
    """
    Lazily expose heavy evaluation symbols.

    This avoids import-time circular dependencies between ``cli.utils`` and
    ``evaluation.table1_reproducer`` while preserving
    ``from neurinspectre.evaluation import Table1Reproducer`` behavior.
    """
    if name == "Table1Reproducer":
        from .table1_reproducer import Table1Reproducer

        return Table1Reproducer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
