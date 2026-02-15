"""Regression tests for import-cycle safety."""

from __future__ import annotations

import importlib


def test_cli_utils_and_table1_reproducer_import_without_cycle() -> None:
    """Import order that previously triggered a circular import."""
    importlib.import_module("neurinspectre.cli.utils")
    importlib.import_module("neurinspectre.evaluation.table1_reproducer")


def test_evaluation_lazy_export_table1_reproducer() -> None:
    """Package-level lazy export should still expose Table1Reproducer."""
    evaluation_pkg = importlib.import_module("neurinspectre.evaluation")
    table1_cls = getattr(evaluation_pkg, "Table1Reproducer")
    assert table1_cls.__name__ == "Table1Reproducer"
