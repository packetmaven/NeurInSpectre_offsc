"""Precision and determinism utilities."""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def deterministic_mode(enabled: bool = True):
    """
    Context manager for deterministic attack runs.
    Useful for reproducibility and fair comparisons.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    if enabled:
        torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)
