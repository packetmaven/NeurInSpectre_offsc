"""Lightweight visualization helpers for attack diagnostics."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt


def plot_metric_over_steps(values: Iterable[float], *, title: str = "Metric over steps", ylabel: str = "Value"):
    """Quick diagnostic plot for attack optimization traces."""
    vals = list(values)
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=120)
    ax.plot(range(len(vals)), vals, color="#1f77b4", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig
