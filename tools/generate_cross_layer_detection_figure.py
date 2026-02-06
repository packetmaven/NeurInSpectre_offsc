#!/usr/bin/env python3
"""
NeurInSpectre Figure 7: Cross-Layer Detection Performance Summary
=================================================================

Generates an ACM-ready bar chart showing detection accuracy across:
  - Layer 1 (Spectral-Temporal): 85% ± 3.2
  - Layer 2 (Volterra Memory):   78% ± 4.1
  - Layer 3 (Krylov Dynamics):   82% ± 3.8
  - Ensemble (3 layers):         94% ± 2.1

Outputs:
  figures/cross_layer_detection.pdf   (vector)
  figures/cross_layer_detection.png   (preview, 300 dpi)
  figures/cross_layer_detection_caption.tex (caption helper)

Design principles:
  - ACM sigconf column width (3.33 in)
  - Colorblind-safe palette + print-friendly hatching
  - Error bars show ±1σ across synthetic scenarios (PoC)
  - Explicit synthetic note in-figure to avoid over-claiming
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


DETECTION_RATES: Final[dict[str, tuple[float, float]]] = {
    "Layer 1\nSpectral": (85.0, 3.2),
    "Layer 2\nVolterra": (78.0, 4.1),
    "Layer 3\nKrylov": (82.0, 3.8),
    "Ensemble\n(3 layers)": (94.0, 2.1),
}

COLORS: Final[list[str]] = [
    "#1E3A8A",  # spectral
    "#B91C1C",  # volterra
    "#166534",  # krylov
    "#831843",  # ensemble
]
HATCHES: Final[list[str]] = ["///", "\\\\\\", "...", "xxx"]

ACM_COLUMN_WIDTH_INCHES: Final[float] = 3.33
FIGURE_HEIGHT_INCHES: Final[float] = 2.6


def setup_matplotlib_for_publication() -> dict[str, any]:
    return {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "STIX Two Text",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.9,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }


def create_detection_accuracy_plot(ax: plt.Axes, labels: list[str], means: np.ndarray, stds: np.ndarray) -> None:
    x = np.arange(len(labels), dtype=int)
    bars = ax.bar(
        x,
        means,
        color=COLORS,
        edgecolor="#1F2937",
        linewidth=0.9,
        width=0.60,
        yerr=stds,
        alpha=0.88,
        error_kw={
            "elinewidth": 1.0,
            "capsize": 3.5,
            "capthick": 1.0,
            "ecolor": "#374151",
        },
    )
    for rect, hatch in zip(bars, HATCHES):
        rect.set_hatch(hatch)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Detection accuracy (%)")
    ax.set_xticks(x, labels)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

    for rect, mean, std in zip(bars, means, stds):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + std + 2.0,
            f"{mean:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    if len(means) == 4:
        max_single = max(means[:3])
        ensemble = means[3]
        improvement = ensemble - max_single
        if improvement > 0:
            ax.annotate(
                f"+{improvement:.0f}% gain",
                xy=(3, ensemble - stds[3] * 0.4),
                xytext=(3.25, ensemble + 4.0),
                fontsize=7.2,
                ha="left",
                va="bottom",
                color=COLORS[3],
                arrowprops=dict(arrowstyle="->", color=COLORS[3], lw=1.0),
            )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#374151")
    ax.spines["bottom"].set_color("#374151")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def add_methodology_legend(ax: plt.Axes) -> None:
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=COLORS[0], ec="#1F2937", hatch=HATCHES[0], alpha=0.88, label="Spectral-temporal (FFT + wavelets)"),
        Rectangle((0, 0), 1, 1, fc=COLORS[1], ec="#1F2937", hatch=HATCHES[1], alpha=0.88, label="Volterra memory (power-law kernels)"),
        Rectangle((0, 0), 1, 1, fc=COLORS[2], ec="#1F2937", hatch=HATCHES[2], alpha=0.88, label="Krylov dynamics (ETD2 + subspace)"),
        Rectangle((0, 0), 1, 1, fc=COLORS[3], ec="#1F2937", hatch=HATCHES[3], alpha=0.88, label="Ensemble (majority voting)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        columnspacing=1.2,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#E5E7EB",
        fancybox=False,
        fontsize=6.5,
    )


def add_experimental_context(fig: plt.Figure, ax: plt.Axes) -> None:
    # Context call removed per request (avoid in-figure synthetic note).
    return None


def generate_latex_caption() -> str:
    return r"""\caption{Cross-layer detection performance on synthetic obfuscation signatures. 
Individual layers achieve 78--85\% accuracy; ensemble voting (majority across 3 layers) 
reaches 94\% by exploiting complementary mathematical signatures. 
Layer 1 (Spectral-Temporal): power spectral density + wavelet analysis detects high-frequency 
injection ($\hat{H} > 0.5$ or $R_{HF} > 0.3$). 
Layer 2 (Volterra): fractional calculus kernels identify non-Markovian memory ($\alpha < 0.7$). 
Layer 3 (Krylov): ETD2 + subspace projection reveals dissipative anomalies (score $> 0.75$). 
Aggregated from Tables~\ref{tab:poc-results}, \ref{tab:volterra-poc}, \ref{tab:krylov-error} 
across 6 synthetic scenarios: clean baseline, shattered gradients, stochastic defense, 
adversarial training artifacts, band-limited evasion, spectral-shaped evasion.}
\label{fig:cross-layer-detection}"""


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "cross_layer_detection.pdf"
    out_png = out_dir / "cross_layer_detection.png"
    out_caption = out_dir / "cross_layer_detection_caption.tex"

    labels = list(DETECTION_RATES.keys())
    means = np.array([v[0] for v in DETECTION_RATES.values()])
    stds = np.array([v[1] for v in DETECTION_RATES.values()])

    mpl_config = setup_matplotlib_for_publication()
    with mpl.rc_context(mpl_config):
        fig = plt.figure(figsize=(ACM_COLUMN_WIDTH_INCHES, FIGURE_HEIGHT_INCHES), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        create_detection_accuracy_plot(ax, labels, means, stds)
        add_methodology_legend(ax)
        fig.tight_layout(pad=0.35, rect=[0, 0.12, 1, 1])
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)

    out_caption.write_text(generate_latex_caption(), encoding="utf-8")

    print("✓ Generated Figure 7:")
    print(f"  PDF: {out_pdf}")
    print(f"  PNG: {out_png}")
    print(f"  Caption: {out_caption}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
