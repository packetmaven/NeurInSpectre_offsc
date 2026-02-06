#!/usr/bin/env python3
"""
Generate Figure 1: NeurInSpectre System Architecture Overview.

Outputs:
  figures/neurinspectre_architecture.pdf

This is a clean, vector PDF block diagram intended for inclusion in the CCS'26 paper:
  \\includegraphics[width=0.95\\columnwidth]{figures/neurinspectre_architecture.pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, xy, wh, title, subtitle="", *, fc="#FFFFFF", ec="#2C3E50"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
    )
    # Subtle shadow improves depth in print without looking "dashboard-y".
    patch.set_path_effects(
        [
            pe.withSimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace=(0, 0, 0), alpha=0.10),
            pe.Normal(),
        ]
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.2,
            color="#333",
        )
    return patch


def _arrow(ax, p0, p1, *, color="#2C3E50"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            shrinkA=6,
            shrinkB=6,
        )
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "neurinspectre_architecture.pdf"
    out_png = out_dir / "neurinspectre_architecture.png"

    # ACM-native-ish styling: serif (Times-like) + STIX math.
    with mpl.rc_context(
        {
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
            "pdf.fonttype": 42,  # TrueType text in PDF (nice for copy/paste)
            "ps.fonttype": 42,
        }
    ):
        # Slightly wider canvas to prevent any label crowding in 1-column layouts.
        fig = plt.figure(figsize=(7.6, 2.35), dpi=220)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Layout grid
        y0 = 0.25
        h = 0.50
        # IMPORTANT: Ensure everything fits inside [0,1] to avoid clipping/overlap.
        # Total width budget: left_margin + (sum widths) + (4 gaps) + right_margin <= 1.0
        left_margin = 0.03
        gap = 0.025
        w_in = 0.18
        w_mid = 0.16
        w_out = 0.18

        x_in = left_margin
        x_s = x_in + w_in + gap
        x_v = x_s + w_mid + gap
        x_k = x_v + w_mid + gap
        x_out = x_k + w_mid + gap

        # Blocks (left-to-right pipeline, as requested)
        _box(
            ax,
            (x_in, y0),
            (w_in, h),
            "Input",
            "Gradient sequences\nT×D arrays\n(.npy/.npz)",
            fc="#F8FAFC",
        )

        _box(
            ax,
            (x_s, y0),
            (w_mid, h),
            "Layer 1",
            "Spectral-temporal\nPSD · entropy · wavelets",
            fc="#ECF3FF",
            ec="#1F5FBF",
        )

        _box(
            ax,
            (x_v, y0),
            (w_mid, h),
            "Layer 2",
            "Volterra memory\nα · c · RMSE",
            fc="#FFF7ED",
            ec="#C2410C",
        )

        _box(
            ax,
            (x_k, y0),
            (w_mid, h),
            "Layer 3",
            "Krylov + ETD\nDynamics · dissipation",
            fc="#F0FDF4",
            ec="#166534",
        )

        _box(
            ax,
            (x_out, y0),
            (w_out, h),
            "Output",
            "Obfuscation verdict\n+ confidence score\n(JSON + plots/report)",
            fc="#FDF2F8",
            ec="#9D174D",
        )

        # Arrows
        # Place arrows slightly above the subtitle line to reduce visual clutter.
        y_mid = y0 + h * 0.56
        _arrow(ax, (x_in + w_in, y_mid), (x_s, y_mid))
        _arrow(ax, (x_s + w_mid, y_mid), (x_v, y_mid))
        _arrow(ax, (x_v + w_mid, y_mid), (x_k, y_mid))
        _arrow(ax, (x_k + w_mid, y_mid), (x_out, y_mid))

        # Minimal footer (keeps figure self-explanatory without crowding).
        ax.text(
            0.5,
            0.065,
            "NeurInSpectre: defense-in-depth detection of gradient obfuscation via complementary mathematical lenses",
            ha="center",
            va="center",
            fontsize=9.0,
            color="#444",
        )

        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=240)
        plt.close(fig)

    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


