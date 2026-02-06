#!/usr/bin/env python3
"""
Generate Figure 3: Volterra Memory Kernel Comparison.

Outputs:
  figures/volterra_kernels.pdf   (vector, publishable)
  figures/volterra_kernels.png   (preview)

Panels (a)-(c):
  (a) Power-law kernel: alpha = 0.3, 0.7, 0.9
  (b) Exponential kernel: lambda = 0.5, 1.0, 2.0
  (c) Matérn kernel: nu = 0.5, 1.5, 2.5  (half-integer closed forms)

Notes on veracity:
- NeurInSpectre's Layer-2 documentation and blog narrative discuss these three kernel families.
- The current CLI `neurinspectre math volterra` implements power-law fitting; this figure is a
  mathematical companion illustrating the kernel families used/considered in the Volterra analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def power_law_kernel(tau: np.ndarray, *, alpha: float, c: float = 1.0) -> np.ndarray:
    """K(tau) = c * tau^(alpha-1) / Gamma(alpha), tau>0."""
    from scipy.special import gamma

    tau = np.asarray(tau, dtype=np.float64)
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0,1)")
    return float(c) * (tau ** (a - 1.0)) / float(gamma(a))


def exponential_kernel(tau: np.ndarray, *, lam: float) -> np.ndarray:
    """K(tau) = exp(-lam * tau), tau>=0 (Markovian decay; normalized to K(0)=1)."""
    lam = float(lam)
    tau = np.asarray(tau, dtype=np.float64)
    return np.exp(-lam * tau)


def matern_kernel_half_integer(tau: np.ndarray, *, nu: float, rho: float = 1.0) -> np.ndarray:
    """
    Matérn-like kernel (half-integer closed forms), normalized so K(0)=1.

    Matches the Matérn parameterization used in the blog/paper:

        K(d) = 2^{1-ν}/Γ(ν) * ( √(2ν) * d / ρ )^ν * K_ν( √(2ν) * d / ρ )

    For half-integers ν ∈ {0.5, 1.5, 2.5}, closed forms exist with:
        x = √(2ν) * d / ρ
        ν=0.5:  exp(-x)
        ν=1.5:  (1 + x) exp(-x)
        ν=2.5:  (1 + x + x^2/3) exp(-x)
    """
    tau = np.asarray(tau, dtype=np.float64)
    nu = float(nu)
    x = np.sqrt(2.0 * nu) * (tau / float(rho))
    if abs(nu - 0.5) < 1e-12:
        return np.exp(-x)
    if abs(nu - 1.5) < 1e-12:
        return (1.0 + x) * np.exp(-x)
    if abs(nu - 2.5) < 1e-12:
        return (1.0 + x + (x * x) / 3.0) * np.exp(-x)
    raise ValueError("Only nu in {0.5, 1.5, 2.5} supported for this figure.")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "volterra_kernels.pdf"
    out_png = out_dir / "volterra_kernels.png"

    # Lag axis: log-spaced avoids the tau→0 singularity for power-law and shows tail behavior cleanly.
    tau = np.logspace(-2, np.log10(20.0), 600)

    alphas = [0.3, 0.7, 0.9]
    lams = [0.5, 1.0, 2.0]
    nus = [0.5, 1.5, 2.5]

    with mpl.rc_context(
        {
            # ACM-native-ish styling: serif (Times-like) + STIX math.
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Standardized axis/tick styling across paper figures
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "axes.linewidth": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "grid.linewidth": 0.8,
        }
    ):
        # Slightly wider/taller canvas so legends can live outside the plot area without overlap.
        fig, axes = plt.subplots(3, 1, figsize=(6.9, 7.1), dpi=220, sharex=True)
        fig.subplots_adjust(right=0.8)

        # --- (a) Power-law
        ax = axes[0]
        for a in alphas:
            ax.plot(
                tau,
                power_law_kernel(tau, alpha=a, c=1.0),
                lw=2.0,
                label=rf"$\alpha={a:g}$",
            )
        ax.set_title("(a) Power-law kernel (fractional dynamics)")
        ax.set_ylabel(r"$K(\tau)$")
        ax.grid(True, alpha=0.22, linewidth=0.9)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.92,
            edgecolor="#D1D5DB",
        )
        ax.set_yscale("log")
        ax.tick_params(labelbottom=False)

        # --- (b) Exponential
        ax = axes[1]
        for lam in lams:
            ax.plot(
                tau,
                exponential_kernel(tau, lam=lam),
                lw=2.0,
                label=rf"$\lambda={lam:g}$",
            )
        ax.set_title("(b) Exponential kernel (Markovian decay)")
        ax.set_ylabel(r"$K(\tau)$")
        ax.grid(True, alpha=0.22, linewidth=0.9)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.92,
            edgecolor="#D1D5DB",
        )
        ax.set_yscale("log")
        ax.tick_params(labelbottom=False)

        # --- (c) Matérn (half-integer)
        ax = axes[2]
        for nu in nus:
            ax.plot(
                tau,
                matern_kernel_half_integer(tau, nu=nu, rho=1.0),
                lw=2.0,
                label=rf"$\nu={nu:g}$",
            )
        ax.set_title(r"(c) Matérn kernel (smooth memory; parameter $\nu$)")
        ax.set_ylabel(r"$K(\tau)$")
        ax.set_xlabel(r"Lag $\tau$ (steps)")
        ax.grid(True, alpha=0.22, linewidth=0.9)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.92,
            edgecolor="#D1D5DB",
        )
        ax.set_yscale("log")
        ax.set_xscale("log")

        # Make all axes log-x for consistency (set after creation so sharex works cleanly).
        for a in axes:
            a.set_xscale("log")
            for spine in ["top", "right"]:
                a.spines[spine].set_visible(False)

        # Larger inter-panel padding prevents title/tick collisions in tight ACM columns.
        fig.tight_layout(pad=0.9, h_pad=1.1)
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=240)
        plt.close(fig)

    print(str(out_pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


