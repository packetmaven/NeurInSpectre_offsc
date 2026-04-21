
#!/usr/bin/env python3
"""
generate_laplacian_figure.py
----------------------------
Generates figures/laplacian_dissipative_3panel.png for NeurInSpectre main.tex.

Derived directly from neurinspectre/mathematical/krylov.py
(GitHub: packetmaven/Neurinspectre) -- build_L / laplacian_1d_matvec logic.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import expm
import os

os.makedirs('figures', exist_ok=True)

N_DIM, DT, DAMPING = 32, 0.5, 0.1

def build_L_matrix(n, damping=0.1):
    L = np.zeros((n, n))
    for i in range(1, n - 1):
        L[i, i - 1] = 1.0
        L[i, i]     = -2.0
        L[i, i + 1] = 1.0
    L[0, 0] = -1.0
    L[0, 1] = 1.0
    L[-1, -2] = 1.0
    L[-1, -1] = -1.0
    L -= damping * np.eye(n)
    return L

L = build_L_matrix(N_DIM, DAMPING)
eigvals = np.sort(np.linalg.eigvalsh(L))[::-1]
damp_f = np.exp(DT * eigvals)
modes = np.arange(N_DIM)

np.random.seed(42)
xv = np.linspace(0, 2 * np.pi, N_DIM)
u0 = np.sin(xv) + 0.3 * np.sin(3 * xv) + 0.05 * np.random.randn(N_DIM)
u1 = expm(DT * L) @ u0

ev0_str = f"{eigvals[0]:.2f}"
stiff_str = f"{abs(eigvals[-1] / eigvals[0]):.0f}"

BLUE = '#2563EB'
RED = '#DC2626'
ORANGE = '#D97706'
GRAY = '#6B7280'
LGRAY = '#F3F4F6'

fig = plt.figure(figsize=(14, 5.0))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(1, 3, wspace=0.42, left=0.06, right=0.97, top=0.80, bottom=0.15)

ax1 = fig.add_subplot(gs[0])
NS = 12
Ls = L[:NS, :NS]
norm = TwoSlopeNorm(vmin=Ls.min(), vcenter=0, vmax=max(Ls.max(), 0.01))
im = ax1.imshow(Ls, cmap='RdBu_r', norm=norm, aspect='auto')
for i in range(NS):
    for j in range(NS):
        v = Ls[i, j]
        if abs(v) > 1e-4:
            ax1.text(j, i, f'{v:.1f}', ha='center', va='center',
                     fontsize=5.5, color='white' if abs(v) > 0.8 else 'black')
cb = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=7)
ax1.set_xticks([0, 3, 6, 9, 11])
ax1.set_yticks([0, 3, 6, 9, 11])
ax1.tick_params(labelsize=7)
ax1.set_title(r'(a) $L$ matrix' + '\n' + r'(Neumann BCs, $\gamma=0.1$)', fontsize=9, fontweight='bold', pad=5)
ax1.set_xlabel(r'column $j$', fontsize=8)
ax1.set_ylabel(r'row $i$', fontsize=8)
ax1.annotate(r'BDY: $[-1,\,1,\,0\ldots]$', xy=(0.5, -0.14), xycoords='axes fraction',
             ha='center', fontsize=7, color=RED)

ax2 = fig.add_subplot(gs[1])
ax2.plot(xv, u0, color=BLUE, lw=2.0, label=r'$u(t)$')
ax2.plot(xv, u1, color=ORANGE, lw=2.0, linestyle='--', label=r'$u(t+\Delta t)$')
ax2.fill_between(xv, u0, u1, alpha=0.15, color=ORANGE)
ax2.axhline(0, color=GRAY, lw=0.6, ls=':')
ax2.set_title('(b) ETD diffusion step\n(high-freq. modes decay faster)', fontsize=9, fontweight='bold', pad=5)
ax2.set_xlabel(r'spatial index $i$', fontsize=8)
ax2.set_ylabel('amplitude', fontsize=8)
ax2.tick_params(labelsize=7)
ax2.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
ax2.set_facecolor(LGRAY)
ax2.spines[['top', 'right']].set_visible(False)

ax3 = fig.add_subplot(gs[2])
lns1 = ax3.plot(modes, eigvals, color=BLUE, lw=1.8, marker='o', markersize=3,
                label=r'$\lambda_k$', zorder=3)
ax3.annotate(r'$\lambda_0=' + ev0_str + '$',
             xy=(0, eigvals[0]), xytext=(8, eigvals[0] - 0.35),
             fontsize=7, color=BLUE,
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
ax3b = ax3.twinx()
lns2 = ax3b.plot(modes, damp_f, color=RED, lw=1.6, linestyle='--', marker='s',
                 markersize=2.5, label=r'$e^{\Delta t\lambda_k}$', zorder=3)
ax3b.axhline(1.0, color=GRAY, lw=0.7, ls=':')
ax3b.set_ylabel(r'damping factor $e^{\Delta t \lambda_k}$', fontsize=8, color=RED)
ax3b.tick_params(axis='y', labelcolor=RED, labelsize=7)
ax3b.set_ylim(-0.05, 1.18)
ax3.set_title(r'(c) Eigenvalue spectrum' + '\n' +
              r'($\lambda_k \leq 0$; stiffness $\approx$' + stiff_str + r'$\times$)',
              fontsize=9, fontweight='bold', pad=5)
ax3.set_xlabel(r'mode index $k$', fontsize=8)
ax3.set_ylabel(r'$\lambda_k$', fontsize=8, color=BLUE)
ax3.tick_params(axis='y', labelcolor=BLUE, labelsize=7)
ax3.tick_params(axis='x', labelsize=7)
ax3.set_facecolor(LGRAY)
ax3.spines[['top']].set_visible(False)
ax3.annotate('', xy=(N_DIM - 1, eigvals[-1]), xytext=(N_DIM - 1, eigvals[0]),
             arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=1.3))
ax3.text(N_DIM - 2.2, (eigvals[0] + eigvals[-1]) / 2,
         stiff_str + r'$\times$' + '\nstiff',
         fontsize=7, color=ORANGE, ha='right', va='center')
lns = lns1 + lns2
ax3.legend(lns, [l.get_label() for l in lns], fontsize=7.5, loc='lower left', framealpha=0.85)

fig.suptitle(r'Dissipative Laplacian $L$: Structure, ETD Diffusion, and Eigenvalue Spectrum (Layer~3)',
             fontsize=10.5, fontweight='bold', y=0.98)

plt.savefig('figures/laplacian_dissipative_3panel.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()