# Quickstart: Reproducing CCS '26 Detection-Framework Paper Results

> **Scope of this document.** This quickstart reproduces **one specific
> CCS '26 submission**: the *detection-framework* paper
> ("NeurInSpectre: A Three-Layer Mathematical Framework for Gradient
> Obfuscation Detection in Adversarial Machine Learning"). It is **not**
> the reproduction guide for the companion offensive-framework submission
> that shares this codebase; that paper has a separate, self-contained
> quickstart at [QUICKSTART_CCS_OFFENSIVE.md](QUICKSTART_CCS_OFFENSIVE.md).
> The two reproduction paths are independent — you only need to follow
> **this** quickstart to evaluate this (detection) paper.

**Paper:** "NeurInSpectre: A Three-Layer Mathematical Framework for Gradient
Obfuscation Detection in Adversarial Machine Learning"

**One-command TL;DR:** `bash scripts/reproduce_detection.sh` runs every
step in this quickstart end-to-end and writes all outputs under
`results/detection/`. The individual steps below let you run each piece
independently if you prefer.

**Output directory convention:** this quickstart writes to
`results/detection/` (and `figures/` for the one figure). The companion
offensive paper writes to `results/offensive_<timestamp>/` and
`results/smoke/` — there is no file-level collision between the two
reproduction paths.

For the full paper-element-to-command mapping across both papers, see
[REPRODUCE.md](REPRODUCE.md).

## 1. Install (5 min)

```bash
python3.10 -m venv .venv-neurinspectre
source .venv-neurinspectre/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
neurinspectre doctor
```

## 2. Reproduce Tables 1-4 (synthetic signatures, ~5 seconds)

```bash
python scripts/run_synthetic_experiments.py --output-dir results/detection
```

Generates `results/detection/synthetic_experiments.json` containing Layer 1
(spectral), Layer 2 (Volterra), and Layer 3 (Krylov) features on all synthetic
sequence classes discussed in Section 5.1 of the paper (clean baseline,
shattered gradients, stochastic defense, periodic bursts, band-limited evasion,
spectral-shaped evasion), plus the Krylov accuracy sweep (Table 4).

**Expected console output:**
- Per-sequence feature summary (one block per signature class)
- Krylov sweep showing exponential convergence of relative error with m

**Note on numerical values:** The synthetic signatures are proof-of-concept
calibration sequences. The qualitative ordering of features (clean < evasions
< obfuscation) matches the paper; absolute numerical values depend on sequence
generation seeds and may differ from the paper's Tables 1-2 by implementation
details. The Krylov convergence in Table 4 is reproduced qualitatively: error
decreases rapidly with m for smooth inputs produced by dissipative operators.

## 3. Reproduce Table 5 (real CIFAR-10 defenses, ~3 hours on MPS / ~30 min on A100)

```bash
# One-time checkpoint download (146MB) — also verifies SHA256
python scripts/download_carmon2019.py

# Full Table 5 reproduction
python -u scripts/run_real_defense_experiments.py --output-dir results/detection \
    2>&1 | tee results/detection/real_defense.log
```

Requires:
- `models/cifar10/Linf/Carmon2019Unlabeled.pt` (~150MB; downloaded by
  the first command above, SHA256-verified)
- `data/cifar10/` (auto-downloaded via torchvision on first run)

Generates `results/detection/real_defense_characterization.json` with four
configurations: clean baseline, JPEG compression ($q{=}75$), randomized
smoothing ($\sigma{=}0.25$), and spatial smoothing ($3{\times}3$ Gaussian,
$\sigma{=}1.0$). Each configuration runs 200-step PGD gradient collection
at $\epsilon{=}8/255$ through the three-layer characterization pipeline
on the Carmon et al. 2019 adversarially-trained WRN-28-10 (vendored
loader in `neurinspectre/models/wide_resnet_carmon.py`; reproduces the
published 89.69% clean accuracy).

**Expected results** (matching Table 5 in the paper; reproduced
2026-04-22):

| Defense | H_S | R_HF | alpha | Krylov err. | Verdict | Class/trigger |
|---|---|---|---|---|---|---|
| None (clean) | 0.084 | 0.000 | 0.11 | 0.67 | obfuscated† | none via $\hat\alpha$ |
| JPEG (q=75) | 0.371 | 0.078 | 0.10 | 0.67 | obfuscated | shattered via grad-zero |
| Rand. smoothing | 0.001 | 0.000 | 0.10 | 0.31 | obfuscated | stochastic via variance |
| Spatial smooth. | 0.004 | 0.001 | 0.10 | 0.37 | obfuscated† | none via $\hat\alpha$ |

†The composite verdict fires on the clean baseline because the
L-BFGS-B Volterra fit collapses to the boundary $\hat\alpha = 0.10 < 0.70$
on this backbone (documented degeneracy, §3.2 + §5.9 paragraph (2)).
This is a 100% false-positive rate on the clean baseline; the paper
(§5.9 + §6.3) treats this as the headline transfer gap and proposes
the autocorrelation-slope estimator of Table 3 (§3.2) as a
degeneracy-free replacement.

## 4. Reproduce Figure 4 (dissipative Laplacian 3-panel, <1 min)

```bash
python scripts/LaPlacian.py
```

Generates `figures/laplacian_dissipative_3panel.png` showing the Laplacian
operator structure (panel a), one ETD diffusion step (panel b), and the
eigenvalue spectrum with damping factors (panel c, 41x stiffness ratio).

## 5. All other figures

Figures 1, 2, 3, 5, 6, 7, 8 are static assets in `figures/`:
- `figures/neurinspectre_architecture.png` (Figure 1)
- `figures/spectral_analysis_4panel.png` (Figure 2)
- `figures/volterra_kernels.png` (Figure 3)
- `figures/krylov_etd_evolution.png` (Figure 5)
- `figures/cross_layer_detection.png` (Figure 6)

## Expected Runtime Summary

| Experiment | Runtime (Apple M3 Pro MPS) | Runtime (1x A100) |
|---|---|---|
| Synthetic experiments (Tables 1-4) | ~5 seconds | ~3 seconds |
| Real defense experiments (Table 5) | ~3 hours | ~30 minutes |
| Figure 4 regeneration | <1 minute | <1 minute |

## Companion offensive-framework paper (CCS '26) — NOT required

This same repository supports reproduction of a companion CCS '26 submission
on the offensive framework (7 attack modules + 12-defense evasion matrix).
Those experiments are **independent** and use different commands
(`neurinspectre table2`, `bash scripts/reproduce_table8.sh`, module-specific
CLIs) with outputs written to `results/offensive_<timestamp>/`; they are
documented in [QUICKSTART_CCS_OFFENSIVE.md](QUICKSTART_CCS_OFFENSIVE.md).

> **Reviewer note.** You do **not** need to run any offensive-paper
> command to verify the claims in this (detection) paper. In particular,
> `scripts/reproduce_all.sh` is the offensive paper's ~28 A100-hour
> harness, not this paper's; do not run it for this review. The
> detection paper's equivalent one-command reproduction is
> `bash scripts/reproduce_detection.sh`.
