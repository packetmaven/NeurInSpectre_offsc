# Quickstart: Reproducing CCS '26 Paper Results

**Paper:** "NeurInSpectre: A Three-Layer Mathematical Framework for Gradient
Obfuscation Detection in Adversarial Machine Learning"

This is the fastest path for a CCS reviewer to verify this paper's results.
A companion CCS '26 submission (the offensive framework paper) shares this
codebase and its reproduction artifacts are delivered separately per that
submission's artifact-evaluation timeline. For the paper-element-to-command
mapping used in this paper, see [REPRODUCE.md](REPRODUCE.md).

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
python scripts/run_synthetic_experiments.py --output-dir results
```

Generates `results/synthetic_experiments.json` containing Layer 1 (spectral),
Layer 2 (Volterra), and Layer 3 (Krylov) features on all synthetic sequence
classes discussed in Section 5.1 of the paper (clean baseline, shattered
gradients, stochastic defense, periodic bursts, band-limited evasion,
spectral-shaped evasion), plus the Krylov accuracy sweep (Table 4).

**Expected console output:**
- Per-sequence feature summary (one block per signature class)
- Krylov sweep showing exponential convergence of relative error with m

**Note on numerical values:** Tables 1–4 reproduce **exactly** at the
declared `seed=42, T=256`. All rows and columns match the paper to the
decimal places shown (Table 1: 5 rows × 3 ranges; Table 2: 6 rows;
Table 3: 4 rows; Table 4: 4 Krylov dimensions with machine-precision
relative error). Runtime: ~2 seconds on CPU.

## 2.5. Download the Carmon2019 WRN-28-10 (~150 MB, required for Table 5)

```bash
python scripts/download_carmon2019.py
```

This fetches the RobustBench Linf `Carmon2019Unlabeled` WRN-28-10 checkpoint
(the adversarially-trained CIFAR-10 model used in §5.9 / Table 5) and places
it at `models/cifar10/Linf/Carmon2019Unlabeled.pt`. The model is not shipped
in the repository because it exceeds GitHub's / anonymous.4open.science's
per-file size limits. The script auto-installs `robustbench` on first run
if it is missing.

## 3. Reproduce Table 5 (real CIFAR-10 defenses, ~3 hours on MPS / ~30 min on A100)

```bash
python scripts/run_real_defense_experiments.py --output-dir results
```

Requires:
- `models/cifar10/Linf/Carmon2019Unlabeled.pt` — obtain via step 2.5.
- `data/cifar10/` (auto-downloaded via torchvision on first run)

Generates `results/real_defense_characterization.json` with four configurations:
clean baseline, JPEG compression ($q{=}75$), randomized smoothing ($\sigma{=}0.25$),
and spatial smoothing ($3{\times}3$ Gaussian, $\sigma{=}1.0$). Each configuration
runs 200-step PGD gradient collection at $\epsilon{=}8/255$ through the
three-layer characterization pipeline.

**Expected results** (matching Table 5 in the paper):

| Defense | H_S | R_HF | alpha | Krylov err. | Classifier |
|---|---|---|---|---|---|
| None (clean) | 0.002 | 0.000 | 0.10 | 0.78 | none |
| JPEG (q=75) | 0.000 | 0.000 | 0.10 | 0.77 | shattered |
| Rand. smoothing | 0.001 | 0.000 | 0.17 | 0.80 | stochastic |
| Spatial smooth. | **0.487** | **0.397** | 0.10 | **1.23** | none* |

*See paragraph (2) in Section 5.9 — spatial smoothing's $R_{HF}$ exceeds the
0.30 threshold; the multi-feature classifier does not yet incorporate this
threshold as a standalone trigger (known integration gap).

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

## Companion Offensive Paper (CCS '26)

This same repository supports reproduction of a companion CCS '26 submission
on the offensive framework (7 attack modules + 12-defense evasion matrix).
Those experiments use different commands (`neurinspectre table2`,
`bash scripts/reproduce_table8.sh`, module-specific CLIs). They are
**not** required to verify this (detection-framework) paper. Artifact
documentation for the offensive paper is delivered separately per its
artifact-evaluation timeline.
