# NeurInSpectre — Detection Paper Artifact (CCS '26 Cycle B)

This repository is the **public reproduction artifact** for the detection
NeurInSpectre paper:

> *"NeurInSpectre: A Three-Layer Mathematical Framework for Gradient
> Obfuscation Detection in Adversarial Machine Learning"*
> ACM CCS 2026 Cycle B Submission.

It is a sibling of the **offensive** paper repository (which uses a different
reproduction harness and target tables). Reviewers reproducing the detection
paper should follow this README and `QUICKSTART_CCS.md`; do not run the
offensive paper's `scripts/reproduce_table8.sh` or
`scripts/reproduce_all.sh` for this review — they reproduce a different
paper.

---

## What this artifact reproduces

| Paper claim | Status | Entry point | Output (under `results/detection/`) |
|---|---|---|---|
| **Tables 1–4** — synthetic calibration sequences (Layer 1 spectral, Layer 2 Volterra, Layer 3 Krylov sweep) | **measured** | `python scripts/run_synthetic_experiments.py --output-dir results/detection` | `synthetic_experiments.json` (keys: `table1_calibration`, `table2_detection`, `table3_volterra`, `table4_krylov`) |
| **Table 5** — real CIFAR-10 defenses on Carmon2019 WRN-28-10 (clean / JPEG / rand-smoothing / spatial-smoothing) | **measured** | `python scripts/run_real_defense_experiments.py --output-dir results/detection` | `real_defense_characterization.json` (4 experiments) |
| **Figure 4** — dissipative Laplacian 3-panel | **measured** | `python scripts/LaPlacian.py` | `figures/laplacian_dissipative_3panel.png` |
| Figures 1–3, 5–8 | static assets | (none) | `figures/*.png` (vendored) |

The full per-element-to-command mapping for the detection paper is in
`QUICKSTART_CCS.md`. End-to-end one-command reproduction:
`bash scripts/reproduce_detection.sh`.

---

## Headline numbers from the as-run artifact

From `results/detection/real_defense_characterization.json`
(200-step PGD on Carmon2019Unlabeled WRN-28-10, $\epsilon{=}8/255$;
89.69% verified clean accuracy of the backbone):

| Defense | $\hat H_S$ | $R_{HF}$ | $\hat\alpha$ | Krylov err. | Composite verdict | Routed bypass |
|---|---:|---:|---:|---:|---|---|
| None (clean baseline) | 0.084 | 0.000 | 0.11 | 0.67 | obfuscated† | none via $\hat\alpha$ |
| JPEG ($q{=}75$) | 0.371 | 0.078 | 0.10 | 0.67 | obfuscated | shattered (grad-zero) |
| Randomized smoothing ($\sigma{=}0.25$) | 0.001 | 0.000 | 0.10 | 0.31 | obfuscated | stochastic (variance) |
| Spatial smoothing ($3{\times}3$, $\sigma{=}1$) | 0.004 | 0.001 | 0.10 | 0.37 | obfuscated† | none via $\hat\alpha$ |

†The composite verdict fires on the clean baseline because the L-BFGS-B
Volterra fit collapses to the box-constraint boundary
$\hat\alpha = 0.10 < 0.70$ on this backbone (documented degeneracy,
§3.2 + §5.9 of the paper). This is a 100% false-positive rate on the
clean (adversarially-trained) baseline; the paper §5.9 + §6.3 treats
this as the headline transfer gap and proposes the autocorrelation-slope
estimator of Table 3 (§3.2) as a degeneracy-free replacement.

The synthetic calibration tables (Tables 1–4) reproduce qualitatively:
clean < band-limited evasion < periodic bursts < shattered, and
Krylov reconstruction error decreases exponentially with subspace size $m$.

---

## Quickstart (≈3 hours on Apple M3 Pro / ≈30 minutes on 1× A100)

```bash
# 1. Environment
python3.10 -m venv .venv-neurinspectre
source .venv-neurinspectre/bin/activate
pip install -U pip
pip install -e ".[dev]"
neurinspectre doctor

# 2. One-command reproduction (Tables 1–5 + Figure 4)
bash scripts/reproduce_detection.sh

# Equivalent step-by-step:
python scripts/run_synthetic_experiments.py --output-dir results/detection
python scripts/download_carmon2019.py    # ~146MB, SHA256-verified
python scripts/run_real_defense_experiments.py --output-dir results/detection
python scripts/LaPlacian.py
```

A 5-second smoke test (Tables 1–4 only):

```bash
python scripts/run_synthetic_experiments.py --output-dir results/smoke
```

Per-element runtime estimates:

| Experiment | Apple M3 Pro (MPS) | 1× A100 |
|---|---|---|
| Synthetic experiments (Tables 1–4) | ~5 s | ~3 s |
| Real-defense experiments (Table 5) | ~3 h | ~30 min |
| Figure 4 regeneration | < 1 min | < 1 min |

---

## Repository layout (detection-paper-relevant entries)

```
.
├── QUICKSTART_CCS.md              # Detailed reproduction walkthrough (this paper)
├── README_DETECTION.md            # This file
├── README.md                      # Index README pointing to both paper artifacts
│
├── neurinspectre/
│   ├── characterization/          # Layer 1 (spectral), Layer 2 (Volterra), Layer 3 (Krylov)
│   ├── mathematical/              # ETD φ-functions, Krylov projections, fractional kernels
│   ├── models/wide_resnet_carmon.py    # Vendored Carmon2019 loader
│   └── ...
│
├── scripts/
│   ├── reproduce_detection.sh        # One-command harness (this paper)
│   ├── run_synthetic_experiments.py  # Tables 1–4
│   ├── run_real_defense_experiments.py  # Table 5
│   ├── download_carmon2019.py        # SHA256-verified checkpoint download
│   └── LaPlacian.py                  # Figure 4
│
├── models/
│   └── cifar10/Linf/Carmon2019Unlabeled.pt    # ~146MB; downloaded by step 2
│
├── results/
│   └── detection/                    # As-run detection-paper artifacts (Tables 1–5)
│
├── figures/                          # Static figures 1–3, 5–8 + Figure 4 output
└── tests/
```

---

## What is NOT in this repo (and why)

- **The submission PDF.** Excluded so the artifact is double-blind-friendly.
- **Standard benchmark datasets.** CIFAR-10 is auto-downloaded by torchvision
  during the first `run_real_defense_experiments.py` invocation.
- **The Carmon2019 checkpoint.** ~146 MB; fetched on demand by
  `python scripts/download_carmon2019.py` with SHA256 verification.

---

## Provenance and integrity

- The Carmon2019 checkpoint is SHA256-verified by
  `scripts/download_carmon2019.py` against the official RobustBench mirror.
- The reproduction harness emits a SHA256 manifest of all
  `results/detection/` outputs upon completion.

---

## Citing this artifact (post-acceptance)

A persistent DOI will be assigned via Zenodo upon paper acceptance. Until
then, cite via the GitHub URL and the commit hash currently in `HEAD`:

```bibtex
@misc{neurinspectre_detection_artifact_2026,
  title  = {{NeurInSpectre} Detection Paper Artifact (CCS '26)},
  author = {Anonymous Authors},
  year   = {2026},
  howpublished = {\url{https://github.com/packetmaven/Neurinspectre}},
  note   = {Commit hash: <fill-in at submission time>}
}
```

---

## Companion offensive-framework paper

This codebase also supports reproduction of a separate CCS '26 submission
on the offensive framework (7 attack modules + 12-defense evasion matrix +
Table 5 subnetwork hijack). Those experiments are **independent** of the
detection paper and use different commands (`bash scripts/reproduce_table8.sh`,
`python scripts/reproduce_module_table5.py`, etc.); see
`README_OFFSEC.md` and `QUICKSTART_CCS_OFFENSIVE.md`. Reviewers for **this**
(detection) paper do not need to run any offensive-paper commands.

---

## License

MIT (see `LICENSE`).

---

## Contact

For double-blind review questions, please use the OpenReview / HotCRP
submission system. Post-acceptance contact details will be added here.
