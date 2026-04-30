# NeurInSpectre — Offensive Paper Artifact (CCS '26 Cycle B)

This repository is the **public reproduction artifact** for the offensive
NeurInSpectre paper:

> *"NeurInSpectre: An Offensive Framework for Breaking Gradient Obfuscation
> in AI Safety Systems via Spectral-Volterra-Krylov Analysis"*
> ACM CCS 2026 Cycle B Submission.

It is a sibling of the **detection** paper repository (which uses a different
reproduction harness and target tables). Reviewers reproducing the offensive
paper should follow this README and `AE.md`; do not run the detection paper's
`scripts/reproduce_detection.sh`.

---

## What this artifact reproduces

| Paper claim | Status | Entry point | Output (under `results/`) |
|---|---|---|---|
| **Table 8** — 12 defenses × 3 attacks; +17.0 pp NeurInSpectre vs. AutoAttack on the validity-passed subset | **measured** | `bash scripts/reproduce_table8.sh` | `repro_harness_table8_*/table8_table2/` (+ `summary.json`, `table10_attack_strength.{json,tex}`, `table8_validation.json`, `table8_table2_artifacts.tgz`, `sha256_manifest.txt`) |
| **Table 5** — subnetwork hijack on WRN-28-10; both subnet- and full-hijack rows evade NC + SS at paper-recommended thresholds | **measured** | `python scripts/reproduce_module_table5.py --n-seeds 3 --baseline` | `table5_rigor_production/results.json` |
| **Table 4** — gradient inversion (design-target SSIM) | design-target | `python scripts/reproduce_module_table4.py` | (script header) |
| Statistical evasion / RL-obfuscation / attention security / EDNN / activation steganography | design-target (CLIs ship + run end-to-end) | per-module CLIs (`neurinspectre …`) | `audit_cli/` |

The paper transparently flags each row in `Table~\ref{tab:cross-module}` as
either measured (✓) or design-target (!). Audit details are in
`DRAFT_CODE_AUDIT.md`.

---

## Headline numbers from the as-run artifact

From `results/table8_run_v2/summary.json` (single-seed, `seed=42`,
n=1000 per row, A100 / CUDA 12.1 / PyTorch 2.10.0):

| Defense | Clean | PGD-20 | AutoAttack | NeurInSpectre | Validity |
|---|---:|---:|---:|---:|:---:|
| JPEG Compression (CIFAR-10) | 79.6% | 9.3% | 98.4% | **99.5%** | ✓ |
| Bit-Depth Reduction (CIFAR-10) | 88.5% | 9.4% | 99.0% | **100.0%** | ✓ |
| Ensemble Diversity (CIFAR-10) | 91.9% | 6.2% | 33.5% | **100.0%** | ✓ |
| Spatial Smoothing (nuScenes) | 55.0% | 93.2% | 100.0% | 87.0% | ✓ |
| Random Pad/Crop (nuScenes) | 62.5% | 44.0% | 100.0% | **100.0%** | ✓ |
| Thermometer Encoding (nuScenes) | 65.0% | 11.5% | 19.2% | **100.0%** | ✓ |
| EMBER Gradient Reg. | 70.9% | 0.0% | 0.0% | 0.0% | ✓ |
| EMBER Defensive Distillation | 70.9% | 0.0% | 0.0% | 0.0% | ✓ |
| **Mean (validity-passed, 8/12)** | — | 21.7% | 56.3% | **73.3%** | — |
| Mean (all 12, including 4 collapsed-clean rows) | — | 20.5% | 39.7% | 51.5% | — |

Subnetwork hijacking on WRN-28-10 (3 seeds, last-Linear protocol; from
`results/table5_rigor_production/results.json`):

- 6%-subnet: BD ASR **24.7%**, ΔAcc −18.9 pp, NC anomaly **1.43**, SS triggered-flag **37.5%**.
- BadNets baseline: BD ASR 23.6%, ΔAcc −11.45 pp, NC anomaly 1.76, SS 37.95%.
- Both rows evade NC (threshold 2.0) and SS (threshold 85%) at the paper-recommended operating points.

---

## Quickstart (≈30 minutes including dataset prep)

```bash
# 1. Environment
python3 -m venv .venv-neurinspectre
source .venv-neurinspectre/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements-frozen.txt

# 2. Datasets
python scripts/download_cifar10.py --root ./data/cifar10 --split both
python scripts/download_ember2018.py --root ./data/ember
python scripts/vectorize_ember_safe.py
# nuScenes v1.0-mini: download manually from https://nuscenes.org (license-gated),
# place under ./data/nuscenes/v1.0-mini/, then:
python scripts/generate_nuscenes_label_map.py \
  --dataroot ./data/nuscenes --version v1.0-mini \
  --output ./data/nuscenes/label_map.json

# 3. (Optional) re-train EMBER defense checkpoints
python scripts/train_ember_defense_models.py --train-standard

# 4. Reproduce Table 8 (~28 A100-hr or ~72 hr on M3 Pro)
bash scripts/reproduce_table8.sh

# 5. Reproduce Table 5 (subnetwork hijack; ~20 min on M3 Pro)
python scripts/reproduce_module_table5.py --n-seeds 3 --baseline

# 6. (Optional) paper-grade multi-seed Table 8
TABLE2_NUM_SEEDS=5 bash scripts/reproduce_table8.sh
```

A 5-minute smoke test that exercises the full pipeline without paper-grade
runtimes:

```bash
neurinspectre table2-smoke --output-dir results/smoke
```

Per-module wall-clock estimates are in Appendix A of the paper
(`Table~\ref{tab:runtime}`).

---

## Repository layout

```
.
├── AE.md                        # Quickstart for AE reviewers
├── REPRODUCE.md                 # Full reproduction guide (extended)
├── DRAFT_CODE_AUDIT.md          # Claim → code → command → artifact audit
├── README.md                    # This file (offensive-paper artifact entry)
│
├── neurinspectre/               # Python package (~77k LOC)
│   ├── attacks/                 # PGD, AutoAttack, BPDA, EOT, MA-PGD, hybrid, ...
│   ├── characterization/        # Layer 1/2/3 detectors
│   ├── defenses/                # 12 defense wrappers (table2_config.yaml)
│   ├── statistical/             # KS/AD/CvM + Fisher + BH FDR drift detectors
│   ├── security/                # 8-component RL-obfuscation detector
│   ├── cli/                     # Click CLI (table2, characterize, attack, ...)
│   └── ...
│
├── scripts/
│   ├── reproduce_table8.sh           # Single-command Table 8 harness
│   ├── reproduce_module_table5.py    # Subnetwork hijack reproduction (3 seeds)
│   ├── reproduce_module_table4.py    # Gradient inversion (design-target script)
│   ├── reproduce_all.sh              # Full harness (Table 8 + Table 10 export + manifest)
│   ├── download_cifar10.py
│   ├── download_ember2018.py
│   ├── vectorize_ember_safe.py
│   ├── generate_nuscenes_label_map.py
│   └── train_ember_defense_models.py
│
├── models/                      # TorchScript artifacts pinned by SHA-256
│   ├── cifar10_resnet20_norm_ts.pt          (sha256: 7b516d50...)
│   ├── ember_mlp_ts.pt
│   ├── md_gradient_reg_ember_ts.pt
│   ├── md_distillation_ember_ts.pt
│   ├── md_at_transform_ember_ts.pt
│   ├── nuscenes_resnet18_trained.pt
│   └── *.meta.json                          (SHA-256 + provenance per file)
│
├── results/
│   ├── table8_run_v2/           # Single-seed 12×3 matrix (Table 8 evidence)
│   └── table5_rigor_production/ # 3-seed subnetwork hijack (Table 5 evidence)
│
├── table2_config.yaml           # 12-defense matrix config consumed by `neurinspectre table2`
├── requirements-frozen.txt      # Pinned dependency versions
└── tests/                       # Unit + integration tests
```

---

## What is NOT in this repo (and why)

- **The submission PDF.** Excluded so the artifact is double-blind-friendly.
- **Standard benchmark datasets.** CIFAR-10 (auto-downloaded by torchvision),
  EMBER 2018 (license-restricted bulk archive), nuScenes v1.0-mini
  (account-gated). See Quickstart Step 2.
- **An ImageNet-100 trained checkpoint.** `models/imagenet100_resnet50.pt` is
  explicitly a stub (`"is_stub": true` in its `.meta.json`); the paper
  excludes ImageNet-100 from Table 8 for this reason.

---

## Provenance and integrity

Every model file under `models/` has a `.meta.json` sidecar containing a
SHA-256 hash of the TorchScript artifact, the architecture, dataset, and a
provenance note. To re-verify:

```bash
shasum -a 256 models/cifar10_resnet20_norm_ts.pt
# expected: 7b516d5013854ae954bb39e7319d4b9a7a071038b85f706eb274cba1c9ac3bce
```

The Table 8 reproduction harness emits a recursive `sha256_manifest.txt`
covering every JSON / log / YAML / tex output for full reviewer-side
integrity verification.

---

## Citing this artifact (post-acceptance)

A persistent DOI will be assigned via Zenodo upon paper acceptance. Until
then, cite via the GitHub URL and the commit hash currently in `HEAD`:

```bibtex
@misc{neurinspectre_offensive_artifact_2026,
  title  = {{NeurInSpectre} Offensive Paper Artifact (CCS '26)},
  author = {Anonymous Authors},
  year   = {2026},
  howpublished = {\url{https://github.com/packetmaven/NeurInSpectre_offsc}},
  note   = {Commit hash: <fill-in at submission time>}
}
```

---

## License

MIT (see `LICENSE`).

---

## Contact

For double-blind review questions, please use the OpenReview / HotCRP
submission system. Post-acceptance contact details will be added here.
