# Quickstart: Reproducing CCS '26 Offensive-Framework Paper Results

> **Scope of this document.** This quickstart reproduces **one specific
> CCS '26 submission**: the *offensive-framework* paper
> ("NeurInSpectre: An Offensive Framework for Breaking Gradient
> Obfuscation in AI Safety Systems via Spectral-Volterra-Krylov Analysis").
> It is **not** the reproduction guide for the companion
> detection-framework submission that shares this codebase; that paper
> has a separate, self-contained quickstart at
> [QUICKSTART_CCS.md](QUICKSTART_CCS.md). The two reproduction paths are
> independent — you only need to follow **this** quickstart to evaluate
> **this** (offensive) paper.

**Paper:** "NeurInSpectre: An Offensive Framework for Breaking Gradient
Obfuscation in AI Safety Systems via Spectral-Volterra-Krylov Analysis"

**One-command TL;DR:** `bash scripts/reproduce_all.sh` runs every
reproducible step in this quickstart end-to-end (~28 A100-hours) and
writes all outputs under `results/offensive_<timestamp>/`. The
individual steps below let you run each piece independently.

**Output directory convention:** this quickstart writes to
`results/offensive_<timestamp>/`, `results/smoke/`,
`results/table5_rigor_production/`, and `results/table4_grad_inversion/`.
The companion detection paper writes to `results/detection/` (and
`figures/` for one figure) — there is no file-level collision between
the two reproduction paths.

This guide maps each paper table/claim to a single reproduction command
and — critically — distinguishes **shipped-reproducible** claims
(marked `[✓]`) from **design-target** claims (marked `[!]`) so that
reviewers know precisely what to expect from the shipped artifact.

For the full paper-element-to-command mapping including appendix tables,
see [REPRODUCE.md](REPRODUCE.md).

---

## Paper numbers vs shipped reproduction at a glance

The paper's Table "Cross-module evaluation summary" (`tab:cross-module`)
classifies every module's headline number as either `[✓]` (shipped
reproduction matches) or `[!]` (design target from originally-designed
protocol; shipped artifact does not reproduce the target number).

| Module / Table | Paper claim | Shipped reproduction | Script |
|---|---|---|---|
| Core evasion (Table "main-results") | **73.3% validity-conditional ASR** across 8/12 defenses; 51.5% all-12; +17.0pp over AutoAttack | **[✓] Reproducible** | `scripts/reproduce_table8.sh` |
| Subnetwork hijacking (Table "subnetwork", bottom group) | WRN-28-10 + last-Linear fine-tune: 24.7% BD ASR, −18.9% Δacc (6%-subnet); evades NC and SS at paper thresholds | **[✓] Reproducible** | `python scripts/reproduce_module_table5.py` |
| Subnetwork hijacking (Table "subnetwork", top group) | ResNet-50 + full-network training: 94–97% BD ASR, −0.3% Δacc | **[!] Design target** | — (protocol not shipped) |
| Gradient inversion (Table "grad-inversion", appendix) | 0.89 SSIM on CIFAR-10; H_S < 0.3 screening; 4.2× speedup | **[!] Design target** | `python scripts/reproduce_module_table4.py` (outputs actual numbers; expect SSIM ≪ 0.89 on ResNet-20 with BatchNorm and H_S ≈ 0.8 > 0.3) |
| Statistical evasion (Table "stat-evasion") | 10/12 per-test, 9/12 Fisher | **[!] Design target** | CLI available: `neurinspectre statistical-evasion score` |
| RL-obfuscation detection (Table "rl-detection") | 96.8% 8-component S_RL accuracy | **[!] Design target** (shipped `srl_weights.json` is zero-initialised placeholder → S_RL ≡ 0.5) | CLI available: `neurinspectre rl-obfuscation analyze` |
| EDNN embedding attacks (§5.4) | 91.7% targeted ASR on BERT/SST-2 | **[!] Design target** | CLI available: `neurinspectre adversarial-ednn` |
| Activation steganography (§5.2) | 3.2 bits/neuron on GPT-2 | **[!] Design target** | CLI available: `neurinspectre activation_steganography encode` |
| Attention analysis (§5.7) | 93.4% vulnerable-head detection | **[!] Design target** | CLI available: `neurinspectre attention-security` |

**Bottom line for reviewers.** Two paper rows have full shipped
reproductions: (a) the core 12-defense ASR matrix and (b) the subnetwork
hijack mechanism (WRN-28-10 bottom group). All other tables are
design-target rows from the originally-designed protocol; their CLIs run
end-to-end but do not reproduce the target numbers because calibrated
weights / pre-trained backbones for the target protocol are not shipped
in this submission. This is disclosed in the paper (`tab:cross-module`
and each module's §5.x caption).

---

## 1. Install (5 min)

```bash
python3.10 -m venv .venv-neurinspectre
source .venv-neurinspectre/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
neurinspectre doctor
```

`neurinspectre doctor` emits environment + GPU + dependency inventory
(no network required).

## 2. Smoke test the full pipeline (~5 min)

Auto-discovers what is runnable locally (datasets + models present) and
exercises the core evaluation stack:

```bash
neurinspectre table2-smoke --output-dir results/smoke
```

Produces `results/smoke/summary.json` (per-pair ASR / robust-accuracy) plus
`run_metadata.json` (environment + git SHA + config hash). This smoke test
is intentionally tolerant of missing optional datasets (ImageNet-100,
nuScenes) and will still succeed on CIFAR-10 + EMBER alone.

## 3. [✓] Core evasion: 12 defenses × 3 attacks (Table `main-results`)

```bash
bash scripts/reproduce_table8.sh
```

This runs PGD-20 → AutoAttack (resume) → NeurInSpectre (adaptive, resume)
across all 12 defended systems in three domains at their respective
threat-model budgets ($\ell_\infty = 8/255$ for CIFAR-10, $\ell_2 = 0.5$
for EMBER, $\ell_2 = 3.0$ for nuScenes). Writes:

- `results/repro_harness_table8_latest*/table8_table2/summary.json` — ASR matrix
- `results/repro_harness_table8_latest*/table8_table2/table10_attack_strength.tex` — per-dataset summary
- `table8_validation.json` — triplet coverage check (all three attacks per defense)

**Expected (from shipped `results/table8_run_v2/summary.json`, reported
verbatim in the paper's Table `main-results`):**

- Mean conditional ASR on validity-passed rows (8/12): **73.3%**
  (+17.0pp over AutoAttack's 56.3% on the same subset)
- Mean conditional ASR across all 12 rows: **51.5%**
- Four rows flagged `validity_passed=false` (randomized smoothing / two
  certified defenses / thermometer encoding) contribute 0% to the all-12
  average; this is the source of the 51.5 → 73.3 gap.

**Reviewer runtime:** ~2 h on NVIDIA A100 40GB; ~6 h on Apple M3 Pro MPS.

## 4. [✓] Subnetwork hijacking mechanism (Table `subnetwork` bottom group)

```bash
python scripts/reproduce_module_table5.py \
    --n-seeds 3 \
    --baseline \
    --output-dir results/table5_rigor_production
```

The `--baseline` flag additionally trains a BadNets full-hijack baseline
(no subnetwork mask) for direct comparison; both rows appear in the
paper's Table `subnetwork` bottom group.

This is a **rigorous mechanism-level reproduction** using gradient-based
neuron importance ranking (Molchanov et al. 2017), a BadNets-style
poisoning-rate training protocol, and faithful Neural Cleanse (Wang et al.
2019) + Spectral Signatures (Tran et al. 2018) detection implementations.
Three seeds are aggregated with mean ± std; the output matches the paper's
Table `subnetwork` bottom group.

**Expected on WRN-28-10 + last-Linear fine-tune (shipped model, 3 seeds):**

- 6%-subnet BD ASR: ~24.7%; Δacc ≈ −18.9%; NC not detected; SS not detected
- BadNets full-hijack: BD ASR ~23.6%; Δacc ≈ −11.5%; NC not detected; SS not detected

**Reviewer runtime:** ~35 min on A100; ~1.5 h on M3 Pro.

**Interpretation.** The mechanism-level finding (a subnetwork-restricted
hijack evades both detectors at the paper-recommended thresholds) is
reproduced. The absolute BD ASR and clean-accuracy preservation are lower
than the `ResNet-50 + full-network training` target in the top group of
the paper table because last-Linear fine-tuning has less backdoor capacity
than full-network training. The top group is a design target; full-network
training on ResNet-50 is listed as a follow-up deliverable.

## 5. [!] Gradient inversion (Appendix F Table `grad-inversion`)

```bash
python scripts/reproduce_module_table4.py \
    --n-samples 20 \
    --max-iter 300 \
    --output-dir results/table4_grad_inversion
```

The paper table is a **design target** (SSIM 0.89, H_S < 0.3 screening,
4.2× speedup). The shipped script runs end-to-end on CIFAR-10 ResNet-20
and emits actual measured numbers; these are expected to differ from the
target because:

1. Real CIFAR-10 ResNet-20 gradients have $\hat H_S \approx 0.79$, above
   the 0.3 screening threshold. The script correctly reports this and
   falls back to unscreened inversion.
2. DLG on a BatchNorm-equipped backbone yields SSIM $\approx 0.01$, not
   0.71; this is a well-known BN-breaks-DLG effect (Geiping et al. 2020).

The script's JSON output (`results/table4_grad_inversion/summary.json`)
records the actual measurements; the paper table is clearly flagged `[!]`
(design target) in `tab:cross-module`.

**Reviewer runtime:** ~5-15 min on M3 Pro / ~2-6 min on A100.

## 6. Per-module CLI invocations (design-target numbers from `tab:cross-module`)

All seven modules expose a CLI. Each of the commands below runs
end-to-end on reviewer-supplied inputs, but — as documented in
`tab:cross-module` (`[!]` rows) — the shipped calibration/weights do not
reproduce the paper's design-target accuracy numbers.

All seven per-module CLI example commands below write under
`results/offensive_modules/` to keep the offensive paper's outputs
namespaced away from the detection paper's `results/detection/`. Create
the directory once: `mkdir -p results/offensive_modules`.

### Gradient inversion

```bash
neurinspectre gradient-inversion recover \
    --gradients path/to/gradients.npy \
    --out-prefix results/offensive_modules/gradinv_
```

### Activation steganography

```bash
neurinspectre activation_steganography encode \
    --model gpt2 \
    --tokenizer gpt2 \
    --prompt "The quick brown fox jumps over the lazy dog" \
    --payload-bits 1,0,1,1,0,1,0,0 \
    --target-neurons 42,137,205,318,512,777,1024,1530 \
    --out-prefix results/offensive_modules/actsteg_
```

Payload is specified as a comma-separated bit-string (not free text) and
target neurons are integer indices into the target model's hidden state.
Paired `extract` subcommand: `neurinspectre activation_steganography extract --help`.

### Subnetwork hijack CLI (KMeans-based; **not** the rigorous reproduction — use `reproduce_module_table5.py` for mechanism validation)

```bash
neurinspectre subnetwork_hijack identify \
    --activations path/to/activations.npy \
    --n_clusters 8 \
    --out-prefix results/offensive_modules/snh_
```

### EDNN embedding attacks

```bash
neurinspectre adversarial-ednn \
    --attack-type inversion \
    --data path/to/embeddings.npy \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --device auto \
    --output-dir results/offensive_modules/ednn
```

### Statistical evasion (paper-grade KS/AD/CvM + Fisher + BH)

```bash
neurinspectre drift-detect \
    --reference path/to/clean_features.npy \
    --current path/to/adv_features.npy \
    --methods ks_ad_cvm \
    --confidence-level 0.95 \
    --output results/offensive_modules/stat_evasion.json
```

This runs the paper's full KS / AD / CvM test suite per feature dimension
with Fisher aggregation and Benjamini–Hochberg FDR correction. The
simpler `neurinspectre statistical-evasion score --method ks ...`
subcommand is a KS-only variant used by some smoke tests; the `ks_ad_cvm`
method above is the one referenced in the paper's `tab:stat-evasion`
caption.

### RL-obfuscation detection (shipped weights are zero-init placeholder)

```bash
neurinspectre rl-obfuscation analyze \
    --input-file path/to/gradient.npy \
    --sensitivity high \
    --output-report results/offensive_modules/rl_obf_report.json
```

Returns $S_{\mathrm{RL}} \equiv 0.5$ with the shipped
`srl_weights.json`; pass `--srl-weights <calibrated.json>` for a
calibrated run (not shipped in this submission). This is explicitly
flagged in the paper's Table `rl-detection` caption and `tab:cross-module`.

### Attention analysis

```bash
neurinspectre attention-security \
    --model gpt2 \
    --prompt "Hello world" \
    --output-png results/offensive_modules/attention_security.png \
    --out-json results/offensive_modules/attention_security.json
```

## 7. Full paper reproduction (~28 A100-hours)

Runs every table and module end-to-end. Writes a timestamped results
directory with a SHA256 manifest of all artifacts:

```bash
bash scripts/reproduce_all.sh
```

Emits:

- `table8_table2/summary.json` + `table8_validation.json`
- `table8_table2/table10_attack_strength.{json,tex}`
- `table_command_map.md`
- `sha256_manifest.txt`
- `table8_table2_artifacts.tgz` (bundled for submission)

Useful harness controls:

```bash
# Multi-seed Table "main-results" (CI95 + stddev across seeds):
TABLE2_NUM_SEEDS=5 RESULTS_DIR=results/repro_multiseed \
    bash scripts/reproduce_table8.sh

# Reuse an existing table2 run without rerunning heavy evaluation:
RESULTS_DIR=results/repackaged SKIP_CORE_EVASION=1 \
    TABLE2_REUSE_DIR=results/table8_run_v2 bash scripts/reproduce_all.sh

# Scope to Table "main-results" only (skip other module runners):
bash scripts/reproduce_table8.sh
```

## Expected Runtime Summary

| Experiment | A100 40GB | Apple M3 Pro MPS |
|---|---|---|
| Smoke test (`table2-smoke`) | ~5 min | ~12 min |
| Table `main-results` (12 defenses, single seed) | ~2 h | ~6 h |
| Subnetwork hijack mechanism (3 seeds) | ~35 min | ~1.5 h |
| Gradient inversion script (N=20) | ~6-10 min | ~15-25 min |
| Full reproduction (all seven modules + Table `main-results`) | ~28 h | ~72 h |

## Data and Model Requirements

By default, the CLI expects:

- `data/cifar10/` (auto-downloadable via `torchvision`)
- `data/imagenet/{train,val}/` (ImageFolder layout; reviewer-provided)
- `data/ember/ember_2018/{X_test.dat,y_test.dat,...}` (vectorized via
  `scripts/vectorize_ember_safe.py`)
- `data/nuscenes/` with `v1.0-mini/` and `label_map.json`
- `models/` with TorchScript artifacts. For the core evasion table, the
  key file is `Carmon2019Unlabeled.pt` (146 MB, adversarially-trained
  WRN-28-10 from RobustBench); download with
  `python scripts/download_carmon2019.py` (auto-installs `robustbench` in
  the venv, pins SHA256, ~1 min).

For AE-friendly discovery of what is runnable on disk, use
`neurinspectre table2-smoke`.

## Companion detection-framework paper (CCS '26) — NOT required

This same repository supports reproduction of a companion CCS '26
submission on the detection framework ("NeurInSpectre: A Three-Layer
Mathematical Framework for Gradient Obfuscation Detection in Adversarial
Machine Learning"). Those experiments are **independent** and use
different commands (`python scripts/run_synthetic_experiments.py`,
`python scripts/run_real_defense_experiments.py`, `python scripts/LaPlacian.py`)
with outputs written to `results/detection/`; they are documented in
[QUICKSTART_CCS.md](QUICKSTART_CCS.md).

> **Reviewer note.** You do **not** need to run any detection-paper
> command to verify the claims in this (offensive) paper. In particular,
> `bash scripts/reproduce_detection.sh` is the detection paper's
> reproduction harness, not this paper's; do not run it for this review.
> The one shared file, `scripts/run_real_defense_experiments.py`, is
> owned by the detection paper and is referenced here only as context
> for the cross-paper threshold-transfer discussion in §5.1 /
> Appendix~C of the detection paper; it is not required for any number
> reported in the offensive paper.
