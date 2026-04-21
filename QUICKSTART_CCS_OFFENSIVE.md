# Quickstart: Reproducing CCS '26 Offensive Paper Results

**Paper:** "NeurInSpectre: A Comprehensive Offensive Framework for Breaking
Gradient Obfuscation in AI Safety Systems via Spectral-Volterra-Krylov Analysis"

This is the fastest path for a CCS reviewer to verify the offensive paper's
results. The offensive paper reports **94.3% average ASR across 12 defenses**
plus seven additional offensive modules sharing the same spectral–Volterra–Krylov
infrastructure.

For the detection framework paper (shared repository), see [QUICKSTART_CCS.md](QUICKSTART_CCS.md).
For the full paper-element-to-command mapping, see [REPRODUCE.md](REPRODUCE.md).

## 1. Install (5 min)

```bash
python3.10 -m venv .venv-neurinspectre
source .venv-neurinspectre/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
neurinspectre doctor
```

## 2. Smoke test the full pipeline (~5 min)

Auto-discovers what is runnable locally (datasets + models present) and exercises
the core evaluation stack:

```bash
neurinspectre table2-smoke --output-dir results/smoke
```

Produces `results/smoke/summary.json` (per-pair ASR/robust-accuracy), plus
`run_metadata.json` (environment + git + config hash).

## 3. Table 8 (main evasion result, 12 defenses × 3 attacks, ~2 h on A100)

```bash
bash scripts/reproduce_table8.sh
```

This runs PGD-20 → AutoAttack (resume) → NeurInSpectre (adaptive, resume) across
all 12 defended systems in three domains (Content Moderation, Malware Detection,
AV Perception) at their respective threat-model budgets ($\ell_\infty = 8/255$
for vision, $\ell_2 = 0.5$ for EMBER, $\ell_2 = 3.0$ for nuScenes).

Produces:
- `results/repro_harness_table8_latest*/table8_table2/summary.json` (ASR matrix)
- `results/repro_harness_table8_latest*/table10_attack_strength.tex` (LaTeX Table 10)
- `table8_validation.json` (triplet coverage check)

**Expected highlights** (from a prior run, `results/table8_run_v2/summary.json`):
- `cm_ensemble_diversity/neurinspectre`: ASR=1.000
- `cm_bit_depth_reduction/neurinspectre`: ASR=1.000
- `av_random_pad_crop/neurinspectre`: ASR=1.000
- Mean conditional ASR on valid rows: 94.3% (matches paper Table 8)

## 4. Offensive module reproduction

Each module is runnable via its own CLI command. Inputs are either
synthetic-ready (gradient inversion on held-out samples) or require a target
model/data the user provides.

### Gradient inversion (Table 4, 0.89 SSIM on CIFAR-10)

```bash
neurinspectre gradient-inversion recover \
  --gradients path/to/gradients.npy \
  --out-prefix results/gradinv_
```

### Activation steganography (3.2 bits/neuron, §5.2)

```bash
neurinspectre activation_steganography encode \
  --model gpt2 --payload "covert message" \
  --output results/stegenc_
```

### Subnetwork hijacking (Table 5, 94.2% backdoor ASR)

```bash
neurinspectre subnetwork_hijack identify \
  --activations path/to/activations.npy \
  --n_clusters 8 --out-prefix results/snh_
```

### EDNN embedding attacks (§5.4, 91.7% ASR on BERT/SST-2)

```bash
neurinspectre adversarial-ednn \
  --attack-type inversion \
  --data path/to/embeddings.npy \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device auto --output-dir results/ednn
```

### Statistical evasion (Table 6, 10/12 defenses evaded)

```bash
neurinspectre statistical-evasion score \
  --clean-features path/to/clean.npy --adv-features path/to/adv.npy \
  --output results/stat_evasion.json
```

(Use `neurinspectre statistical-evasion generate` to synthesize adversarial
feature batches first; then `score` to run KS/AD/CvM and Fisher aggregation.)

### RL-obfuscation detection (Table 7, 96.8% accuracy)

```bash
neurinspectre rl-obfuscation analyze \
  --input-file path/to/gradient.npy \
  --sensitivity high \
  --output-report results/rl_obf_report.json
```

### Attention analysis (93.4% vulnerable-head detection)

```bash
neurinspectre attention-security \
  --model gpt2 --prompt "Hello world" \
  --output-png results/attention_security.png \
  --out-json results/attention_security.json
```

## 5. Full paper reproduction (~28 A100-hours)

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
# Multi-seed Table 8 (CI95 + stddev across seeds):
TABLE2_NUM_SEEDS=5 RESULTS_DIR=results/repro_multiseed \
  bash scripts/reproduce_table8.sh

# Reuse an existing table2 run without rerunning heavy evaluation:
RESULTS_DIR=results/repackaged SKIP_CORE_EVASION=1 \
  TABLE2_REUSE_DIR=results/table8_run_v2 bash scripts/reproduce_all.sh
```

## Expected Runtime Summary

| Experiment | A100 40GB | Apple M3 Pro MPS |
|---|---|---|
| Smoke test | ~5 min | ~12 min |
| Table 8 (12 defenses) | ~2 h | ~6 h |
| Full reproduction (all 7 modules + Table 8) | ~28 h | ~72 h |

## Data and Model Requirements

By default, the CLI expects:
- `data/cifar10/` (auto-downloadable via torchvision)
- `data/imagenet/{train,val}/` (ImageFolder layout; user-provided)
- `data/ember/ember_2018/{X_test.dat,y_test.dat,...}` (vectorized via
  `scripts/vectorize_ember_safe.py`)
- `data/nuscenes/` with `v1.0-mini/` and `label_map.json`
- `models/` with TorchScript artifacts (Carmon2019 for CIFAR-10 is the main one)

For AE-friendly discovery of what is runnable on disk, use
`neurinspectre table2-smoke`.

## Companion detection-framework paper

This repository also reproduces the CCS '26 detection-framework paper
("NeurInSpectre: A Three-Layer Mathematical Framework for Gradient Obfuscation
Detection in Adversarial Machine Learning"). See [QUICKSTART_CCS.md](QUICKSTART_CCS.md).
