# Reproduction Guide — Two CCS '26 Papers (Shared Codebase)

> **Which paper are you reviewing?** This repository contains the artifact
> for **two independent CCS '26 submissions** sharing the same
> spectral–Volterra–Krylov codebase. The two reproduction paths are
> **independent** — a reviewer only needs to follow the quickstart
> corresponding to the paper they are evaluating. If you run both, they
> write to disjoint output directories so there is no collision.

| You are reviewing | Quickstart | One-command harness | Output dir |
|---|---|---|---|
| Detection paper (*"A Three-Layer Mathematical Framework..."*) | [QUICKSTART_CCS.md](QUICKSTART_CCS.md) | `bash scripts/reproduce_detection.sh` | `results/detection/` |
| Offensive paper (*"An Offensive Framework for Breaking Gradient Obfuscation..."*) | [QUICKSTART_CCS_OFFENSIVE.md](QUICKSTART_CCS_OFFENSIVE.md) | `bash scripts/reproduce_all.sh` | `results/offensive_<ts>/` |

The two harness scripts will **not** overwrite each other's outputs and can
be run in any order. Neither is required for reviewing the other paper.

The sections below preserve the detailed paper-element-to-command mapping
for both papers; if you only need a reviewer-facing entry point, the
quickstarts above are sufficient.

The repository is designed to run on real datasets and real model artifacts.
It intentionally does not hard-code paper baseline numbers in-code; validation
against expected values (if desired) must be supplied via external files.

If you only want the **verbatim Table 8 reproduction** quickstart (single command + expected outputs),
start at `AE.md`.

## Hardware Requirements

**Primary evaluation hardware:**

- GPU: NVIDIA A100 40GB (CUDA) or Apple Silicon M3/M4 Pro (MPS)
- RAM: >=32GB (full Table 2/8 evaluation); >=16GB (smoke tests)
- Storage: >=50GB (models + datasets)

**Minimum (smoke tests only):**

- Any GPU with >=8GB VRAM, or CPU-only (can be ~10x slower)
- RAM: >=16GB

## Software Stack

| Component | Version |
| --- | --- |
| Python | 3.10.x (required, see `pyproject.toml`) |
| PyTorch | >=2.0.0 |
| torchvision | >=0.18.1 |
| CUDA | 12.1+ (NVIDIA) or MPS (Apple Silicon) |
| OS | Ubuntu 22.04 LTS / macOS 14+ |

Note: `autoattack` is installed via a VCS dependency declared in `pyproject.toml`.

## Installation

```bash
# Double-blind friendly: start from the provided artifact archive.
# (Do not use a public git URL here; it deanonymizes the submission.)
tar -xzf <artifact>.tar.gz
cd <artifact_root>
python -m pip install -e ".[dev]"
neurinspectre doctor
```

## Docker (CUDA)

Build and run the CUDA container (requires NVIDIA Container Toolkit on the host):

```bash
docker build --platform=linux/amd64 -t neurinspectre:ae .
docker run --gpus all --rm -it neurinspectre:ae doctor

# To run evaluations, mount data/ and models/ (excluded from image for size):
docker run --gpus all --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  neurinspectre:ae table2 --config table2_config.yaml --strict-real-data
```

## Smoke Test (5 minutes)

For a fast end-to-end check (auto-discovers what is runnable locally):

```bash
neurinspectre table2-smoke --output-dir results/smoke
```

Outputs include:

- `results/smoke/run_metadata.json` (environment + git + config hash)
- `results/smoke/summary.json` (evaluation summary)

## Paper Element → Command Mapping

### Paper 1: CCS '26 — Detection Framework (Spectral–Volterra–Krylov)

For a fast reviewer-focused quickstart, see [QUICKSTART_CCS.md](QUICKSTART_CCS.md).

| Paper Element | Command / Source | Est. Runtime | Hardware |
|---|---|---|---|
| Tables 1–4 (synthetic signatures) | `python scripts/run_synthetic_experiments.py --output-dir results` | ~5 s | CPU |
| Table 5 (real CIFAR-10 defenses) | `python scripts/run_real_defense_experiments.py --output-dir results` | ~3 h (MPS) / ~30 min (A100) | GPU |
| Figure 1 (architecture diagram) | `figures/neurinspectre_architecture.png` (static asset) | — | — |
| Figure 2 (spectral-temporal) | `figures/spectral_analysis_4panel.png` (static asset) | — | — |
| Figure 3 (kernel families) | `figures/volterra_kernels.png` (static asset) | — | — |
| Figure 4 (Laplacian 3-panel) | `python scripts/LaPlacian.py` → `figures/laplacian_dissipative_3panel.png` | <1 min | CPU |
| Figure 5 (guardrail diagnostic) | `figures/krylov_etd_evolution.png` (static asset) | — | — |
| Figure 6 (cross-layer detection) | `figures/cross_layer_detection.png` (static asset) | — | — |

**Note:** Tables 1–4 are proof-of-concept calibration sequences. The script
reproduces the qualitative feature ordering and the Krylov convergence pattern;
absolute numerical values depend on sequence generation seeds. The main
reviewer-facing experiment is Table 5 (real defenses).

### Paper 2: CCS '26 — Offensive Framework (7 attack modules + 12-defense evasion)

For a reviewer-focused quickstart, see [QUICKSTART_CCS_OFFENSIVE.md](QUICKSTART_CCS_OFFENSIVE.md).

The paper's `tab:cross-module` classifies every row as either **[✓]** (shipped
reproduction matches the paper number) or **[!]** (design target from the
originally-designed protocol; CLI runs but calibrated weights / pre-trained
backbones for the target protocol are not shipped). The table below preserves
that distinction.

| Paper Element | Status | Command | Est. Runtime | Hardware |
|---|---|---|---|---|
| Table `main-results` (12 defenses × 3 attacks; 73.3% valid-cond. ASR / 51.5% all-12) | **[✓]** | `bash scripts/reproduce_table8.sh` | ~2 h | 1×A100 40GB |
| Table `subnetwork` bottom group (WRN-28-10 + last-Linear fine-tune, 3 seeds) | **[✓]** | `python scripts/reproduce_module_table5.py --n-seeds 3 --baseline --output-dir results/table5_rigor_production` | ~35 min (A100) / ~1.5 h (M3 Pro) | GPU |
| Table `subnetwork` top group (ResNet-50 + full-network; 94–97% BD ASR target) | **[!]** | — (protocol not shipped; see `reproduce_module_table5.py` design note) | — | — |
| Table `grad-inversion` (Appendix F; 0.89 SSIM target) | **[!]** | `python scripts/reproduce_module_table4.py --n-samples 20 --max-iter 300 --output-dir results/table4_grad_inversion` | ~6-10 min (A100) / ~15-25 min (M3 Pro) | GPU |
| Table `stat-evasion` (10/12 per-test, 9/12 Fisher target) | **[!]** | `neurinspectre drift-detect --reference clean_features.npy --current adv_features.npy --methods ks_ad_cvm --confidence-level 0.95 --output results/stat_evasion.json` | ~15 min (A100) | GPU |
| Table `rl-detection` (96.8% S_RL target; shipped weights are zero-init placeholder) | **[!]** | `neurinspectre rl-obfuscation analyze --input-file gradient.npy --sensitivity high --output-report results/rl_obf_report.json` | ~1 h (A100) | GPU |
| EDNN embedding attacks (§5.4; 91.7% targeted ASR target) | **[!]** | `neurinspectre adversarial-ednn --attack-type inversion --data embeddings.npy --model sentence-transformers/all-MiniLM-L6-v2 --device auto --output-dir results/ednn` | ~25 min (A100) | GPU |
| Activation steganography (§5.2; 3.2 bits/neuron target) | **[!]** | `neurinspectre activation_steganography encode --model gpt2 --tokenizer gpt2 --prompt "The quick brown fox" --payload-bits 1,0,1,1,0,1,0,0 --target-neurons 42,137,205,318,512,777,1024,1530 --out-prefix results/actsteg_` | ~5 min | CPU/GPU |
| Attention analysis (§5.7; 93.4% detection target) | **[!]** | `neurinspectre attention-security --model gpt2 --prompt "Hello world" --output-png results/attention_security.png --out-json results/attention_security.json` | ~30 min (A100) | GPU |
| Full paper reproduction (all tables + modules) | mixed | `bash scripts/reproduce_all.sh` | ~28 h (A100) | 1×A100 40GB |

**Reading the `[✓]/[!]` distinction:** `[✓]` rows produce numbers that match
the paper (within documented seed variation). `[!]` rows execute end-to-end
and emit valid output JSON, but the numbers will not match the design-target
values in the paper because the calibrated weights or target-protocol
backbones for those rows are not shipped in this submission. This is
disclosed in the paper's `tab:cross-module`, in each module's §5.x caption,
and in `QUICKSTART_CCS_OFFENSIVE.md`.

### Shared utilities

| Task | Command | Est. Runtime | Hardware |
|---|---|---|---|
| Smoke test (quick validation) | `neurinspectre table2-smoke --output-dir results/smoke` | ~5 min | any GPU / CPU |
| Environment inventory | `neurinspectre doctor --as-json` | <10 s | CPU |

## Reproducing Core Evasion Runs (12 defenses)

### Single defense (JPEG compression, CIFAR-10)

```bash
mkdir -p results
neurinspectre analyze \
  --dataset cifar10 \
  --defense jpeg \
  --epsilon 0.03137254901960784 --norm Linf \
  --iterations 100 \
  --num-samples 1000 \
  --no-report --no-progress \
  --output results/core_evasion_cifar10_jpeg.json
```

### Full evaluation pipeline (Table2-style orchestrator)

`neurinspectre table2` normalizes a Table2-style spec into the core evaluation
engine and enforces strict real-data gates if enabled.

```bash
neurinspectre table2 \
  --config table2_config.yaml \
  --output-dir results/table2_run \
  --strict-real-data \
  --strict-dataset-budgets
```

Single-command Table 8 harness (runs `pgd`, then `autoattack --resume`, then `neurinspectre --resume`,
validates triplet coverage, and packages artifacts):

```bash
bash scripts/reproduce_table8.sh
```

Important:

- By default, `neurinspectre table2` expects datasets under `./data/*` and models under `./models/*`.
  For from-scratch asset setup, see `docs/guides/ARTIFACT_EVALUATION_GUIDE.md` (Section 4) and the helpers:
  `scripts/download_cifar10.py`, `scripts/download_ember2018.py`, `scripts/vectorize_ember_safe.py`,
  `scripts/generate_nuscenes_label_map.py`.
- To generate *defense-trained* EMBER checkpoints referenced by `table2_config.yaml` (`md_gradient_reg_ember`,
  `md_distillation_ember`, `md_at_transform_ember`), run:
  `python scripts/train_ember_defense_models.py --train-standard`
  This writes TorchScript artifacts under `models/` that `neurinspectre table2` will automatically pick up
  via checkpoint-tag resolution.
- In strict mode, placeholder/stub models are refused and clean-accuracy validity gates can fail fast.

## Extended Modules (Optional) — offensive paper only

All commands in this section belong to the **offensive-framework paper's**
reproduction path. They write under `results/offensive_modules/` (create
once with `mkdir -p results/offensive_modules`) to keep outputs namespaced
away from the detection paper's `results/detection/`. Several extended
modules live in the legacy (argparse-based) CLI; the Click entrypoint
automatically routes a small allowlist of these commands.

### Gradient Inversion (requires gradient file)

```bash
neurinspectre gradient-inversion recover \
  --gradients path/to/gradients.npy \
  --out-prefix results/offensive_modules/gradinv_
```

### RL Obfuscation Detection (requires gradient file)

```bash
neurinspectre rl-obfuscation analyze \
  --input-file path/to/gradient.npy \
  --sensitivity high \
  --output-report results/offensive_modules/rl_obf_report.json \
  --output-plot results/offensive_modules/rl_obf_plot.png
```

### Attention Security Analysis (requires HF model access)

```bash
neurinspectre attention-security \
  --model gpt2 \
  --prompt "Hello world" \
  --device auto \
  --output-png results/offensive_modules/attention_security.png \
  --out-json results/offensive_modules/attention_security.json \
  --out-html results/offensive_modules/attention_security.html
```

### Subnetwork Hijack — KMeans CLI (requires activations file)

```bash
neurinspectre subnetwork_hijack identify \
  --activations path/to/activations.npy \
  --n_clusters 8 \
  --out-prefix results/offensive_modules/snh_ \
  --interactive
```

> **Note:** This KMeans-activation-clustering CLI is an exploration tool,
> *not* the paper-grade reproduction of Table `subnetwork`. For the
> rigorous, mechanism-level reproduction used to generate the bottom
> group of the paper's `tab:subnetwork` (gradient-based neuron importance
> ranking per Molchanov et al. 2017, BadNets poisoning-rate training,
> faithful Neural Cleanse + Spectral Signatures detectors), run
> `python scripts/reproduce_module_table5.py --n-seeds 3 --baseline
> --output-dir results/table5_rigor_production` instead.

### EDNN Embedding Attacks (requires embeddings file)

```bash
neurinspectre adversarial-ednn \
  --attack-type inversion \
  --data path/to/embeddings.npy \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device auto \
  --output-dir results/offensive_modules/ednn \
  --verbose
```

## Reproduction Script

`scripts/reproduce_all.sh` is a convenience harness that:

- writes a timestamped results directory
- captures environment information (`neurinspectre doctor`)
- runs a small set of core CLI invocations and saves JSON outputs
- validates Table 8 triplet coverage (`pgd`/`autoattack`/`neurinspectre`) when `table2` artifacts are present
- exports a paper-facing Table 10-style aggregate from the Table 8 `summary.json`
- emits:
  - `table8_table2/table8_validation.json`
  - `table8_table2/table10_attack_strength.json`
  - `table8_table2/table10_attack_strength.tex`
  - `table8_table2_artifacts.tgz`
  - `table_command_map.md`
  - `sha256_manifest.txt`

```bash
bash scripts/reproduce_all.sh
```

Useful harness controls:

```bash
# Package an already-complete table2 run (no heavy rerun):
RESULTS_DIR=results/repro_harness_table8_v2 \
SKIP_SMOKE=1 \
SKIP_CORE_EVASION=1 \
TABLE2_REUSE_DIR=results/table8_run_v2 \
bash scripts/reproduce_all.sh

# Paper-grade multi-seed Table 8 (writes seed_<seed>/ subdirs; adds *_std and *_ci95 fields):
TABLE2_NUM_SEEDS=5 RESULTS_DIR=results/repro_harness_table8_multiseed bash scripts/reproduce_table8.sh
```

## Expected Output Format

- `neurinspectre analyze` / `neurinspectre attack` write an `--output` JSON file.
- `neurinspectre evaluate` / `neurinspectre table2` write `summary.json` to the output directory.
- `neurinspectre doctor --as-json` prints a structured JSON inventory (env + deps + GPU + package hash).

## Artifact Hash

To compute a SHA256 for a deterministic source tarball of the current commit:

```bash
bash scripts/make_artifact_tarball.sh
```
