# Reproduction Guide for WOOT '26 Paper

This repository is designed to run on real datasets and real model artifacts.
It intentionally does not hard-code paper baseline numbers in-code; validation
against expected values (if desired) must be supplied via external files.

## Hardware Requirements

**Primary evaluation hardware:**

- GPU: NVIDIA A100 40GB (CUDA) or Apple Silicon M3/M4 Pro (MPS)
- RAM: >=32GB
- Storage: >=50GB (models + datasets)

**Minimum (smoke tests only):**

- Any GPU with >=8GB VRAM, or CPU-only (can be ~10x slower)

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
git clone https://github.com/packetmaven/NeurInSpectre.git
cd NeurInSpectre
python -m pip install -e ".[dev]"
neurinspectre doctor
```

## Smoke Test (5 minutes)

For a fast end-to-end check (auto-discovers what is runnable locally):

```bash
neurinspectre table2-smoke --output-dir results/smoke
```

Outputs include:

- `results/smoke/run_metadata.json` (environment + git + config hash)
- `results/smoke/summary.json` (evaluation summary)

## Reproducing Core Evasion Runs (12 defenses)

### Single defense (JPEG compression, CIFAR-10)

```bash
mkdir -p results
neurinspectre attack \
  --model models/cifar10_resnet20_norm_ts.pt \
  --dataset cifar10 --data-path data/cifar10 \
  --defense jpeg \
  --epsilon 0.03137254901960784 --norm Linf \
  --attack-type neurinspectre \
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

Important:

- You must set dataset roots and model paths that exist on your machine (see `table2_config.yaml`).
- In strict mode, placeholder/stub models are refused and clean-accuracy validity gates can fail fast.

## Extended Modules (Optional)

Several extended modules live in the legacy (argparse-based) CLI. The Click
entrypoint will automatically route a small allowlist of these commands.

### Gradient Inversion (requires gradient file)

```bash
neurinspectre gradient-inversion recover \
  --gradients path/to/gradients.npy \
  --out-prefix results/gradinv_
```

### RL Obfuscation Detection (requires gradient file)

```bash
neurinspectre rl-obfuscation analyze \
  --input-file path/to/gradient.npy \
  --sensitivity high \
  --output-report results/rl_obf_report.json \
  --output-plot results/rl_obf_plot.png
```

### Attention Security Analysis (requires HF model access)

```bash
neurinspectre attention-security \
  --model gpt2 \
  --prompt "Hello world" \
  --device auto \
  --output-png results/attention_security.png \
  --out-json results/attention_security.json \
  --out-html results/attention_security.html
```

## Reproduction Script

`scripts/reproduce_all.sh` is a convenience harness that:

- writes a timestamped results directory
- captures environment information (`neurinspectre doctor`)
- runs a small set of core CLI invocations and saves JSON outputs

```bash
bash scripts/reproduce_all.sh
```

## Expected Output Format

- `neurinspectre attack` writes an `--output` JSON file.
- `neurinspectre evaluate` / `neurinspectre table2` write `summary.json` to the output directory.
- `neurinspectre doctor --as-json` prints a structured JSON inventory (env + deps + GPU + package hash).

## Artifact Hash

SHA256: TBD (computed at camera-ready)
