# WOOT '26 Artifact Evaluation (Quickstart)

This repo includes a single-command harness for **Table 8** reproduction (12 defenses x 3 attacks)
using `neurinspectre table2` + `table2_config.yaml`.

## 0) Environment

```bash
python3 -m venv .venv-neurinspectre
source .venv-neurinspectre/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Optional (exact dependency pinning, if provided):

```bash
python -m pip install -r requirements-frozen.txt
python -m pip install -e .
```

## 1) Datasets (expected under `./data/*`)

### CIFAR-10

```bash
python scripts/download_cifar10.py --root ./data/cifar10 --split both
```

### EMBER 2018 (raw -> vectorized memmaps)

```bash
python scripts/download_ember2018.py --root ./data/ember
python scripts/vectorize_ember_safe.py
```

This produces the vectorized memmaps that `table2` expects:
`./data/ember/ember_2018/{X_train.dat,y_train.dat,X_test.dat,y_test.dat}`.

### nuScenes (manual download required)

nuScenes cannot be fetched automatically from this repo (license / account gated).

1) Download **v1.0-mini** from the official nuScenes site and extract so that:
`./data/nuscenes/v1.0-mini/scene.json` exists.

2) Generate the single-label proxy task label map used by `table2`:

```bash
python scripts/generate_nuscenes_label_map.py \
  --dataroot ./data/nuscenes \
  --version v1.0-mini \
  --output ./data/nuscenes/label_map.json
```

## 2) Models (expected under `./models/*`)

If the Table 8 TorchScript artifacts are already present under `./models/`, you can skip this step.

To (re)train + export the EMBER TorchScript checkpoints referenced by `checkpoint_tag` in `table2_config.yaml`:

```bash
python scripts/train_ember_defense_models.py --train-standard
```

## 3) Reproduce Table 8 (Table2-style orchestrator)

Single command (runs `table2` in 3 resume-able stages and packages outputs):

```bash
bash scripts/reproduce_table8.sh
```

Optional (choose an explicit output directory and device):

```bash
RESULTS_DIR=results/ae_table8 DEVICE=auto bash scripts/reproduce_table8.sh
```

Optional (paper-grade multi-seed replication; writes `seed_<seed>/` subdirs and adds `*_std` + `*_ci95` fields):

```bash
TABLE2_NUM_SEEDS=5 RESULTS_DIR=results/ae_table8_multiseed bash scripts/reproduce_table8.sh
```

## 4) Expected outputs

Under `$RESULTS_DIR` (default: `results/paper_<timestamp>/`):

- `table8_table2/summary.json`
- `table8_table2/resolved_table2_config.yaml`
- `table8_table2/table8_validation.json`
- `table8_table2/table10_attack_strength.json` (attack-strength aggregates derived from `summary.json`)
- `table8_table2/table10_attack_strength.tex` (copy/paste LaTeX table for the paper-side replacement)
- `table8_table2_artifacts.tgz` (tarball of `*.json/*.log/*.yaml` from `table8_table2/`)
- `table_command_map.md`
- `sha256_manifest.txt`
