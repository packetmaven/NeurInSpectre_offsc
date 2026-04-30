# Artifact Evaluation Guide (ACM CCS '26)

This guide is for reviewers (and for our own sanity) to run NeurInSpectre end-to-end
in a reproducible, scientifically defensible way.

Key policy notes:
- This repo intentionally does not ship paper baseline numbers. Any "expected" ASR/RA matrices
  must be supplied via external files.
- The evaluated defenses are representative defense mechanisms (not vendor production systems).
  If the paper uses the term "deployed", it should be interpreted as "deployed in research
  pipelines / representative mechanisms", unless an explicit vendor API/model is evaluated.

## 1) Environment (exact stack capture)

At minimum, record:
- OS + kernel, GPU model, RAM
- Python version
- PyTorch version (+ CUDA version if applicable)

Suggested commands:
```bash
python -V
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('mps', hasattr(torch.backends,'mps') and torch.backends.mps.is_available())"
python -m pip freeze > artifact_pip_freeze.txt
```

## 2) Install

```bash
python -m venv .venv-neurinspectre
source .venv-neurinspectre/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Optional dependencies (only needed if you run those datasets/modules):
```bash
# nuScenes
python -m pip install nuscenes-devkit

# EMBER vectorization/training helpers (not needed for evaluation if vectorized memmaps exist)
python -m pip install lief==0.12.3
python -m pip install git+https://github.com/elastic/ember.git

# AutoAttack (optional; only needed if you run AutoAttack parity checks)
python -m pip install git+https://github.com/fra31/auto-attack
```

## 3) Fast functional check (real-data smoke)

This is designed to be an AE-friendly retry loop: it discovers what is runnable
on disk, generates a tiny Table2-style config, and runs the normal `table2`
pipeline with strict real-data checks enabled.

```bash
neurinspectre table2-smoke --device auto --attacks pgd --no-progress --summary-only
```

Artifacts written:
- Generated config: `results/.../table2_smoke_real_config.yaml`
- Resolved strict config: `results/.../resolved_table2_config.yaml`
- Summary JSON: `results/.../summary.json`

Notes:
- If `nuscenes-devkit` is missing, the smoke runner will skip nuScenes and print install hints.
- EMBER evaluation loads vectorized memmaps and does not require the `ember` Python package.
- If the smoke runner cannot find any runnable (dataset, model) pair, it will fail fast.

## 4) Dataset + model prerequisites (strict real-data mode)

NeurInSpectre supports multiple datasets. Strict real-data mode intentionally refuses
missing assets and placeholder models.

### CIFAR-10
- Root: `./data/cifar10` (can be downloaded automatically by torchvision)
  - From scratch:
    ```bash
    python scripts/download_cifar10.py
    ```
- Model: a small TorchScript CIFAR-10 model is included for smoke runs.
  - Optional: regenerate the TorchScript artifact + pinned meta:
    ```bash
    python scripts/export_cifar10_torchscript.py
    ```

### ImageNet-100 (optional)
- Root should contain `train/` and/or `val/` folders in ImageFolder layout.
- You must provide a trained ImageNet-100 model artifact; placeholders/stubs are refused in strict mode.
  - This artifact includes `models/imagenet100_resnet50.pt` marked `is_stub: true` in its `.meta.json` as a placeholder.

### EMBER (optional)
- Root: `./data/ember`
- Vectorized features expected at: `./data/ember/ember_2018/{X_train.dat,y_train.dat,X_test.dat,y_test.dat}`
- From scratch (raw -> vectorized):
  ```bash
  python scripts/download_ember2018.py
  python scripts/vectorize_ember_safe.py
  ```
- Optional: retrain + re-export the TorchScript model artifact:
  ```bash
  python scripts/train_ember_real.py
  python scripts/export_ember_torchscript.py
  ```
- Optional: train/export defense-specific EMBER checkpoints referenced by `table2_config.yaml` checkpoint tags:
  ```bash
  python scripts/train_ember_defense_models.py --train-standard
  ```
  This writes TorchScript artifacts under `./models/` such as:
  - `md_gradient_reg_ember_ts.pt`
  - `md_distillation_ember_ts.pt`
  - `md_at_transform_ember_ts.pt`
- If you need macOS-safe vectorization only, use:
  ```bash
  python scripts/vectorize_ember_safe.py
  ```

### nuScenes (optional)
- Root: `./data/nuscenes` (supports `v1.0-mini`)
  - Expected layout includes `./data/nuscenes/v1.0-mini/` (metadata) and `./data/nuscenes/samples/` (images).
- Label map: `./data/nuscenes/label_map.json`
  - Generate via:
    ```bash
    python scripts/generate_nuscenes_label_map.py --dataroot data/nuscenes --version v1.0-mini --output data/nuscenes/label_map.json
    ```
- Model artifact should include metadata: `<model>.meta.json`
  - Strict mode can enforce a SHA256 match between `label_map.json` and model metadata.

## 5) Running a Table2-style pipeline (spec config)

Example spec-style config (shipped): `table2_config.yaml`

```bash
neurinspectre table2 -c table2_config.yaml -o results/table2 --device auto
```

`table2` normalizes the spec into a runnable evaluation config and then calls the
standard evaluation runner.

## 6) Baseline validation (external-only)

This repo does not ship paper baseline numbers. If you want strict validation against
an expected ASR matrix, supply it externally:

```bash
neurinspectre compare --mode baseline results/table2/summary.json --expected-asr-path /abs/path/to/expected_asr.yaml
```

## 7) Common failure modes (intended guardrails)

- Validity gate failure (clean accuracy too low):
  - This is an intentional fail-fast to prevent meaningless ASR reporting.
  - Fix by providing a trained model and verifying dataset/label-map alignment.

- nuScenes label-map hash gate failure:
  - Indicates a mismatch between model-side metadata and the evaluation-time label map.
  - Fix by regenerating label_map.json and retraining/re-exporting the model artifact (or updating metadata).

