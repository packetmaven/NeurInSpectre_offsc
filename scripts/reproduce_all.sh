#!/bin/bash
set -euo pipefail

echo "=== NeurInSpectre Reproduction Script ==="
echo "=== WOOT '26 Artifact Harness ==="
echo ""

RESULTS_DIR="${RESULTS_DIR:-results/paper_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"

# CIFAR-10 core evasion defaults (override via env if desired)
MODEL_CIFAR10="${MODEL_CIFAR10:-models/cifar10_resnet20_norm_ts.pt}"
DATA_CIFAR10="${DATA_CIFAR10:-data/cifar10}"
EPS_CIFAR10="${EPS_CIFAR10:-0.03137254901960784}" # 8/255
ITERATIONS="${ITERATIONS:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"

echo "--- Environment ---"
"$PYTHON_BIN" --version | tee "$RESULTS_DIR/python_version.txt"
"$PYTHON_BIN" -c "import torch; print(f'PyTorch: {torch.__version__}')" | tee "$RESULTS_DIR/torch_version.txt"
"$PYTHON_BIN" -c "import torch; print(f'CUDA: {torch.version.cuda or \"N/A\"}')" | tee "$RESULTS_DIR/torch_cuda_version.txt"
"$PYTHON_BIN" -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')" | tee "$RESULTS_DIR/torch_mps.txt"
neurinspectre doctor --as-json > "$RESULTS_DIR/doctor.json"

echo ""
echo "--- Core Evasion (CIFAR-10, 12 defenses) ---"
if [[ ! -f "$MODEL_CIFAR10" ]]; then
  echo "ERROR: Missing model artifact: $MODEL_CIFAR10"
  echo "Hint: put a runnable TorchScript CIFAR-10 model at that path (or set MODEL_CIFAR10=...)."
  exit 2
fi

FAILS=0
DEFENSES=(
  "jpeg"
  "bitdepth"
  "randsmooth"
  "ensemble"
  "feature_squeezing"
  "gradient_regularization"
  "distillation"
  "at_transform"
  "spatial_smoothing"
  "random_pad_crop"
  "thermometer"
  "certified_defense"
)

for defense in "${DEFENSES[@]}"; do
  echo "  Running: $defense"
  out_json="$RESULTS_DIR/core_evasion_${defense}.json"
  out_log="$RESULTS_DIR/core_evasion_${defense}.log"
  if ! neurinspectre attack \
    --model "$MODEL_CIFAR10" \
    --dataset cifar10 --data-path "$DATA_CIFAR10" \
    --defense "$defense" \
    --attack-type neurinspectre \
    --epsilon "$EPS_CIFAR10" --norm Linf \
    --iterations "$ITERATIONS" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --no-report --no-progress --report-format text \
    --output "$out_json" \
    > "$out_log" 2>&1; then
      echo "    FAILED: $defense (see $out_log)"
      FAILS=$((FAILS + 1))
  fi
done

echo ""
echo "--- Optional: Gradient inversion (requires GRADIENTS_FILE) ---"
if [[ -n "${GRADIENTS_FILE:-}" && -f "${GRADIENTS_FILE:-}" ]]; then
  neurinspectre gradient-inversion recover \
    --gradients "$GRADIENTS_FILE" \
    --out-prefix "$RESULTS_DIR/gradinv_" \
    > "$RESULTS_DIR/gradient_inversion.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set GRADIENTS_FILE=/path/to/gradients.npy)"
fi

echo ""
echo "--- Optional: RL obfuscation detection (requires RL_GRADIENT_FILE) ---"
if [[ -n "${RL_GRADIENT_FILE:-}" && -f "${RL_GRADIENT_FILE:-}" ]]; then
  neurinspectre rl-obfuscation analyze \
    --input-file "$RL_GRADIENT_FILE" \
    --sensitivity high \
    --output-report "$RESULTS_DIR/rl_obfuscation_report.json" \
    --output-plot "$RESULTS_DIR/rl_obfuscation_plot.png" \
    > "$RESULTS_DIR/rl_obfuscation.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set RL_GRADIENT_FILE=/path/to/gradient.npy)"
fi

echo ""
echo "=== All results saved to: $RESULTS_DIR ==="
echo "=== SHA256 of JSON outputs ==="
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$RESULTS_DIR"/*.json 2>/dev/null || true
else
  shasum -a 256 "$RESULTS_DIR"/*.json 2>/dev/null || true
fi

if [[ "$FAILS" -ne 0 ]]; then
  echo ""
  echo "ERROR: $FAILS run(s) failed."
  exit 1
fi

echo ""
echo "OK"

