#!/bin/bash
set -euo pipefail

echo "=== NeurInSpectre Reproduction Script ==="
echo "=== CCS '26 Artifact Harness ==="
echo ""

RESULTS_DIR="${RESULTS_DIR:-results/paper_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

DEVICE="${DEVICE:-auto}"
# Prefer a repo-local venv if present; otherwise rely on PATH.
NS_BIN="${NS_BIN:-}"
if [[ -z "${NS_BIN}" ]]; then
  if [[ -x ".venv-neurinspectre/bin/neurinspectre" ]]; then
    NS_BIN=".venv-neurinspectre/bin/neurinspectre"
  elif [[ -x ".venv/bin/neurinspectre" ]]; then
    NS_BIN=".venv/bin/neurinspectre"
  else
    NS_BIN="neurinspectre"
  fi
fi

# CIFAR-10 core evasion defaults (override via env if desired)
MODEL_CIFAR10="${MODEL_CIFAR10:-}" # optional override; analyze has defaults for CIFAR-10
DATA_CIFAR10="${DATA_CIFAR10:-}"   # optional override; omit to allow torchvision download
EPS_CIFAR10="${EPS_CIFAR10:-0.03137254901960784}" # 8/255
ITERATIONS="${ITERATIONS:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"

echo "--- Environment ---"
"$NS_BIN" doctor --as-json > "$RESULTS_DIR/doctor.json"
"$NS_BIN" doctor > "$RESULTS_DIR/doctor.txt" 2>&1 || true

FAILS=0

echo ""
echo "--- Smoke Test (Click evaluate --smoke-test) ---"
if [[ -f "configs/evaluation_smoke.yaml" ]]; then
  "$NS_BIN" evaluate \
    --config configs/evaluation_smoke.yaml \
    --smoke-test \
    --output-dir "$RESULTS_DIR/evaluate_smoke" \
    --no-report --no-progress --report-format text \
    --device "$DEVICE" \
    > "$RESULTS_DIR/evaluate_smoke.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (missing configs/evaluation_smoke.yaml)"
fi

echo ""
echo "--- Core Evasion (CIFAR-10, 12 defenses) ---"
if [[ -n "${DEFENSES:-}" ]]; then
  # Space-separated list: DEFENSES="jpeg bitdepth ..."
  read -r -a DEFENSE_LIST <<< "${DEFENSES}"
else
  DEFENSE_LIST=(
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
fi

for defense in "${DEFENSE_LIST[@]}"; do
  echo "  Running: $defense"
  out_json="$RESULTS_DIR/core_evasion_${defense}.json"
  out_log="$RESULTS_DIR/core_evasion_${defense}.log"
  cmd=(neurinspectre analyze
    --dataset cifar10
    --defense "$defense"
    --epsilon "$EPS_CIFAR10" --norm Linf
    --iterations "$ITERATIONS"
    --num-samples "$NUM_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --no-report --no-progress
    --output "$out_json"
  )
  cmd[0]="$NS_BIN"
  if [[ -n "${MODEL_CIFAR10}" ]]; then
    cmd+=(--model "$MODEL_CIFAR10")
  fi
  if [[ -n "${DATA_CIFAR10}" ]]; then
    cmd+=(--data-path "$DATA_CIFAR10")
  fi

  if ! "${cmd[@]}" > "$out_log" 2>&1; then
      echo "    FAILED: $defense (see $out_log)"
      FAILS=$((FAILS + 1))
  fi
done

echo ""
echo "--- Optional: Gradient inversion (requires GRADIENTS_FILE) ---"
if [[ -n "${GRADIENTS_FILE:-}" && -f "${GRADIENTS_FILE:-}" ]]; then
  "$NS_BIN" gradient-inversion recover \
    --gradients "$GRADIENTS_FILE" \
    --out-prefix "$RESULTS_DIR/gradinv_" \
    > "$RESULTS_DIR/gradient_inversion.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set GRADIENTS_FILE=/path/to/gradients.npy)"
fi

echo ""
echo "--- Optional: RL obfuscation detection (requires RL_GRADIENT_FILE) ---"
if [[ -n "${RL_GRADIENT_FILE:-}" && -f "${RL_GRADIENT_FILE:-}" ]]; then
  "$NS_BIN" rl-obfuscation analyze \
    --input-file "$RL_GRADIENT_FILE" \
    --sensitivity high \
    --output-report "$RESULTS_DIR/rl_obfuscation_report.json" \
    --output-plot "$RESULTS_DIR/rl_obfuscation_plot.png" \
    > "$RESULTS_DIR/rl_obfuscation.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set RL_GRADIENT_FILE=/path/to/gradient.npy)"
fi

echo ""
echo "--- Optional: Attention security (requires ATTENTION_PROMPT + HF model access) ---"
if [[ -n "${ATTENTION_PROMPT:-}" ]]; then
  ATTENTION_MODEL="${ATTENTION_MODEL:-gpt2}"
  "$NS_BIN" attention-security \
    --model "$ATTENTION_MODEL" \
    --prompt "$ATTENTION_PROMPT" \
    --device "$DEVICE" \
    --output-png "$RESULTS_DIR/attention_security.png" \
    --out-json "$RESULTS_DIR/attention_security.json" \
    --out-html "$RESULTS_DIR/attention_security.html" \
    > "$RESULTS_DIR/attention_security.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set ATTENTION_PROMPT='...'; optional ATTENTION_MODEL=gpt2)"
fi

echo ""
echo "--- Optional: Subnetwork hijack (requires ACTIVATIONS_FILE) ---"
if [[ -n "${ACTIVATIONS_FILE:-}" && -f "${ACTIVATIONS_FILE:-}" ]]; then
  "$NS_BIN" subnetwork_hijack identify \
    --activations "$ACTIVATIONS_FILE" \
    --n_clusters "${SNH_CLUSTERS:-8}" \
    --out-prefix "$RESULTS_DIR/snh_" \
    --interactive \
    > "$RESULTS_DIR/subnetwork_hijack.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set ACTIVATIONS_FILE=/path/to/activations.npy)"
fi

echo ""
echo "--- Optional: EDNN (requires EDNN_EMBEDDINGS_FILE) ---"
if [[ -n "${EDNN_EMBEDDINGS_FILE:-}" && -f "${EDNN_EMBEDDINGS_FILE:-}" ]]; then
  EDNN_MODEL="${EDNN_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
  "$NS_BIN" adversarial-ednn \
    --attack-type inversion \
    --data "$EDNN_EMBEDDINGS_FILE" \
    --model "$EDNN_MODEL" \
    --device "$DEVICE" \
    --output-dir "$RESULTS_DIR/ednn" \
    --verbose \
    > "$RESULTS_DIR/ednn.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set EDNN_EMBEDDINGS_FILE=/path/to/embeddings.npy; optional EDNN_MODEL=...)"
fi

echo ""
echo "--- Optional: Statistical evasion scoring (requires STAT_ATTACK + STAT_REFERENCE) ---"
if [[ -n "${STAT_ATTACK:-}" && -f "${STAT_ATTACK:-}" && -n "${STAT_REFERENCE:-}" && -f "${STAT_REFERENCE:-}" ]]; then
  "$NS_BIN" statistical_evasion score \
    --data "$STAT_ATTACK" \
    --reference "$STAT_REFERENCE" \
    --method ks \
    --alpha 0.05 \
    --out-prefix "$RESULTS_DIR/stat_" \
    > "$RESULTS_DIR/statistical_evasion.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set STAT_ATTACK=/path/to/attack.npy and STAT_REFERENCE=/path/to/reference.npy)"
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

