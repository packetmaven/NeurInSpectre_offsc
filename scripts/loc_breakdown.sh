#!/usr/bin/env bash
set -euo pipefail

# LOC breakdown helper for artifact evaluation.
#
# Writes a machine-readable JSON report so reviewers can verify the
# "lines of code" claims without trusting a hand-count.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_PATH="${1:-$ROOT_DIR/results/loc_breakdown.json}"

if ! command -v cloc >/dev/null 2>&1; then
  echo "error: cloc is not installed."
  echo "install:"
  echo "  macOS:  brew install cloc"
  echo "  Ubuntu: sudo apt-get update && sudo apt-get install -y cloc"
  exit 2
fi

mkdir -p "$(dirname "$OUT_PATH")"

cloc \
  --json \
  --quiet \
  --exclude-dir=".git,.venv-neurinspectre,htmlcov,__pycache__,node_modules,_output,_cli_smokes,_smoke,results,cache" \
  "$ROOT_DIR" \
  >"$OUT_PATH"

echo "$OUT_PATH"

#!/usr/bin/env bash
set -euo pipefail

# LOC decomposition for artifact evaluation / paper hygiene.
#
# Writes: results/loc_breakdown.json
#
# Dependencies:
#   - cloc (https://github.com/AlDanial/cloc)
#
# Install:
#   - macOS: brew install cloc
#   - Ubuntu: sudo apt-get update && sudo apt-get install -y cloc

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${OUT_DIR:-"$repo_root/results"}"
out_file="${OUT_FILE:-"$out_dir/loc_breakdown.json"}"

mkdir -p "$out_dir"

if ! command -v cloc >/dev/null 2>&1; then
  echo "ERROR: cloc is not installed (required for LOC breakdown)." >&2
  echo "Install it with: brew install cloc   (macOS)  OR  sudo apt-get install -y cloc (Ubuntu)" >&2
  exit 2
fi

# Prefer explicit roots to avoid counting datasets/results/artifacts.
cloc \
  --quiet \
  --json \
  --out "$out_file" \
  "$repo_root/neurinspectre" \
  "$repo_root/tests" \
  "$repo_root/scripts" \
  "$repo_root/docs" \
  "$repo_root/README.md" \
  "$repo_root/REPRODUCE.md" \
  "$repo_root/pyproject.toml" \
  "$repo_root/table2_config.yaml"

echo "$out_file"

