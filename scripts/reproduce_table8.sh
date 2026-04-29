#!/bin/bash
set -euo pipefail

# Thin wrapper: run the Table 8 / Table2-style reproduction only.
# This delegates to `scripts/reproduce_all.sh` while skipping the non-Table8 steps by default.
#
# Supported env vars (same as reproduce_all.sh):
# - RESULTS_DIR, DEVICE, NS_BIN
# - TABLE2_REUSE_DIR (package an existing table2 run instead of rerunning)
# - EXPECTED_TABLE2_DEFENSES (default: 12)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Default to Table8-only scope, but allow overrides.
export SKIP_SMOKE="${SKIP_SMOKE:-1}"
export SKIP_CORE_EVASION="${SKIP_CORE_EVASION:-1}"

# This wrapper is for Table 8; do not skip the table2 stage.
export SKIP_TABLE2=0

bash scripts/reproduce_all.sh

