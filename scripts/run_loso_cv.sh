#!/usr/bin/env bash
# Run leave-one-site-out cross-validation (LOSO-CV) for GCN.
#
# Usage:
#   ./scripts/run_loso_cv.sh
#   ./scripts/run_loso_cv.sh --epochs 100 --cpu
#   ./scripts/run_loso_cv.sh --sites CMU,YALE,NYU   # debug subset
#
# Full run: 20 folds (one per acquisition site), ~30-40 min on Apple Silicon MPS.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  PASSTHROUGH+=("$1")
  shift
done

echo "==> Running LOSO-CV (GCN)"
if ((${#PASSTHROUGH[@]} > 0)); then
  uv run python -m neuroasd.loso_cv "${PASSTHROUGH[@]}"
else
  uv run python -m neuroasd.loso_cv
fi

echo "==> Done"
