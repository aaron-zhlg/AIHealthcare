#!/usr/bin/env bash
# Attribute a trained GCN and write figures into that experiment's folder.
#
# Usage:
#   ./scripts/run_explain.sh
#   ./scripts/run_explain.sh --checkpoint outputs/gcn_baseline/final_model.pt \
#       --output-dir experiments/baseline_gcn_v1/figures
#   ./scripts/run_explain.sh --checkpoint outputs/loso_cv_gcn_v1/folds/NYU/final_model.pt \
#       --output-dir experiments/loso_cv_gcn_v1/figures/NYU --split all
#
# Any extra flags are forwarded to `python -m aihealthcare.explain`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  PASSTHROUGH+=("$1")
  shift
done

echo "==> Explaining GCN"
if ((${#PASSTHROUGH[@]} > 0)); then
  uv run python -m aihealthcare.explain "${PASSTHROUGH[@]}"
else
  uv run python -m aihealthcare.explain \
    --checkpoint outputs/gcn_baseline/final_model.pt \
    --output-dir experiments/baseline_gcn_v1/figures
fi

echo "==> Done"
