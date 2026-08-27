#!/usr/bin/env bash
# Run GCN baseline training and/or evaluation.
#
# Usage:
#   ./scripts/run_gcn.sh                 # train + eval (default)
#   ./scripts/run_gcn.sh --train-only    # train only
#   ./scripts/run_gcn.sh --eval-only     # eval only (requires checkpoint)
#   ./scripts/run_gcn.sh --all --epochs 50 --cpu
#
# Any extra flags are forwarded to the Python scripts.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="all"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-only)
      MODE="train"
      shift
      ;;
    --eval-only)
      MODE="eval"
      shift
      ;;
    --all)
      MODE="all"
      shift
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

run_train() {
  echo "==> Training GCN baseline"
  uv run python scripts/train_gnn_baseline.py "${PASSTHROUGH[@]}"
}

run_eval() {
  echo "==> Evaluating GCN baseline"
  uv run python scripts/eval_gnn_baseline.py "${PASSTHROUGH[@]}"
}

case "$MODE" in
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  all)
    run_train
    run_eval
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac

echo "==> Done ($MODE)"
