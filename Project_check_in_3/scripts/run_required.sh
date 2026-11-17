#!/bin/bash
# The two required CI3 runs: Graph WaveNet with the graph and without it.
# Same seed (42) and settings; only --graph-mode differs. About 2.5 h on MPS.
# Usage: bash scripts/run_required.sh
set -e
cd "$(dirname "$0")/.."

run_one() {
  name=$1
  shift
  mkdir -p runs/$name
  conda run -n dsan6600 --no-capture-output python -u -m src.train --model gwnet --dropout 0.3 --max-epochs 48 \
    --run-name $name "$@" > runs/$name/train.log 2>&1
  conda run -n dsan6600 --no-capture-output python -m src.evaluate --run-name $name \
    > runs/$name/eval.log 2>&1 || echo "eval failed for $name"
  echo "finished $name"
}

run_one gwnet_full --graph-mode fixed+adaptive
run_one gwnet_nograph --graph-mode none
echo "all required runs done"
