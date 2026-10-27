#!/bin/bash
# Optional CI2 extras: 3 more seeds and a run without the mask channel.
# Runs one after another (about 75 min on MPS). Usage: bash scripts/run_extras.sh
set -e
cd "$(dirname "$0")/.."

run_one() {
  name=$1
  shift
  mkdir -p runs/$name
  conda run -n dsan6600 --no-capture-output python -u -m src.train --run-name $name "$@" > runs/$name/train.log 2>&1
  conda run -n dsan6600 --no-capture-output python -m src.evaluate --run-name $name --results-dir runs/$name --no-preds \
    > runs/$name/eval.log 2>&1 || echo "eval failed for $name"
  echo "finished $name"
}

run_one gru_h64_seed0 --seed 0
run_one gru_h64_seed1 --seed 1
run_one gru_h64_seed2 --seed 2
run_one gru_h64_nomask --no-mask
echo "all extras done"
