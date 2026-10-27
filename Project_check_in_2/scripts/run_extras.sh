#!/bin/bash
# optional CI2 extras: seed repeats + no-mask ablation (sequential, MPS)
set -e
cd "$(dirname "$0")/.."
for spec in "gru_h64_seed0 --seed 0" "gru_h64_seed1 --seed 1" "gru_h64_seed2 --seed 2" "gru_h64_nomask --no-mask"; do
  set -- $spec; name=$1; shift
  mkdir -p runs/$name
  conda run -n dsan6600 --no-capture-output python -u -m src.train --run-name $name "$@" > runs/$name/train.log 2>&1
  conda run -n dsan6600 --no-capture-output python -m src.evaluate --run-name $name --results-dir runs/$name --no-preds > runs/$name/eval.log 2>&1 || echo "eval failed $name"
  echo "finished $name"
done
echo "ALL EXTRAS DONE"
