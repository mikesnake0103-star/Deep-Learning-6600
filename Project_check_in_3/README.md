# Check-in 3: Graph WaveNet for METR-LA traffic forecasting

This is our Project Check-in 3 for DSAN 6600 (Neural Nets & Advanced Deep Learning, Georgetown, Fall 2026). We add one advanced model, Graph WaveNet, on top of the Check-in 2 pipeline and compare it with the Check-in 2 GRU on exactly the same test windows. We also train the same network with its graph convolutions removed, to see how much of the improvement really comes from the road graph.

* **Write-up:** [check-in-3.md](check-in-3.md)
* **Results notebook:** [notebooks/03_ci3_results.ipynb](notebooks/03_ci3_results.ipynb)
* **CI2 (GRU baseline):** [../Project_check_in_2/check-in-2.md](../Project_check_in_2/check-in-2.md)
* **CI1 (problem, data, EDA):** [../Project_check_in_1/check-in-1.md](../Project_check_in_1/check-in-1.md)
* **Video:** : add video link
* **Team:** : Mike Kwon, Jason wang

## What's here
```
check-in-3.md               write-up (why Graph WaveNet, model, training, comparison with CI2, ablation, failure analysis, plan)
src/data.py, metrics.py,    copied unchanged from Project_check_in_2/src (same split, normalization, mask,
    baselines.py, config.py     windows, masked metrics, persistence + HA baselines, paths)
src/models/gru.py           the CI2 GRU, unchanged (needed to load CI2 checkpoints)
src/models/gwnet.py         Graph WaveNet, adapted from github.com/nnzhan/Graph-WaveNet (MIT license)
src/train.py                CI2 training CLI + --model gwnet, --graph-mode, --channels, --blocks
src/evaluate.py             CI2 evaluation; now also works for Graph WaveNet runs and records inference time
scripts/run_required.sh     trains and evaluates the two required runs (gwnet_full, gwnet_nograph)
scripts/check_gwnet.py      self-check: output shapes for every graph mode, and that the no-graph model
                            cannot pass information between sensors
notebooks/03_ci3_results.ipynb  every table and figure (reads saved predictions; does not run a model)
figures/                    PNG figures + CSV/markdown tables + ci3_numbers.json (every number quoted in the write-up)
logs/                       tracked copies of the run logs (config, per-epoch metrics, train.log, test metrics)
runs/<run>/                 (git-ignored) the above + best.pt, learning_curve.png, test_preds.npz
```

## Data location (`DATA_DIR`)
No data is copied into this folder. All the code reads from `DATA_DIR`, which points to `../Project_check_in_1/data` by default (relative to `Project_check_in_3/`). That folder needs to contain:
* `processed/metr_la_processed.npz` and `processed/meta.json`, built by CI1's `prepare_processed()`
* `raw/metr-la-dataset/{METR-LA.h5, adj_METR-LA.pkl, graph_sensor_locations.csv}`

[../Project_check_in_1/data/README.md](../Project_check_in_1/data/README.md) explains how to download these files. If your data lives somewhere else, pass `--data-dir /path/to/data` to `src.train` and `src.evaluate`, and change the `data_paths()` call in the notebook.

The notebook also needs the CI2 GRU's saved test predictions, `../Project_check_in_2/runs/gru_h64/test_preds.npz`. That file is git-ignored, so on a fresh clone you first have to train and evaluate the GRU by following the CI2 README (about 20 minutes).

## Reproduce
Run everything from `Project_check_in_3/` inside the `dsan6600` conda environment (Python 3.11, torch 2.14):
```bash
conda activate dsan6600
cd Project_check_in_3

# 0. sanity checks (optional)
PYTHONPATH=. python scripts/check_gwnet.py                          # shapes + no information leak without the graph
python -m src.train --model gwnet --overfit --overfit-steps 1000    # fit 200 train windows (dropout is switched off)
python -m src.train --model gwnet --dropout 0.3 --time-epoch        # time one epoch

# 1. train + evaluate both required runs, one after the other (about 2.5 h on an Apple M3 Pro / MPS)
#    writes runs/gwnet_full/ and runs/gwnet_nograph/, each with test_preds.npz
nohup bash scripts/run_required.sh > runs/required.log 2>&1 &

#    or one run by hand:
python -m src.train --model gwnet --dropout 0.3 --max-epochs 48 --graph-mode fixed+adaptive --run-name gwnet_full
python -m src.evaluate --run-name gwnet_full

# 2. time the CI2 GRU's inference with the same code (writes only into this folder)
python -m src.evaluate --run-name gru_h64 --runs-dir ../Project_check_in_2/runs \
    --results-dir runs/gru_h64_ci2_recheck --no-preds

# 3. re-run the results notebook end to end
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dsan6600 notebooks/03_ci3_results.ipynb
```
`--graph-mode` takes `fixed+adaptive` (the full model), `fixed`, `adaptive` or `none` (the no-graph ablation). The other `src.train` flags are the same as in CI2; run `python -m src.train -h` to see all of them. Each run saves its full config to `runs/<run>/config.json`.
