# Check-in 2: A shared-weight GRU baseline for METR-LA traffic forecasting

This is our Project Check-in 2 for DSAN 6600 (Neural Nets & Advanced Deep Learning, Georgetown, Fall 2026). Here we train and evaluate our first neural model, a GRU that looks at the last hour of traffic speed at each of the 207 Los Angeles loop detectors and forecasts the next 5 to 60 minutes. Everything runs on the same pipeline we built in Check-in 1: the same chronological 70/10/20 split, normalization, missing-value mask, windows and masked metrics.

* **Write-up:** [check-in-2.md](check-in-2.md)
* **Results notebook:** [notebooks/02_ci2_results.ipynb](notebooks/02_ci2_results.ipynb)
* **CI1 (problem, data, EDA):** [../Project_check_in_1/check-in-1.md](../Project_check_in_1/check-in-1.md)
* **Video:** 
* **Team:** Jason Wang/ Junyoung Kwon

## What's here
```
check-in-2.md               write-up (model, training setup, results, qualitative, failure analysis, CI3 plan)
src/data.py, metrics.py,    copied unchanged from Project_check_in_1/src (split, normalization, mask,
    baselines.py            windowing, torch Dataset, masked MAE/RMSE/MAPE, persistence + HA probes)
src/config.py               DATA_DIR, device selection (cuda > mps > cpu), seeding
src/models/gru.py           SharedGRU: per-sensor GRU encoder (shared weights) + direct 12-step MLP head
src/train.py                training CLI (overfit check, epoch timing, full run with early stopping)
src/evaluate.py             test-set inference + probes on the same windows -> results table, test_preds.npz
notebooks/02_ci2_results.ipynb  all result figures and failure analysis (reads test_preds.npz; no inference)
figures/                    PNG figures + CSV/markdown tables + ci2_numbers.json (every number quoted in the write-up)
logs/                       tracked copies of run logs (config, per-epoch metrics, train.log, test metrics) + overfit checks
runs/<run>/                 (git-ignored) the above + best.pt, learning_curve.png, test_preds.npz
scripts/run_extras.sh       optional extras: seeds 0/1/2 + no-mask ablation, trained and evaluated sequentially
```

## Data location (`DATA_DIR`)
We didn't copy the data into this folder. Instead, all the code reads from `DATA_DIR`, which points to `../Project_check_in_1/data` by default (relative to `Project_check_in_2/`). That folder needs to contain:
* `processed/metr_la_processed.npz` and `processed/meta.json`, built by CI1's `prepare_processed()`
* `raw/metr-la-dataset/{METR-LA.h5, adj_METR-LA.pkl, graph_sensor_locations.csv}`

[../Project_check_in_1/data/README.md](../Project_check_in_1/data/README.md) explains how to download these files. If your data lives somewhere else, pass `--data-dir /path/to/data` to `src.train` and `src.evaluate`, and change the `data_paths()` call in the notebook.

## Reproduce
Run everything from `Project_check_in_2/` inside the `dsan6600` conda environment (Python 3.11, torch 2.14):
```bash
conda activate dsan6600
cd Project_check_in_2

# 0. sanity checks (optional)
python -m src.train --overfit                   # memorize 200 train windows for 100 steps
python -m src.train --time-epoch                # time one epoch, project the full-run cost

# 1. train (≈15–20 min on an Apple M3 Pro / MPS); writes runs/gru_h64/
mkdir -p runs/gru_h64
nohup python -u -m src.train --run-name gru_h64 > runs/gru_h64/train.log 2>&1 &

# 2. evaluate best checkpoint on the test split -> runs/gru_h64/test_preds.npz, figures/results_table.{csv,md}
python -m src.evaluate --run-name gru_h64

# 3. (optional, ≈75 min) seeds 0/1/2 + no-mask ablation -> runs/gru_h64_{seed0,seed1,seed2,nomask}/
bash scripts/run_extras.sh

# 4. re-run the results notebook end to end (§5 picks up the extra runs if present)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dsan6600 notebooks/02_ci2_results.ipynb
```
Some handy `src.train` flags are `--hidden`, `--layers`, `--dropout`, `--no-mask` (leave out the mask channel), `--seed`, `--max-epochs` and `--device {cuda,mps,cpu}`. Run `python -m src.train -h` to see all of them. Each run saves its full config to `runs/<run>/config.json`.
