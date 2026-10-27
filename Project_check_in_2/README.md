# CI2 — Shared-weight GRU baseline for METR-LA traffic forecasting

DSAN 6600 (Neural Nets & Advanced Deep Learning, Georgetown, Fall 2026), Project Check-in 2. We train and evaluate the first neural baseline on the CI1 pipeline: the same chronological 70/10/20 split, global z-score, missing-value mask, windows and masked metrics. The model forecasts speed at all 207 Los Angeles loop detectors 5–60 min ahead from the previous hour.

* **Write-up:** [check-in-2.md](check-in-2.md)
* **Results notebook:** [notebooks/02_ci2_results.ipynb](notebooks/02_ci2_results.ipynb)
* **CI1 (problem, data, EDA):** [../Project_check_in_1/check-in-1.md](../Project_check_in_1/check-in-1.md)
* **Video:** TODO(Mike): add video link
* **Team:** TODO(Mike): names

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
No data is copied into this folder. All code reads from **`DATA_DIR`**, which defaults to **`../Project_check_in_1/data`** (resolved relative to `Project_check_in_2/`). It must contain:
* `processed/metr_la_processed.npz` and `processed/meta.json`, built by CI1's `prepare_processed()`
* `raw/metr-la-dataset/{METR-LA.h5, adj_METR-LA.pkl, graph_sensor_locations.csv}`

See [../Project_check_in_1/data/README.md](../Project_check_in_1/data/README.md) for how to download these. To point somewhere else, pass `--data-dir /path/to/data` to `src.train` / `src.evaluate` (in the notebook, change the `data_paths()` call).

## Reproduce
All commands run from `Project_check_in_2/` in the `dsan6600` conda env (Python 3.11, torch 2.14):
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
Useful `src.train` flags: `--hidden`, `--layers`, `--dropout`, `--no-mask` (drop the mask channel), `--seed`, `--max-epochs`, `--device {cuda,mps,cpu}`. See `python -m src.train -h` for the full list. The resolved config is saved to `runs/<run>/config.json`.
