# METR-LA Spatio-temporal Traffic Speed Forecasting

DSAN 6600 (Neural Nets & Advanced Deep Learning, Georgetown, Fall 2026). We forecast highway speed at **207 Los Angeles loop detectors** 15–60 minutes ahead from the previous hour of readings. The plan starts from a GRU seq2seq baseline and moves to a graph-based spatio-temporal model (DCRNN / Graph WaveNet).

* **Check-in 1 write-up:** [check-in-1.md](check-in-1.md)
* **EDA notebook:** [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)
* **Dataset access notes:** [data/README.md](data/README.md)
* **Team:**: Junyoung (Mike) Kwon, Jason Wang
* **Video**: https://www.loom.com/share/5a2acec4c3a54ec69a6aef8a1a8459d1

## Repo layout
```
check-in-1.md            write-up (problem, data, EDA, evaluation plan, planned models)
data/README.md           source, size, license/usage, download instructions
data/raw/                raw files (git-ignored)
data/processed/          cached normalized arrays + meta.json (git-ignored, auto-built)
notebooks/01_eda.ipynb   data audit / EDA + non-neural probes
src/data.py              loader, masks, time features, split, z-score, windowing, cache, torch Dataset
src/metrics.py           masked MAE / RMSE / MAPE (numpy or torch)
src/baselines.py         persistence + historical-average probes
scripts/download_data.py fetches sensor locations, checks raw files
figures/                 all plots (PNG) + summary CSVs
```

## Setup
```bash
conda create -n dsan6600 python=3.11 -y
conda activate dsan6600
pip install -r requirements.txt          # add `pip install torch` for training (optional for EDA)
python -m ipykernel install --user --name dsan6600

# data: put METR-LA.h5 + adj_METR-LA.pkl in data/raw/metr-la-dataset/ (see data/README.md), then
python scripts/download_data.py

# re-run the EDA end to end
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dsan6600 notebooks/01_eda.ipynb
```

## Using the data layer for training
```python
from torch.utils.data import DataLoader
from src.data import load_processed, MetrLADataset
from src.metrics import masked_mae

arrays, meta = load_processed()                        # builds data/processed/ on first call
train = MetrLADataset("train", arrays=arrays, meta=meta, include_mask_channel=False)
loader = DataLoader(train, batch_size=64, shuffle=True)
x, y, y_mask = next(iter(loader))                      # x (64,12,207,3) z-scored speed+tod+dow; y (64,12,207) mph
# pred_mph = model(x) * meta["std"] + meta["mean"];  loss = masked_mae(pred_mph, y)
```
The splits are chronological 70/10/20 (split first, then window), with a global z-score from nonzero train values. Missing inputs are filled with 0 after z-scoring (= train mean). Targets are in raw mph, and zeros are masked in the loss and metrics.
