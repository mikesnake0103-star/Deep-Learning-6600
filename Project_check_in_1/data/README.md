# Data: METR-LA

## Source
METR-LA contains traffic speed readings from **207 loop detectors on Los Angeles County highways**, recorded **2012-03-01 → 2012-06-27** at **5-minute** resolution. The raw data comes from the Los Angeles Metropolitan Transportation Authority (LA Metro) and was described in Jagadish et al., *"Big Data and Its Technical Challenges"*, Communications of the ACM 57(7), 2014. The benchmark version used here (sensor selection, 5-min aggregation, adjacency matrix) was released with **DCRNN** (Li, Yu, Shahabi & Liu, *"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"*, ICLR 2018): <https://github.com/liyaguang/DCRNN>.

## Files (all under `data/raw/metr-la-dataset/`, git-ignored)

| File | Size | Contents | How to obtain |
|---|---|---|---|
| `METR-LA.h5` | 57 MB | pandas/PyTables HDF5, group `df`: `block0_values` float64 (34,272 × 207) speeds in mph; `axis0` sensor IDs (bytes); `axis1` int64 ns timestamps | **Used here:** Kaggle mirror [annnnguyen/metr-la-dataset](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset/data) (ships it together with the adjacency pickle). Original: DCRNN README → Google Drive / Baidu Yun links. |
| `adj_METR-LA.pkl` | 0.7 MB | Python-2 pickle: `[sensor_ids (207), sensor_id_to_ind (dict), adj_mx (207×207 float32)]` — identical in structure to DCRNN's `data/sensor_graph/adj_mx.pkl` | Same Kaggle mirror, or DCRNN repo `data/sensor_graph/adj_mx.pkl` (rename). |
| `graph_sensor_locations.csv` | 6 KB | `index, sensor_id, latitude, longitude` for the 207 sensors | `python scripts/download_data.py` (fetches from the DCRNN repo) |

Data encoding notes (verified in `notebooks/01_eda.ipynb`):
* **Missing readings are stored as `0.0`** (8.11% of entries). There are no NaNs.
* The maximum value is exactly **70.0 mph** (a cap).
* The adjacency sensor order matches the `.h5` column order exactly. `src/data.load_adj` raises an error if it doesn't.
* `pd.read_hdf` fails on this legacy file with pandas ≥ 3. `src/data.load_metr_la` falls back to reading it with `h5py`.

## Setup
```bash
# 1a. browser: download https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset/data
#     and unzip so that METR-LA.h5 and adj_METR-LA.pkl sit in data/raw/metr-la-dataset/
# 1b. or Kaggle CLI (needs ~/.kaggle/kaggle.json API token):
kaggle datasets download -d annnnguyen/metr-la-dataset -p data/raw/metr-la-dataset --unzip
# 2. sensor locations + sanity check
python scripts/download_data.py        # fetches graph_sensor_locations.csv, checks the other two files
```
`data/processed/` is created on the first `prepare_processed()` / `load_processed()` call (normalized arrays + `meta.json`). It is also git-ignored.

## License / usage
* **DCRNN code repository:** MIT License, Copyright (c) 2017 Yaguang Li (from the repo's `LICENSE` file). That license covers the repository's *code*. The DCRNN README doesn't state a separate license for the data files.
* **Underlying data:** collected by LA Metro and distributed through the DCRNN authors. It is widely used as an academic research benchmark (DCRNN, Graph WaveNet, STGCN, GMAN, …). **We couldn't find an explicit data license.** We therefore treat it as research/educational-use data: we don't redistribute the raw files in this repo (they are git-ignored), and we cite Jagadish et al. 2014 and Li et al. 2018.
* The Kaggle mirror we used ([annnnguyen/metr-la-dataset](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset/data)) is a third-party re-upload. Its page lists an **MIT License**, but the attached text names "Mark Otto (2013)" and "Andrew Fong (2017)" as copyright holders. Neither is connected to LA Metro, Jagadish et al., or the DCRNN authors, so this looks like a generic MIT template chosen by the uploader. It is not a grant from the data's owners and is **not authoritative** for the underlying data. Our usage stance above doesn't change.
