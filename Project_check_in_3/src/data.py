"""METR-LA data layer: loading, missingness, splitting, normalization, windowing.

Conventions (shared by the EDA notebook, CI2 and CI3):
* Missing readings are encoded as 0.0 in the raw file -> ``mask = raw != 0``.
* Split by timestamp first (70/10/20, chronological), then window *within* each
  split, so no window straddles a split boundary.
* Z-score with a single global mean/std over nonzero TRAIN values (DCRNN
  convention). Missing positions are set to 0.0 *after* normalization, i.e. the
  train mean, so outages don't look like -2.7 sigma "congestion".
* Targets are returned in raw mph together with a mask; metrics/losses ignore
  masked positions (see ``src/metrics.py``).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "metr-la-dataset"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_metr_la(path: str | Path = RAW_DIR / "METR-LA.h5") -> pd.DataFrame:
    """Load METR-LA speeds as a (T, N) DataFrame (DatetimeIndex x sensor-id str).

    ``pd.read_hdf`` fails on pandas >= 3 for this legacy file, so fall back to
    reading the PyTables "fixed" layout directly with h5py.
    """
    path = Path(path)
    try:
        df = pd.read_hdf(path)
        df.columns = [c.decode() if isinstance(c, bytes) else str(c) for c in df.columns]
        return df
    except Exception:
        import h5py

        with h5py.File(path, "r") as f:
            values = f["df/block0_values"][:]
            index = pd.to_datetime(f["df/axis1"][:])
            columns = [c.decode() for c in f["df/axis0"][:]]
        return pd.DataFrame(values, index=index, columns=columns)


def load_adj(path: str | Path = RAW_DIR / "adj_METR-LA.pkl", sensor_ids=None):
    """Load the DCRNN adjacency pickle -> (sensor_ids, sensor_id_to_ind, adj_mx).

    The file is a Python-2 pickle (a 3-element list). If ``sensor_ids`` is given
    (e.g. the h5 column order), raise if the orders differ.
    """
    with open(path, "rb") as f:
        ids, id2ind, adj = pickle.load(f, encoding="latin1")
    ids = [i.decode() if isinstance(i, bytes) else str(i) for i in ids]
    adj = np.asarray(adj, dtype=np.float32)
    assert adj.shape == (len(ids), len(ids)), adj.shape
    if sensor_ids is not None and list(sensor_ids) != ids:
        diff = sum(a != b for a, b in zip(sensor_ids, ids))
        raise ValueError(f"adjacency sensor order differs from data columns ({diff} mismatches)")
    return ids, id2ind, adj


def load_locations(path: str | Path = RAW_DIR / "graph_sensor_locations.csv"):
    """Sensor lat/lon table (index by sensor_id str), or None if not downloaded."""
    path = Path(path)
    if not path.exists():
        return None
    loc = pd.read_csv(path, dtype={"sensor_id": str})
    return loc.set_index("sensor_id")[["latitude", "longitude"]]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def missing_mask(values) -> np.ndarray:
    """True where a reading is observed (raw value != 0)."""
    return np.asarray(values) != 0


def time_features(index: pd.DatetimeIndex, dow_onehot: bool = False) -> np.ndarray:
    """(T, F) float32: time-of-day in [0, 1) and day-of-week (0=Mon..6, or one-hot)."""
    tod = ((index.hour * 60 + index.minute) / 1440.0).to_numpy()
    dow = index.dayofweek.to_numpy()
    if dow_onehot:
        return np.column_stack([tod, np.eye(7)[dow]]).astype(np.float32)
    return np.column_stack([tod, dow]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Split + normalization
# --------------------------------------------------------------------------- #
def chrono_split(n_steps: int, ratios=(0.7, 0.1, 0.2)) -> dict[str, tuple[int, int]]:
    """Contiguous [start, end) timestamp ranges for train/val/test."""
    assert abs(sum(ratios) - 1) < 1e-9
    n_train = int(round(n_steps * ratios[0]))
    n_val = int(round(n_steps * ratios[1]))
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_steps),
    }


@dataclass
class Scaler:
    mean: np.ndarray | float
    std: np.ndarray | float

    def transform(self, x, mask=None):
        z = (np.asarray(x, dtype=np.float32) - self.mean) / self.std
        if mask is not None:
            z = np.where(mask, z, 0.0).astype(np.float32)  # missing -> train mean
        return z

    def inverse(self, z):
        return z * self.std + self.mean


def zscore_fit(train_values, per_sensor: bool = False) -> Scaler:
    """Fit z-score stats on nonzero TRAIN values (global by default; per-sensor optional)."""
    v = np.asarray(train_values, dtype=np.float64)
    m = v != 0
    if per_sensor:
        vv = np.where(m, v, np.nan)
        return Scaler(np.nanmean(vv, 0).astype(np.float32), np.nanstd(vv, 0).astype(np.float32))
    return Scaler(float(v[m].mean()), float(v[m].std()))


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def window_starts(start: int, end: int, seq_in: int = 12, seq_out: int = 12) -> np.ndarray:
    """Start indices t such that [t, t+seq_in+seq_out) lies inside [start, end)."""
    return np.arange(start, end - seq_in - seq_out + 1)


def make_windows(arr, mask, tfeat=None, seq_in: int = 12, seq_out: int = 12,
                 include_mask_channel: bool = False, y_arr=None):
    """Sliding windows over one contiguous split (pass already-sliced arrays).

    Args:
        arr:   (T, N) model input values (e.g. normalized, missing filled with 0).
        mask:  (T, N) bool, True = observed.
        tfeat: optional (T, F) time features, broadcast to every sensor.
        include_mask_channel: append the observed-mask as an extra input channel.
        y_arr: (T, N) values to use for targets (default: ``arr``); pass raw mph
               to evaluate/train in original units.

    Returns:
        x (W, seq_in, N, C), y (W, seq_out, N), x_mask (W, seq_in, N), y_mask (W, seq_out, N).
        Uses stride tricks (views) for the value/mask windows; ``x`` is materialized
        only when channels are stacked.
    """
    from numpy.lib.stride_tricks import sliding_window_view as swv

    arr = np.asarray(arr, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    y_arr = arr if y_arr is None else np.asarray(y_arr, dtype=np.float32)
    L = seq_in + seq_out
    W = arr.shape[0] - L + 1
    win = lambda a: swv(a, L, axis=0)[:W].transpose(0, -1, *range(1, a.ndim))  # (W, L, ...)
    xa, ma, ya = win(arr), win(mask), win(y_arr)
    chans = [xa[:, :seq_in, :, None]]
    if tfeat is not None:
        tf = win(np.asarray(tfeat, dtype=np.float32))[:, :seq_in]  # (W, seq_in, F)
        chans.append(np.broadcast_to(tf[:, :, None, :], (W, seq_in, arr.shape[1], tf.shape[-1])))
    if include_mask_channel:
        chans.append(ma[:, :seq_in, :, None].astype(np.float32))
    x = chans[0] if len(chans) == 1 else np.concatenate(chans, axis=-1)
    return x, ya[:, seq_in:], ma[:, :seq_in], ma[:, seq_in:]


# --------------------------------------------------------------------------- #
# Cached processed data (identical inputs for CI2 / CI3)
# --------------------------------------------------------------------------- #
def prepare_processed(raw_dir: str | Path = RAW_DIR, out_dir: str | Path = PROCESSED_DIR,
                      ratios=(0.7, 0.1, 0.2), seq_in: int = 12, seq_out: int = 12) -> dict:
    """Build and cache normalized arrays + metadata. Returns the metadata dict."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metr_la(raw_dir / "METR-LA.h5")
    raw = df.to_numpy(dtype=np.float32)
    mask = missing_mask(raw)
    bounds = chrono_split(len(df), ratios)
    s, e = bounds["train"]
    scaler = zscore_fit(raw[s:e])
    norm = scaler.transform(raw, mask)
    tfeat = time_features(df.index)
    np.savez_compressed(out_dir / "metr_la_processed.npz", raw=raw, norm=norm, mask=mask,
                        tfeat=tfeat, timestamps=df.index.asi8)
    meta = {
        "sensor_ids": list(df.columns),
        "n_steps": len(df), "n_sensors": raw.shape[1],
        "ratios": list(ratios), "splits": {k: list(v) for k, v in bounds.items()},
        "split_times": {k: [str(df.index[a]), str(df.index[b - 1])] for k, (a, b) in bounds.items()},
        "mean": scaler.mean, "std": scaler.std, "normalization": "global z-score, nonzero train values",
        "missing_fill": "0.0 after z-score (= train mean); mask stored separately",
        "seq_in": seq_in, "seq_out": seq_out,
        "tfeat_columns": ["time_of_day_frac", "day_of_week"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_processed(out_dir: str | Path = PROCESSED_DIR):
    """Return (arrays dict, meta dict). Builds the cache on first use."""
    out_dir = Path(out_dir)
    if not (out_dir / "meta.json").exists():
        prepare_processed(out_dir=out_dir)
    with np.load(out_dir / "metr_la_processed.npz") as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads((out_dir / "meta.json").read_text())
    return arrays, meta


# --------------------------------------------------------------------------- #
# Torch Dataset (optional dependency)
# --------------------------------------------------------------------------- #
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # EDA runs without torch
    torch = None
    _TorchDataset = object


class MetrLADataset(_TorchDataset):
    """Lazily sliced windows for one split.

    Each item:
        x      (seq_in, N, C) float32  -- [normalized speed, tod, dow, (mask)]
        y      (seq_out, N)   float32  -- raw mph targets
        y_mask (seq_out, N)   bool     -- True where target observed
    """

    def __init__(self, split: str = "train", out_dir: str | Path = PROCESSED_DIR,
                 include_time: bool = True, include_mask_channel: bool = False,
                 arrays=None, meta=None):
        if torch is None:
            raise ImportError("MetrLADataset requires torch (pip install torch)")
        if arrays is None or meta is None:
            arrays, meta = load_processed(out_dir)
        self.meta = meta
        self.seq_in, self.seq_out = meta["seq_in"], meta["seq_out"]
        start, end = meta["splits"][split]
        self.starts = window_starts(start, end, self.seq_in, self.seq_out)
        feats = [arrays["norm"][..., None]]
        if include_time:
            tf = arrays["tfeat"]
            feats.append(np.broadcast_to(tf[:, None, :], (*arrays["norm"].shape, tf.shape[1])))
        if include_mask_channel:
            feats.append(arrays["mask"][..., None].astype(np.float32))
        self.x = torch.from_numpy(np.concatenate(feats, axis=-1).astype(np.float32))  # (T, N, C)
        self.y = torch.from_numpy(arrays["raw"].astype(np.float32))
        self.m = torch.from_numpy(arrays["mask"])

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = int(self.starts[i])
        a, b = t + self.seq_in, t + self.seq_in + self.seq_out
        return self.x[t:a], self.y[a:b], self.m[a:b]
