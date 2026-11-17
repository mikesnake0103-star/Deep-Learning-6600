"""Non-neural probes: last-value persistence and historical average (dow x time-of-day)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistence(x_raw: np.ndarray, seq_out: int = 12) -> np.ndarray:
    """Repeat the last observed (nonzero) input value per sensor.

    x_raw: (W, seq_in, N) raw mph inputs (0 = missing). If a sensor's whole input
    window is missing, the prediction stays 0; since that is at most a few
    windows it is left as-is (and 0-valued targets are masked anyway).
    """
    seq_in = x_raw.shape[1]
    # index of the last nonzero step (all-zero windows -> last step, whose value is 0)
    idx = seq_in - 1 - np.argmax(x_raw[:, ::-1] != 0, axis=1)  # (W, N)
    last = np.take_along_axis(x_raw, idx[:, None, :], axis=1)[:, 0]
    return np.repeat(last[:, None, :], seq_out, axis=1)


def fit_historical_average(train_df: pd.DataFrame) -> np.ndarray:
    """Mean nonzero speed per (day-of-week, 5-min slot, sensor): array (7, 288, N).

    Slots never observed in train fall back to the sensor's overall train mean.
    """
    v = train_df.replace(0, np.nan)
    slot = train_df.index.hour * 12 + train_df.index.minute // 5
    ha = v.groupby([train_df.index.dayofweek, slot]).mean()
    full = pd.MultiIndex.from_product([range(7), range(288)])
    ha = ha.reindex(full).fillna(v.mean())
    return ha.to_numpy().reshape(7, 288, -1)


def historical_average(ha: np.ndarray, target_index: pd.DatetimeIndex) -> np.ndarray:
    """Look up the HA table for each target timestamp -> (len(index), N)."""
    slot = target_index.hour * 12 + target_index.minute // 5
    return ha[target_index.dayofweek, slot]
