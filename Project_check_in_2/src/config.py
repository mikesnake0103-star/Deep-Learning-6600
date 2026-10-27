"""Shared paths and runtime helpers for CI2.

Data is NOT copied into Project_check_in_2: everything is read from DATA_DIR,
which defaults to ../Project_check_in_1/data and can be overridden with
``--data-dir`` (train/evaluate).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

CI2_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (CI2_ROOT.parent / "Project_check_in_1" / "data").resolve()
RUNS_DIR = CI2_ROOT / "runs"
FIG_DIR = CI2_ROOT / "figures"


def data_paths(data_dir: str | Path = DATA_DIR) -> dict[str, Path]:
    """Explicit paths, so the copied src/data.py never falls back to its own REPO_ROOT defaults."""
    d = Path(data_dir).resolve()
    raw = d / "raw" / "metr-la-dataset"
    return {"data_dir": d, "processed": d / "processed", "raw": raw,
            "h5": raw / "METR-LA.h5", "adj": raw / "adj_METR-LA.pkl",
            "locations": raw / "graph_sensor_locations.csv"}


def get_device(prefer: str | None = None):
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
