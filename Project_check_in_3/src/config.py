"""Paths and small helpers shared by train.py, evaluate.py and the notebook.

We don't copy the data into this folder. Everything is read from DATA_DIR,
which points to ../Project_check_in_1/data (use --data-dir to change it).
"""
import random
from pathlib import Path

import numpy as np

CI2_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (CI2_ROOT.parent / "Project_check_in_1" / "data").resolve()
RUNS_DIR = CI2_ROOT / "runs"
FIG_DIR = CI2_ROOT / "figures"


def data_paths(data_dir=DATA_DIR):
    # pass every path explicitly so the copied data.py doesn't use its own defaults
    data_dir = Path(data_dir).resolve()
    raw = data_dir / "raw" / "metr-la-dataset"
    return {
        "data_dir": data_dir,
        "processed": data_dir / "processed",
        "raw": raw,
        "h5": raw / "METR-LA.h5",
        "adj": raw / "adj_METR-LA.pkl",
        "locations": raw / "graph_sensor_locations.csv",
    }


def get_device(name=None):
    import torch

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed=42):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
