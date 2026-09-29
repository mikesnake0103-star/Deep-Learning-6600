"""Fetch auxiliary METR-LA files from the DCRNN repository.

Usage (from the repo root):
    python scripts/download_data.py

What this does
--------------
* Downloads ``graph_sensor_locations.csv`` (sensor id, latitude, longitude) from
  the DCRNN repo into ``data/raw/metr-la-dataset/``.
* Checks that ``METR-LA.h5`` and ``adj_METR-LA.pkl`` are present. These are NOT
  downloaded here -- see ``data/README.md`` for where to get them
  (Kaggle: https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset,
  or the DCRNN README Google Drive / Baidu links).

A failed download is non-fatal: the EDA notebook skips the spatial map when the
locations file is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "metr-la-dataset"
DCRNN_RAW = "https://raw.githubusercontent.com/liyaguang/DCRNN/master"
FILES = {
    "graph_sensor_locations.csv": f"{DCRNN_RAW}/data/sensor_graph/graph_sensor_locations.csv",
}
REQUIRED_LOCAL = ["METR-LA.h5", "adj_METR-LA.pkl"]


def download(url: str, dest: Path, timeout: float = 30.0) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] could not download {url}: {e}")
        return False
    dest.write_bytes(r.content)
    print(f"[OK] {dest.name} ({len(r.content):,} bytes)")
    return True


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_LOCAL:
        p = RAW_DIR / name
        status = "found" if p.exists() else "MISSING -- download from https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset (see data/README.md)"
        print(f"[{'OK' if p.exists() else 'WARN'}] {name}: {status}")

    for name, url in FILES.items():
        dest = RAW_DIR / name
        if dest.exists():
            print(f"[OK] {name} already present, skipping")
            continue
        if not download(url, dest):
            print(f"[WARN] {name} unavailable; spatial map in the EDA notebook will be skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
