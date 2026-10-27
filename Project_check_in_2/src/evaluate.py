"""Evaluate a trained GRU run on the test split against the CI1 probes.

    python -m src.evaluate --run-name gru_h64

Writes runs/<run>/test_preds.npz (GRU / persistence / HA predictions on the same
6,832 test windows; git-ignored) and figures/results_table.{csv,md}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.baselines import fit_historical_average, historical_average, persistence
from src.config import CI2_ROOT, DATA_DIR, FIG_DIR, RUNS_DIR, data_paths, get_device
from src.data import load_processed
from src.metrics import horizon_report
from src.train import build_model, load_split, predict

HORIZONS = ["15 min", "30 min", "60 min"]
METRICS = ["MAE", "RMSE", "MAPE (%)"]
CI1_PROBES = CI2_ROOT.parent / "Project_check_in_1" / "figures" / "probe_results.csv"
MODEL_NAMES = {"gru": "GRU (shared-weight, ours)", "persistence": "Persistence (last observed)",
               "ha": "Historical average (dow × tod, train)"}


def test_inputs(arrays, meta):
    """Raw-mph inputs, HA predictions and timestamps aligned with the test Dataset windows."""
    s, e = meta["splits"]["test"]
    a, b = meta["splits"]["train"]
    ds = load_split("test", arrays, meta)
    starts = ds.starts
    si, so = meta["seq_in"], meta["seq_out"]
    raw = arrays["raw"]
    x_raw = raw[starts[:, None] + np.arange(si)]                                     # (W, 12, N)
    idx = pd.to_datetime(arrays["timestamps"])
    train_df = pd.DataFrame(raw[a:b], index=idx[a:b], columns=meta["sensor_ids"])
    ha_t = historical_average(fit_historical_average(train_df), idx)                 # (T, N)
    tgt = starts[:, None] + si + np.arange(so)                                        # (W, 12) target time indices
    return ds, starts, x_raw, ha_t[tgt].astype(np.float32), tgt


def results_table(preds: dict, y: np.ndarray) -> pd.DataFrame:
    rows = {}
    for key, p in preds.items():
        rep = horizon_report(p, y)
        rows[MODEL_NAMES[key]] = {(h, m): rep[h][m] for h in HORIZONS for m in METRICS}
    df = pd.DataFrame(rows).T
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["horizon", "metric"])
    df.index.name = "model"
    return df


def to_markdown(df: pd.DataFrame) -> str:
    head = "| model | " + " | ".join(f"{h} {m}" for h, m in df.columns) + " |"
    sep = "|---|" + "---|" * len(df.columns)
    lines = [head, sep]
    for name, r in df.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def check_probes(df: pd.DataFrame):
    ci1 = pd.read_csv(CI1_PROBES, header=[0, 1], index_col=0).dropna(how="all")
    for key in ("persistence", "ha"):
        name = MODEL_NAMES[key]
        ours = df.loc[name].astype(float).round(2)
        theirs = ci1.loc[name].astype(float)
        diff = float(np.abs(ours.values - theirs.values).max())
        print(f"{name}: max |diff| vs CI1 probe_results.csv = {diff:.3f}")
        assert diff <= 0.011, f"{name} does not match CI1 probes"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="gru_h64")
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--device", default=None)
    p.add_argument("--results-dir", default=str(FIG_DIR), help="where results_table.{csv,md} go (default: figures/)")
    p.add_argument("--no-preds", action="store_true", help="skip writing test_preds.npz")
    args = p.parse_args(argv)

    run = Path(args.runs_dir) / args.run_name
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    arrays, meta = load_processed(data_paths(args.data_dir)["processed"])
    device = get_device(args.device)
    model = build_model(ckpt["config"], meta).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {run / 'best.pt'} (epoch {ckpt['epoch']}, val MAE {ckpt['val_mae']:.3f}) on {device}")

    ds, starts, x_raw, pred_ha, tgt = test_inputs(arrays, meta)
    pred_gru, y = predict(model, DataLoader(ds, batch_size=256, shuffle=False), device)
    pred_pers = persistence(x_raw, meta["seq_out"]).astype(np.float32)
    preds = {"gru": pred_gru, "persistence": pred_pers, "ha": pred_ha}

    if not args.no_preds:
        np.savez(run / "test_preds.npz", gru=pred_gru, persistence=pred_pers, ha=pred_ha, y=y,
                 x_missing_frac=(x_raw == 0).mean(1).astype(np.float32), x_last=x_raw[:, -1], starts=starts,
                 target_index=tgt)

    df = results_table(preds, y)
    check_probes(df)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(out / "results_table.csv")
    (out / "results_table.md").write_text(to_markdown(df))
    (run / "test_metrics.json").write_text(json.dumps(horizon_report(pred_gru, y), indent=2))
    print(to_markdown(df))


if __name__ == "__main__":
    main()
