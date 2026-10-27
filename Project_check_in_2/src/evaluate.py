"""Evaluate a trained run on the test split and compare with the CI1 baselines.

    python -m src.evaluate --run-name gru_h64

Saves runs/<run>/test_preds.npz (GRU, persistence and HA predictions on the
same 6,832 test windows; git-ignored) and figures/results_table.csv / .md.
"""
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
MODEL_NAMES = {
    "gru": "GRU (shared-weight, ours)",
    "persistence": "Persistence (last observed)",
    "ha": "Historical average (dow × tod, train)",
}


def test_inputs(arrays, meta):
    """Inputs in mph, HA predictions and target time indices for the test windows."""
    train_start, train_end = meta["splits"]["train"]
    seq_in = meta["seq_in"]
    seq_out = meta["seq_out"]
    raw = arrays["raw"]

    ds = load_split("test", arrays, meta)
    starts = ds.starts
    x_raw = raw[starts[:, None] + np.arange(seq_in)]  # (windows, 12, sensors)

    # fit HA on the train period only, then look it up for every timestamp
    times = pd.to_datetime(arrays["timestamps"])
    train_df = pd.DataFrame(raw[train_start:train_end], index=times[train_start:train_end],
                            columns=meta["sensor_ids"])
    ha_all = historical_average(fit_historical_average(train_df), times)

    target_idx = starts[:, None] + seq_in + np.arange(seq_out)  # (windows, 12)
    pred_ha = ha_all[target_idx].astype(np.float32)
    return ds, starts, x_raw, pred_ha, target_idx


def results_table(preds, y):
    rows = {}
    for key, pred in preds.items():
        report = horizon_report(pred, y)
        rows[MODEL_NAMES[key]] = {(h, m): report[h][m] for h in HORIZONS for m in METRICS}
    df = pd.DataFrame(rows).T
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["horizon", "metric"])
    df.index.name = "model"
    return df


def to_markdown(df):
    header = "| model | " + " | ".join(f"{h} {m}" for h, m in df.columns) + " |"
    lines = [header, "|---|" + "---|" * len(df.columns)]
    for name, row in df.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |")
    return "\n".join(lines) + "\n"


def check_probes(df):
    # our persistence / HA numbers should be the same as the ones reported in CI1
    ci1 = pd.read_csv(CI1_PROBES, header=[0, 1], index_col=0).dropna(how="all")
    for key in ["persistence", "ha"]:
        name = MODEL_NAMES[key]
        ours = df.loc[name].astype(float).round(2).values
        theirs = ci1.loc[name].astype(float).values
        diff = float(np.abs(ours - theirs).max())
        print(f"{name}: max |diff| vs CI1 probe_results.csv = {diff:.3f}")
        assert diff <= 0.011, f"{name} does not match CI1 probes"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="gru_h64")
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--device", default=None)
    p.add_argument("--results-dir", default=str(FIG_DIR), help="where results_table.csv/.md go")
    p.add_argument("--no-preds", action="store_true", help="don't write test_preds.npz")
    args = p.parse_args(argv)

    run = Path(args.runs_dir) / args.run_name
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    arrays, meta = load_processed(data_paths(args.data_dir)["processed"])
    device = get_device(args.device)
    model = build_model(ckpt["config"], meta).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {run / 'best.pt'} (epoch {ckpt['epoch']}, val MAE {ckpt['val_mae']:.3f}) on {device}")

    ds, starts, x_raw, pred_ha, target_idx = test_inputs(arrays, meta)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    pred_gru, y = predict(model, loader, device)
    pred_pers = persistence(x_raw, meta["seq_out"]).astype(np.float32)

    if not args.no_preds:
        np.savez(
            run / "test_preds.npz",
            gru=pred_gru,
            persistence=pred_pers,
            ha=pred_ha,
            y=y,
            x_missing_frac=(x_raw == 0).mean(1).astype(np.float32),
            x_last=x_raw[:, -1],
            starts=starts,
            target_index=target_idx,
        )

    df = results_table({"gru": pred_gru, "persistence": pred_pers, "ha": pred_ha}, y)
    check_probes(df)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(out / "results_table.csv")
    (out / "results_table.md").write_text(to_markdown(df))
    (run / "test_metrics.json").write_text(json.dumps(horizon_report(pred_gru, y), indent=2))
    print(to_markdown(df))


if __name__ == "__main__":
    main()
