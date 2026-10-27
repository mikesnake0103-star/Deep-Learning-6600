"""Train the shared-weight GRU baseline on METR-LA.

Examples (run from Project_check_in_2/):
    python -m src.train --overfit                 # sanity check: memorize ~200 train windows
    python -m src.train --time-epoch              # time one epoch, project full-run cost
    python -m src.train --run-name gru_h64        # full training -> runs/gru_h64/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import DATA_DIR, RUNS_DIR, data_paths, get_device, seed_everything
from src.data import MetrLADataset, load_processed
from src.metrics import horizon_report, masked_mae
from src.models.gru import SharedGRU


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR), help="folder with processed/ and raw/ (default: ../Project_check_in_1/data)")
    p.add_argument("--run-name", default="gru_h64")
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    # model
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head-hidden", type=int, default=128)
    p.add_argument("--no-mask", dest="use_mask", action="store_false", help="drop the observed-mask input channel")
    # optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--es-patience", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="force cuda/mps/cpu (default: cuda > mps > cpu)")
    # modes
    p.add_argument("--overfit", action="store_true", help="sanity check on a few train windows")
    p.add_argument("--overfit-windows", type=int, default=200)
    p.add_argument("--overfit-steps", type=int, default=100)
    p.add_argument("--time-epoch", action="store_true", help="time one epoch + validation, then exit")
    return p.parse_args(argv)


def log(*a):
    print(*a, flush=True)


def build_model(cfg: dict, meta: dict) -> SharedGRU:
    return SharedGRU(meta["mean"], meta["std"], seq_out=meta["seq_out"], hidden=cfg["hidden"], layers=cfg["layers"],
                     dropout=cfg["dropout"], head_hidden=cfg["head_hidden"], use_mask=cfg["use_mask"])


def load_split(split: str, arrays, meta):
    return MetrLADataset(split, arrays=arrays, meta=meta, include_time=True, include_mask_channel=True)


@torch.no_grad()
def predict(model, loader, device):
    """Concatenated (W, 12, N) mph predictions and targets (numpy, on CPU)."""
    model.eval()
    preds, ys = [], []
    for x, y, _ in loader:
        preds.append(model(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


def train_epoch(model, loader, opt, device, clip, max_steps=None):
    model.train()
    tot, n = 0.0, 0
    for step, (x, y, _) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        loss = masked_mae(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        tot += loss.item() * len(x)
        n += len(x)
    return tot / max(n, 1)


def run_overfit(cfg, arrays, meta, device):
    train = load_split("train", arrays, meta)
    idx = np.random.default_rng(cfg["seed"]).choice(len(train), cfg["overfit_windows"], replace=False)
    loader = DataLoader(Subset(train, idx), batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    model = build_model({**cfg, "dropout": 0.0}, meta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])  # no weight decay: we *want* to memorize
    log(f"overfit check: {len(idx)} train windows ({len(idx) * meta['n_sensors']:,} sensor-sequences), "
        f"{cfg['overfit_steps']} steps, dropout 0, no weight decay, device {device}")
    step, hist = 0, []
    model.train()
    while step < cfg["overfit_steps"]:
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            loss = masked_mae(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            hist.append(loss.item())
            step += 1
            if step == 1 or step % 10 == 0:
                log(f"  step {step:4d}  masked MAE {loss.item():.3f} mph")
            if step >= cfg["overfit_steps"]:
                break
    log(f"overfit: first {hist[0]:.3f} -> last {hist[-1]:.3f} mph (min {min(hist):.3f})")
    return hist


def fit(cfg, arrays, meta, device, out: Path):
    train = load_split("train", arrays, meta)
    val = load_split("val", arrays, meta)
    g = torch.Generator().manual_seed(cfg["seed"])
    tr_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=g)
    va_loader = DataLoader(val, batch_size=256, shuffle=False)

    model = build_model(cfg, meta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg["lr_factor"], patience=cfg["lr_patience"])
    log(f"device {device} | params {n_params:,} | train windows {len(train):,} ({len(tr_loader)} batches) | val {len(val):,}")

    if cfg["time_epoch"]:
        t0 = time.time()
        tl = train_epoch(model, tr_loader, opt, device, cfg["clip"])
        t_tr = time.time() - t0
        t1 = time.time()
        p, y = predict(model, va_loader, device)
        t_va = time.time() - t1
        ep = t_tr + t_va
        log(f"1 epoch: train {t_tr:.1f}s + val {t_va:.1f}s = {ep:.1f}s | train loss {tl:.3f} | val MAE {masked_mae(p, y):.3f}")
        log(f"projection: {cfg['max_epochs']} epochs (upper bound) = {ep * cfg['max_epochs'] / 60:.1f} min")
        return

    out.mkdir(parents=True, exist_ok=True)
    rows, best, bad, t_start = [], float("inf"), 0, time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        tl = train_epoch(model, tr_loader, opt, device, cfg["clip"])
        p, y = predict(model, va_loader, device)
        va = float(masked_mae(p, y))
        rep = horizon_report(p, y)
        lr = opt.param_groups[0]["lr"]
        sched.step(va)
        dt = time.time() - t0
        rows.append({"epoch": epoch, "train_loss": tl, "val_mae": va, "val_mae_15": rep["15 min"]["MAE"],
                     "val_mae_30": rep["30 min"]["MAE"], "val_mae_60": rep["60 min"]["MAE"], "lr": lr, "epoch_sec": dt})
        improved = va < best - 1e-4
        if improved:
            best, bad = va, 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_mae": va}, out / "best.pt")
        else:
            bad += 1
        log(f"epoch {epoch:3d} | train {tl:.3f} | val {va:.3f} (15/30/60: {rep['15 min']['MAE']:.2f}/"
            f"{rep['30 min']['MAE']:.2f}/{rep['60 min']['MAE']:.2f}) | lr {lr:.1e} | {dt:.1f}s{' *' if improved else ''}")
        with open(out / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        if bad >= cfg["es_patience"]:
            log(f"early stopping: no val improvement for {bad} epochs")
            break
    total = time.time() - t_start
    summary = {"epochs_trained": len(rows), "best_epoch": int(min(rows, key=lambda r: r["val_mae"])["epoch"]),
               "best_val_mae": best, "wall_clock_min": total / 60, "n_params": n_params, "device": str(device)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_curve(rows, out / "learning_curve.png")
    log(f"done: {json.dumps(summary)}")


def plot_curve(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(ep, [r["train_loss"] for r in rows], "o-", ms=3, color="#2a78d6", label="train masked MAE")
    ax.plot(ep, [r["val_mae"] for r in rows], "o-", ms=3, color="#eb6834", label="val masked MAE")
    b = min(rows, key=lambda r: r["val_mae"])
    ax.axvline(b["epoch"], color="#8a8984", ls="--", lw=1)
    ax.set(xlabel="epoch", ylabel="masked MAE (mph)", title=f"GRU learning curve (best val {b['val_mae']:.3f} @ epoch {b['epoch']})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    cfg = vars(args)
    paths = data_paths(args.data_dir)
    cfg["data_dir"] = str(paths["data_dir"])
    seed_everything(args.seed)
    arrays, meta = load_processed(paths["processed"])
    device = get_device(args.device)
    out = Path(args.runs_dir) / args.run_name
    cfg["device_requested"] = str(device)
    log(f"config: {json.dumps(cfg)}")

    def go(dev):
        if args.overfit:
            return run_overfit(cfg, arrays, meta, dev)
        if not args.time_epoch:
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps({**cfg, "device": str(dev)}, indent=2))
        return fit(cfg, arrays, meta, dev, out)

    try:
        go(device)
    except (NotImplementedError, RuntimeError) as e:
        if device.type != "mps":
            raise
        log(f"!! MPS failed ({type(e).__name__}: {e}); falling back to CPU")
        seed_everything(args.seed)
        go(torch.device("cpu"))


if __name__ == "__main__":
    sys.exit(main())
