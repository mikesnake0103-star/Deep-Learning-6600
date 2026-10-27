"""Train the shared-weight GRU on METR-LA.

Run from Project_check_in_2/:
    python -m src.train --overfit            # sanity check on ~200 train windows
    python -m src.train --time-epoch         # time one epoch to estimate the full run
    python -m src.train --run-name gru_h64   # full training, saves to runs/gru_h64/
"""
import argparse
import csv
import json
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
    p.add_argument("--data-dir", default=str(DATA_DIR), help="folder with processed/ and raw/")
    p.add_argument("--run-name", default="gru_h64")
    p.add_argument("--runs-dir", default=str(RUNS_DIR))

    # model
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head-hidden", type=int, default=128)
    p.add_argument("--no-mask", dest="use_mask", action="store_false", help="don't feed the mask channel")

    # training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--es-patience", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: pick automatically)")

    # sanity checks
    p.add_argument("--overfit", action="store_true", help="try to memorize a few train windows")
    p.add_argument("--overfit-windows", type=int, default=200)
    p.add_argument("--overfit-steps", type=int, default=100)
    p.add_argument("--time-epoch", action="store_true", help="run one epoch, print the time and exit")
    return p.parse_args(argv)


def log(*args):
    print(*args, flush=True)


def build_model(cfg, meta):
    return SharedGRU(
        meta["mean"],
        meta["std"],
        seq_out=meta["seq_out"],
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        dropout=cfg["dropout"],
        head_hidden=cfg["head_hidden"],
        use_mask=cfg["use_mask"],
    )


def load_split(split, arrays, meta):
    # 4 channels: [norm speed, tod, dow, mask] (the model expects this layout)
    return MetrLADataset(split, arrays=arrays, meta=meta, include_time=True, include_mask_channel=True)


@torch.no_grad()
def predict(model, loader, device):
    """Returns predictions and targets in mph as numpy arrays of shape (windows, 12, sensors)."""
    model.eval()
    preds = []
    targets = []
    for x, y, _ in loader:
        preds.append(model(x.to(device)).cpu())
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def train_step(model, x, y, opt, clip):
    loss = masked_mae(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return loss.item()


def train_epoch(model, loader, opt, device, clip):
    model.train()
    total = 0.0
    count = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        loss = train_step(model, x, y, opt, clip)
        total += loss * len(x)
        count += len(x)
    return total / max(count, 1)


def run_overfit(cfg, arrays, meta, device):
    train = load_split("train", arrays, meta)
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(train), cfg["overfit_windows"], replace=False)
    loader = DataLoader(Subset(train, idx), batch_size=cfg["batch_size"], shuffle=True)

    # no dropout and no weight decay here, since we want it to memorize
    model = build_model({**cfg, "dropout": 0.0}, meta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    n_seq = len(idx) * meta["n_sensors"]
    log(f"overfit check: {len(idx)} train windows ({n_seq:,} sensor-sequences), "
        f"{cfg['overfit_steps']} steps, dropout 0, no weight decay, device {device}")

    model.train()
    losses = []
    while len(losses) < cfg["overfit_steps"]:
        for x, y, _ in loader:
            loss = train_step(model, x.to(device), y.to(device), opt, cfg["clip"])
            losses.append(loss)
            step = len(losses)
            if step == 1 or step % 10 == 0:
                log(f"  step {step:4d}  masked MAE {loss:.3f} mph")
            if step >= cfg["overfit_steps"]:
                break
    log(f"overfit: first {losses[0]:.3f} -> last {losses[-1]:.3f} mph (min {min(losses):.3f})")
    return losses


def time_one_epoch(model, train_loader, val_loader, opt, device, cfg):
    t0 = time.time()
    train_loss = train_epoch(model, train_loader, opt, device, cfg["clip"])
    t_train = time.time() - t0

    t0 = time.time()
    pred, y = predict(model, val_loader, device)
    t_val = time.time() - t0

    epoch_sec = t_train + t_val
    log(f"1 epoch: train {t_train:.1f}s + val {t_val:.1f}s = {epoch_sec:.1f}s | "
        f"train loss {train_loss:.3f} | val MAE {masked_mae(pred, y):.3f}")
    log(f"projection: {cfg['max_epochs']} epochs (upper bound) = {epoch_sec * cfg['max_epochs'] / 60:.1f} min")


def save_metrics(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(cfg, arrays, meta, device, out):
    train = load_split("train", arrays, meta)
    val = load_split("val", arrays, meta)
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val, batch_size=256, shuffle=False)

    model = build_model(cfg, meta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg["lr_factor"], patience=cfg["lr_patience"])
    log(f"device {device} | params {n_params:,} | train windows {len(train):,} "
        f"({len(train_loader)} batches) | val {len(val):,}")

    if cfg["time_epoch"]:
        time_one_epoch(model, train_loader, val_loader, opt, device, cfg)
        return

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    best = float("inf")
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, opt, device, cfg["clip"])
        pred, y = predict(model, val_loader, device)
        val_mae = float(masked_mae(pred, y))
        report = horizon_report(pred, y)
        lr = opt.param_groups[0]["lr"]
        sched.step(val_mae)
        epoch_sec = time.time() - t0

        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_mae,
            "val_mae_15": report["15 min"]["MAE"],
            "val_mae_30": report["30 min"]["MAE"],
            "val_mae_60": report["60 min"]["MAE"],
            "lr": lr,
            "epoch_sec": epoch_sec,
        })

        improved = val_mae < best - 1e-4
        if improved:
            best = val_mae
            epochs_without_improvement = 0
            ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_mae": val_mae}
            torch.save(ckpt, out / "best.pt")
        else:
            epochs_without_improvement += 1

        mae15 = report["15 min"]["MAE"]
        mae30 = report["30 min"]["MAE"]
        mae60 = report["60 min"]["MAE"]
        star = " *" if improved else ""
        log(f"epoch {epoch:3d} | train {train_loss:.3f} | val {val_mae:.3f} "
            f"(15/30/60: {mae15:.2f}/{mae30:.2f}/{mae60:.2f}) | lr {lr:.1e} | {epoch_sec:.1f}s{star}")
        save_metrics(rows, out / "metrics.csv")

        if epochs_without_improvement >= cfg["es_patience"]:
            log(f"early stopping: no val improvement for {epochs_without_improvement} epochs")
            break

    best_row = min(rows, key=lambda r: r["val_mae"])
    summary = {
        "epochs_trained": len(rows),
        "best_epoch": int(best_row["epoch"]),
        "best_val_mae": best,
        "wall_clock_min": (time.time() - start) / 60,
        "n_params": n_params,
        "device": str(device),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_curve(rows, out / "learning_curve.png")
    log(f"done: {json.dumps(summary)}")


def plot_curve(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]
    best = min(rows, key=lambda r: r["val_mae"])

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(epochs, [r["train_loss"] for r in rows], "o-", ms=3, color="#2a78d6", label="train masked MAE")
    ax.plot(epochs, [r["val_mae"] for r in rows], "o-", ms=3, color="#eb6834", label="val masked MAE")
    ax.axvline(best["epoch"], color="#8a8984", ls="--", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MAE (mph)")
    ax.set_title(f"GRU learning curve (best val {best['val_mae']:.3f} @ epoch {best['epoch']})")
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

    def run(dev):
        if args.overfit:
            run_overfit(cfg, arrays, meta, dev)
            return
        if not args.time_epoch:
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps({**cfg, "device": str(dev)}, indent=2))
        fit(cfg, arrays, meta, dev, out)

    try:
        run(device)
    except (NotImplementedError, RuntimeError) as e:
        # some ops aren't supported on MPS; if that happens just redo it on CPU
        if device.type != "mps":
            raise
        log(f"MPS failed ({type(e).__name__}: {e}), falling back to CPU")
        seed_everything(args.seed)
        run(torch.device("cpu"))


if __name__ == "__main__":
    main()
