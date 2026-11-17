"""Masked regression metrics for METR-LA (ground truth == 0 means missing).

Each function accepts numpy arrays or torch tensors (same type for pred/true)
and returns the same kind of scalar. Torch versions are differentiable, so
``masked_mae`` can be used directly as a training loss. If every target is
masked, the result is 0 (so a training step contributes no gradient).
"""
from __future__ import annotations

import numpy as np


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _mask(y_true, null_val: float = 0.0):
    if _is_torch(y_true):
        import torch

        m = (y_true != null_val).float()
        return m / torch.clamp(m.mean(), min=1e-12) if m.sum() > 0 else m
    m = (np.asarray(y_true) != null_val).astype(np.float64)
    return m / m.mean() if m.sum() > 0 else m


def _nan_to_zero(x):
    if _is_torch(x):
        import torch

        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def masked_mae(y_pred, y_true, null_val: float = 0.0):
    m = _mask(y_true, null_val)
    return _nan_to_zero(abs(y_pred - y_true) * m).mean()


def masked_rmse(y_pred, y_true, null_val: float = 0.0):
    m = _mask(y_true, null_val)
    mse = _nan_to_zero((y_pred - y_true) ** 2 * m).mean()
    return mse ** 0.5


def masked_mape(y_pred, y_true, null_val: float = 0.0):
    """Mean absolute percentage error in percent."""
    m = _mask(y_true, null_val)
    if _is_torch(y_true):
        return 100 * _nan_to_zero(abs(y_pred - y_true) / abs(y_true) * m).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * _nan_to_zero(abs(y_pred - y_true) / abs(y_true) * m).mean()


def horizon_report(y_pred, y_true, horizons=(3, 6, 12), step_minutes: int = 5):
    """Dict {"15 min": {"MAE":..,"RMSE":..,"MAPE":..}, ...} for (W, H, N) numpy arrays."""
    out = {}
    for h in horizons:
        p, t = y_pred[:, h - 1], y_true[:, h - 1]
        out[f"{h * step_minutes} min"] = {
            "MAE": float(masked_mae(p, t)),
            "RMSE": float(masked_rmse(p, t)),
            "MAPE (%)": float(masked_mape(p, t)),
        }
    return out
