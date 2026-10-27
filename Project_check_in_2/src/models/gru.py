"""Shared-weight GRU encoder + direct multi-horizon MLP head (CI2 neural baseline).

Input is the raw ``MetrLADataset`` item with ``include_time=True,
include_mask_channel=True``: x (B, 12, N, 4) = [norm speed, tod frac, dow, mask].
Every sensor is an independent sequence through the same weights (no sensor
embedding, no graph), so the model is sensor-agnostic and is the exact
"no-graph" ablation of a CI3 spatio-temporal model.
"""
from __future__ import annotations

import math

import torch
from torch import nn

STEPS_PER_DAY = 288  # 5-min resolution


class SharedGRU(nn.Module):
    def __init__(self, mean: float, std: float, seq_out: int = 12, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, head_hidden: int = 128, use_mask: bool = True):
        super().__init__()
        self.seq_out, self.use_mask = seq_out, use_mask
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))
        n_in = 1 + int(use_mask) + 2
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 2 * seq_out, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, seq_out))

    @staticmethod
    def _sincos(tod: torch.Tensor) -> torch.Tensor:
        a = 2 * math.pi * tod
        return torch.stack([torch.sin(a), torch.cos(a)], dim=-1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, N, 4) dataset channels -> (B, T, N, F) model inputs."""
        feats = [x[..., :1]]
        if self.use_mask:
            feats.append(x[..., 3:4])
        feats.append(self._sincos(x[..., 1]))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = x.shape
        f = self.features(x).permute(0, 2, 1, 3).reshape(B * N, T, -1)   # (B*N, T, F)
        _, h = self.gru(f)                                                  # h: (layers, B*N, H)
        # target-step time of day is deterministic on the regular 5-min grid
        k = torch.arange(1, self.seq_out + 1, device=x.device, dtype=x.dtype) / STEPS_PER_DAY
        fut = self._sincos(torch.remainder(x[:, -1, :, 1:2] + k, 1.0))      # (B, N, seq_out, 2)
        z = self.head(torch.cat([h[-1], fut.reshape(B * N, -1)], dim=-1))  # (B*N, seq_out)
        z = z.reshape(B, N, self.seq_out).permute(0, 2, 1)                  # (B, seq_out, N)
        return z * self.std + self.mean
