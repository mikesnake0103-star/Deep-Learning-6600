"""Shared-weight GRU with a direct 12-step MLP head (our CI2 baseline).

Input x has shape (batch, 12, n_sensors, 4) with channels
[normalized speed, time of day (0-1), day of week, observed mask],
which is what MetrLADataset returns.

Each sensor is run through the same GRU on its own. There are no sensor
embeddings and no graph, so this is the "no-graph" version of the CI3 model.
"""
import math

import torch
from torch import nn

STEPS_PER_DAY = 288  # 5-minute data


def sin_cos(tod):
    angle = 2 * math.pi * tod
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


class SharedGRU(nn.Module):
    def __init__(self, mean, std, seq_out=12, hidden=64, layers=2, dropout=0.1, head_hidden=128, use_mask=True):
        super().__init__()
        self.seq_out = seq_out
        self.use_mask = use_mask
        # stored as buffers so they are saved in the checkpoint
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

        n_features = 1 + 2  # speed + sin/cos of time of day
        if use_mask:
            n_features += 1
        if layers == 1:
            dropout = 0.0  # GRU only applies dropout between layers
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, dropout=dropout, batch_first=True)

        # head input: last hidden state + sin/cos time of day of the 12 target steps
        self.head = nn.Sequential(
            nn.Linear(hidden + 2 * seq_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, seq_out),
        )

    def features(self, x):
        speed = x[..., 0:1]
        tod = x[..., 1]
        mask = x[..., 3:4]
        if self.use_mask:
            return torch.cat([speed, mask, sin_cos(tod)], dim=-1)
        return torch.cat([speed, sin_cos(tod)], dim=-1)

    def forward(self, x):
        batch, steps, n_sensors, _ = x.shape

        # treat every (window, sensor) pair as its own sequence
        f = self.features(x)
        f = f.permute(0, 2, 1, 3).reshape(batch * n_sensors, steps, -1)
        _, h = self.gru(f)
        last_hidden = h[-1]  # (batch * n_sensors, hidden)

        # time of day of the future steps: last input time + k * 5 min
        k = torch.arange(1, self.seq_out + 1, device=x.device, dtype=x.dtype) / STEPS_PER_DAY
        future_tod = torch.remainder(x[:, -1, :, 1:2] + k, 1.0)
        future_tod = sin_cos(future_tod).reshape(batch * n_sensors, -1)

        z = self.head(torch.cat([last_hidden, future_tod], dim=-1))
        z = z.reshape(batch, n_sensors, self.seq_out).permute(0, 2, 1)  # (batch, 12, n_sensors)
        return z * self.std + self.mean  # back to mph
