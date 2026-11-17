"""Graph WaveNet (Wu et al., IJCAI 2019) for our METR-LA pipeline (CI3 model).

Adapted from the official implementation, github.com/nnzhan/Graph-WaveNet
(MIT license, Copyright (c) 2019 Zonghan Wu). The layer structure is the same:
gated dilated temporal convolutions, diffusion graph convolutions (order 2) over
forward/backward transition matrices plus a learned adaptive adjacency, skip
connections and a two-layer output head.

What we changed:
* Input/output match our CI2 GRU: the model takes the MetrLADataset item
  (batch, 12, n_sensors, 4) and returns (batch, 12, n_sensors) in mph.
* Input features are the same four as the GRU: speed, mask, sin/cos time of day
  (the official code uses speed + time of day).
* One graph_mode flag picks the supports. "none" uses the official no-graph
  branch (a 1x1 conv where the graph convolution would be), so everything else
  in the network stays identical.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.gru import sin_cos

GRAPH_MODES = ["fixed+adaptive", "fixed", "adaptive", "none"]


def transition_matrix(adj):
    """Random-walk transition matrix D^-1 A (rows sum to 1, all-zero rows stay zero)."""
    rowsum = adj.sum(1, keepdims=True)
    return np.divide(adj, rowsum, out=np.zeros_like(adj), where=rowsum > 0).astype(np.float32)


class GraphConv(nn.Module):
    """Diffusion convolution: mix x with A x and A^2 x for every support A."""

    def __init__(self, c_in, c_out, n_supports, dropout, order=2):
        super().__init__()
        self.order = order
        self.dropout = dropout
        self.mlp = nn.Conv2d((order * n_supports + 1) * c_in, c_out, kernel_size=1)

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            xk = x
            for _ in range(self.order):
                xk = torch.einsum("ncvl,vw->ncwl", xk, a)
                out.append(xk)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, mean, std, adj, graph_mode="fixed+adaptive", seq_out=12, use_mask=True,
                 channels=32, blocks=4, layers=2, dropout=0.3, emb_dim=10):
        super().__init__()
        assert graph_mode in GRAPH_MODES, graph_mode
        self.graph_mode = graph_mode
        self.use_mask = use_mask
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

        n_nodes = adj.shape[0]
        self.use_fixed = "fixed" in graph_mode
        self.use_adaptive = "adaptive" in graph_mode
        if self.use_fixed:
            self.register_buffer("fwd", torch.from_numpy(transition_matrix(adj)))
            self.register_buffer("bwd", torch.from_numpy(transition_matrix(adj.T)))
        if self.use_adaptive:
            self.nodevec1 = nn.Parameter(torch.randn(n_nodes, emb_dim))
            self.nodevec2 = nn.Parameter(torch.randn(emb_dim, n_nodes))
        n_supports = 2 * self.use_fixed + self.use_adaptive

        n_features = 3 + int(use_mask)  # speed, (mask), sin/cos time of day
        skip_channels = channels * 8
        end_channels = channels * 16
        self.start_conv = nn.Conv2d(n_features, channels, kernel_size=1)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.mix = nn.ModuleList()  # graph conv, or a 1x1 conv when graph_mode is "none"
        self.bn = nn.ModuleList()
        self.receptive_field = 1
        for _ in range(blocks):
            dilation = 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(channels, channels, kernel_size=(1, 2), dilation=dilation))
                self.gate_convs.append(nn.Conv2d(channels, channels, kernel_size=(1, 2), dilation=dilation))
                self.skip_convs.append(nn.Conv2d(channels, skip_channels, kernel_size=1))
                if n_supports > 0:
                    self.mix.append(GraphConv(channels, channels, n_supports, dropout))
                else:
                    self.mix.append(nn.Conv2d(channels, channels, kernel_size=1))
                self.bn.append(nn.BatchNorm2d(channels))
                self.receptive_field += dilation
                dilation *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv2d(end_channels, seq_out, kernel_size=1)

    def adaptive_adj(self):
        return F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)

    def supports(self):
        s = []
        if self.use_fixed:
            s += [self.fwd, self.bwd]
        if self.use_adaptive:
            s.append(self.adaptive_adj())
        return s

    def features(self, x):
        speed = x[..., 0:1]
        tod = x[..., 1]
        mask = x[..., 3:4]
        if self.use_mask:
            return torch.cat([speed, mask, sin_cos(tod)], dim=-1)
        return torch.cat([speed, sin_cos(tod)], dim=-1)

    def forward(self, x):
        h = self.features(x).permute(0, 3, 2, 1)  # (batch, features, sensors, time)
        # the stack needs 13 input steps, so pad one zero step on the left (as the official code does)
        if h.size(3) < self.receptive_field:
            h = F.pad(h, (self.receptive_field - h.size(3), 0, 0, 0))
        h = self.start_conv(h)
        supports = self.supports()

        skip = 0
        for i in range(len(self.filter_convs)):
            residual = h
            h = torch.tanh(self.filter_convs[i](residual)) * torch.sigmoid(self.gate_convs[i](residual))
            s = self.skip_convs[i](h)
            if i > 0:
                skip = skip[..., -s.size(3):]
            skip = s + skip
            h = self.mix[i](h, supports) if supports else self.mix[i](h)
            h = h + residual[..., -h.size(3):]
            h = self.bn[i](h)

        out = F.relu(skip)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)  # (batch, 12, sensors, 1)
        return out.squeeze(-1) * self.std + self.mean  # back to mph
