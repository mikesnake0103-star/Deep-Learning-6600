"""Self-check for the Graph WaveNet model. Run: PYTHONPATH=. python scripts/check_gwnet.py"""
import torch, numpy as np
from src.config import data_paths
from src.data import load_processed, load_adj
from src.train import build_model, load_split
P = data_paths()
arrays, meta = load_processed(P["processed"])
ids, _, adj = load_adj(P["adj"], sensor_ids=meta["sensor_ids"])
print("adj order matches data columns; shape", adj.shape, "nonzero", int((adj > 0).sum()), "diag>0", int((np.diag(adj) > 0).sum()))
ds = load_split("train", arrays, meta)
x = torch.stack([ds[i][0] for i in range(4)])
base = dict(model="gwnet", data_dir=str(P["data_dir"]), use_mask=True, channels=32, blocks=4, dropout=0.3)
for mode in ["fixed+adaptive", "fixed", "adaptive", "none"]:
    m = build_model({**base, "graph_mode": mode}, meta)
    out = m(x)
    print(f"{mode:15s} params {sum(p.numel() for p in m.parameters()):,} out {tuple(out.shape)} mean {out.mean().item():.1f} rf {m.receptive_field}")
# no-graph model must not let one sensor's input change another sensor's output (eval mode)
m = build_model({**base, "graph_mode": "none"}, meta).eval()
x2 = x.clone(); x2[:, :, 0, 0] += 5
d = (m(x) - m(x2)).abs().amax(dim=(0, 1))
assert d[0] > 0 and d[1:].max() == 0, d[:5]
m = build_model({**base, "graph_mode": "fixed"}, meta).eval()
d = (m(x) - m(x2)).abs().amax(dim=(0, 1))
print("no-graph: only sensor 0 changes (ok). fixed graph: sensors affected by sensor 0 =", int((d > 0).sum()))
