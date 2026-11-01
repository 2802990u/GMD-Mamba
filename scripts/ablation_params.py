"""Print parameter counts for every ablation variant (the 'per-parameter' axis).
Run in Envi, from the BraTS dir (needs resmamba_model.py + agile_mamba_model.py)."""
import torch
from resmamba_model import ResMamba3D

configs = [
    ("noMamba",      "grouped", ()),        # pure SegResNet (scan irrelevant)
    ("gmd_deep",     "grouped",       (2, 3)),    # ours (ungated)
    ("gmd_gated",    "grouped_gated", (2, 3)),    # ours + selective fusion gate
    ("stas_deep",    "stas",    (2, 3)),
    ("bidir_deep",   "bidir",   (2, 3)),
    ("gmd_shallow1", "grouped", (1, 2, 3)), # more Mamba (adds 64^3 stage)
]
print(f"{'variant':14s} {'scan':8s} {'stages':10s} {'params(M)':>10s}")
for name, scan, ms in configs:
    try:
        m = ResMamba3D(4, 4, init_filters=32, scan=scan, mamba_stages=ms)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name:14s} {scan:8s} {str(ms):10s} {n:10.2f}")
    except Exception as e:
        print(f"{name:14s} {scan:8s} {str(ms):10s}   ERROR: {e}")
