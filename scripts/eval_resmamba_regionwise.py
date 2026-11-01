"""
Region-wise (WT/TC/ET) evaluation of a trained ResMamba3D checkpoint on the
250-case test split, self-contained: uses only the preprocessed .npz (image+GT)
on reznor + monai sliding-window + mirror TTA. No raw niftis, no test_list.py.

This is a close approximation of SegMamba-V2's reported protocol (full-res eval
vs their original-res eval differ only marginally for Dice), so the numbers are
directly interpretable against 91.6 / 91.94.

Run in Envi, from ~/LW3DSegmentation/brats_segv2.
"""
import os
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
from resmamba_model import ResMamba3D

WORK = os.path.expanduser("~/LW3DSegmentation/brats_segv2")
data_dir = f"{WORK}/data/fullres/train"
split_dir = f"{WORK}/splits"
ckpt = os.environ.get("CKPT", f"{WORK}/logs/resmamba_gmd_split/model/best_model_0.8477.pt")

# architecture must match the checkpoint being loaded (ablation variants differ)
SCAN = os.environ.get("SCAN", "grouped")                 # grouped | stas | bidir
_ms = os.environ.get("MAMBA_STAGES", "2,3").strip()      # "2,3" deep; "1,2,3"; "" none
MAMBA_STAGES = tuple(int(x) for x in _ms.split(",") if x.strip() != "") if _ms else ()
print(f"[eval] SCAN={SCAN}  MAMBA_STAGES={MAMBA_STAGES}")

patch = [128, 128, 128]
device = "cuda:0"


def read_ids(name):
    with open(os.path.join(split_dir, f"{name}.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def to_regions(lab):                      # (D,H,W) int -> (3,D,H,W): TC, WT, ET
    return np.stack([((lab == 1) | (lab == 3)),
                     ((lab == 1) | (lab == 3) | (lab == 2)),
                     (lab == 3)]).astype(np.uint8)


def dice(p, g, eps=1e-6):
    p = p.astype(bool); g = g.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    return float(2 * np.logical_and(p, g).sum() / (p.sum() + g.sum() + eps))


# HD95 (95th-percentile Hausdorff distance), BraTS empty-mask convention:
#   both empty -> 0 (perfect);  exactly one empty -> undefined (NaN, excluded
#   from the HD95 average, as is standard);  both present -> medpy hd95.
try:
    from medpy.metric.binary import hd95 as _medpy_hd95
    _HAS_HD = True
except Exception:
    _HAS_HD = False


def hd95(p, g):
    p = p.astype(bool); g = g.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return np.nan
    if not _HAS_HD:
        return np.nan
    return float(_medpy_hd95(p, g))


def filt(sd):
    if "module" in sd:
        sd = sd["module"]
    return {(k[7:] if str(k).startswith("module") else str(k)): v for k, v in sd.items()}


if __name__ == "__main__":
    print("checkpoint:", ckpt)
    model = ResMamba3D(4, 4, init_filters=32, scan=SCAN, mamba_stages=MAMBA_STAGES, deep_supervision=False)
    model.load_state_dict(filt(torch.load(ckpt, map_location="cpu")))
    model.eval().to(device)
    inferer = SlidingWindowInferer(roi_size=patch, sw_batch_size=2, overlap=0.5, mode="gaussian")

    _, test_ds = get_train_test_loader_from_test_list(data_dir, test_list=read_ids("test_ids"))
    names = ["TC", "WT", "ET"]
    res = []
    hds = []
    flips = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
    with torch.no_grad():
        for batch in test_ds:
            img = torch.as_tensor(batch["data"]).float()
            if img.ndim == 4:
                img = img.unsqueeze(0)                 # (1,C,D,H,W)
            img = img.to(device)
            logits = inferer(img, model)               # identity
            for ax in flips:                           # mirror TTA (3 axes)
                logits = logits + torch.flip(inferer(torch.flip(img, ax), model), ax)
            logits = logits / 8.0
            pred = logits.argmax(1)[0].cpu().numpy()   # (D,H,W) in {0..3}

            seg = torch.as_tensor(batch["seg"])
            gt = (seg[0] if seg.ndim == 4 else seg).cpu().numpy().astype(np.int32)

            pr, gr = to_regions(pred), to_regions(gt)
            d = [dice(pr[i], gr[i]) for i in range(3)]
            h = [hd95(pr[i], gr[i]) for i in range(3)]
            res.append(d)
            hds.append(h)
            nm = batch["properties"]["name"]
            print(nm, {names[i]: (round(d[i], 4), round(h[i], 2)) for i in range(3)}, flush=True)

    res = np.array(res)                 # (N, 3)  Dice per case/region
    hds = np.array(hds)                 # (N, 3)  HD95 per case/region (NaN allowed)
    print("\n==== RESULTS (", len(res), "cases ) ====")
    print("  (Dice in %, mean +/- std over cases;  HD95 in mm, nan-excluded)")
    for i in range(3):
        dm, ds = res[:, i].mean() * 100, res[:, i].std() * 100
        hm, hs = np.nanmean(hds[:, i]), np.nanstd(hds[:, i])
        print(f"  {names[i]:>2}:  Dice {dm:5.2f} +/- {ds:4.2f}   HD95 {hm:5.2f} +/- {hs:4.2f}")
    # average across the three regions (per-case, then over cases)
    davg_c = res.mean(axis=1)                       # (N,) per-case region-avg Dice
    havg_c = np.nanmean(hds, axis=1)                # (N,) per-case region-avg HD95
    print(f"  Avg: Dice {davg_c.mean()*100:5.2f} +/- {davg_c.std()*100:4.2f}"
          f"   HD95 {np.nanmean(havg_c):5.2f} +/- {np.nanstd(havg_c):4.2f}")
    if not _HAS_HD:
        print("  [warn] medpy not installed -> HD95 is NaN. pip install medpy")
