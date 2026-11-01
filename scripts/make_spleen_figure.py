"""
Qualitative spleen panels (image | GT | GMD-Mamba prediction), matching the
BraTS figure style. Single-class CT: spleen overlaid in one colour.

Run in Envi from ~/LW3DSegmentation (where train_singlesplit/evaluate live):
  SPLEEN_RAW=~/data/Task09_Spleen PP=~/data/spleen_pp \
  CKPT=ckpt_spleen_gmd_s0/best_model.pt SPLIT_DIR=ckpt_spleen_gmd_s0 \
  AXIS=2 python make_spleen_figure.py
Adjust AXIS (0/1/2) if the slice is not a clean top-down axial view; run with
DIAG=1 first to dump the three planes and pick.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.inferers import SlidingWindowInferer
from resmamba_model import ResMamba3D
from msd_dataset import MSDDataset, read_dataset_json

DATA = os.environ["SPLEEN_RAW"]
PP = os.environ["PP"]
CKPT = os.environ.get("CKPT", "ckpt_spleen_gmd_s0/best_model.pt")
SPLIT_DIR = os.environ.get("SPLIT_DIR", "ckpt_spleen_gmd_s0")
OUT = os.environ.get("OUT", "spleen_gmd_qualitative.png")
AXIS = int(os.environ.get("AXIS", "2"))
ALPHA = float(os.environ.get("ALPHA", "0.5"))
MARGIN = int(os.environ.get("MARGIN", "8"))
WANT = os.environ.get("NAME", "").strip()
patch = (128, 128, 128)
device = "cuda"
SPLEEN_RGB = (0.95, 0.30, 0.20)   # single-class overlay colour


def window(a):
    fg = a[a > a.min() + 1e-6]
    lo, hi = np.percentile(fg, 0.5), np.percentile(fg, 99.5)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def save_panel(bg, lab, out):
    h, w = bg.shape
    fig = plt.figure(figsize=(w / 80, h / 80), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(window(bg), cmap="gray", interpolation="bicubic")
    if lab is not None:
        m = np.ma.masked_where(lab < 0.5, lab)
        ax.imshow(m, cmap=ListedColormap([SPLEEN_RGB]), vmin=1, vmax=1,
                  alpha=ALPHA, interpolation="nearest")
    fig.savefig(out, dpi=200, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    meta = read_dataset_json(DATA)
    test_ids = [l.strip() for l in open(os.path.join(SPLIT_DIR, "test_ids.txt")) if l.strip()]
    ds = MSDDataset(DATA, case_ids=test_ids, patch_size=patch, train=False,
                    full_volume=True, meta=meta, preprocessed_dir=PP)
    ARCH = os.environ.get("ARCH", "resmamba")
    if ARCH == "resmamba":
        model = ResMamba3D(1, 1, init_filters=32, scan="grouped", deep_supervision=False).to(device)
    else:                                  # e.g. ARCH=segresnet baseline
        from baseline_models import build_baseline
        model, _ = build_baseline(ARCH, in_channels=1, num_classes=1, patch_size=patch)
        model = model.to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")); model.eval()
    inferer = SlidingWindowInferer(roi_size=patch, sw_batch_size=1, overlap=0.5, mode="gaussian")

    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            name = s.get("name", test_ids[i] if i < len(test_ids) else str(i))
            if WANT and name != WANT:
                continue
            img = s["image"]; tgt = s["target"]
            img = img if torch.is_tensor(img) else torch.as_tensor(img)
            tgt = tgt if torch.is_tensor(tgt) else torch.as_tensor(tgt)
            if img.ndim == 3: img = img.unsqueeze(0)
            vol = img[0].cpu().numpy()                         # (D,H,W)
            gt = (tgt[0] if tgt.ndim == 4 else tgt).cpu().numpy()
            gt = (gt > 0.5).astype(np.uint8)

            logits = inferer(img.unsqueeze(0).to(device), model)   # (1,1,D,H,W)
            pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

            if os.environ.get("DIAG", "0") == "1":
                for ax in (0, 1, 2):
                    mid = vol.shape[ax] // 2
                    save_panel(np.rot90(np.take(vol, mid, ax)), None,
                               OUT.replace(".png", f"_plane{ax}.png"))
                print("saved _plane0/1/2; pick AXIS and rerun"); break

            other = tuple(a for a in (0, 1, 2) if a != AXIS)
            prof = gt.sum(axis=other)
            z = int(prof.argmax()) if prof.max() > 0 else vol.shape[AXIS] // 2

            def take(a): return np.rot90(np.take(a, z, axis=AXIS))
            bg, gt_s, pr_s = take(vol), take(gt), take(pred)
            ys, xs = np.where(bg > bg.min() + 1e-6)
            r0, r1 = max(ys.min()-MARGIN, 0), min(ys.max()+MARGIN, bg.shape[0])
            c0, c1 = max(xs.min()-MARGIN, 0), min(xs.max()+MARGIN, bg.shape[1])
            bg, gt_s, pr_s = bg[r0:r1, c0:c1], gt_s[r0:r1, c0:c1], pr_s[r0:r1, c0:c1]

            save_panel(bg, None,  OUT.replace(".png", "_image.png"))
            save_panel(bg, gt_s,  OUT.replace(".png", "_gt.png"))
            save_panel(bg, pr_s,  OUT.replace(".png", "_pred.png"))
            print(f"saved spleen panels (case {name}, axis={AXIS}, slice {z}, shape {bg.shape})")
            break
