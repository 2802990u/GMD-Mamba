"""
Qualitative BraTS figure, three clean standalone axial panels:
  <OUT>_image.png   plain MRI axial slice (model input)
  <OUT>_gt.png      same slice, ground-truth tumour overlay
  <OUT>_pred.png    same slice, GMD-ResMamba predicted overlay

Picks ONE test volume, runs sliding-window inference, takes the axial slice
(top-down view) with the most enhancing tumour, and renders each panel smoothly.
Regions: necrotic core=blue, edema=red, enhancing=cyan (same map for GT & pred).

Run in Envi, from ~/LW3DSegmentation/brats_segv2, GPU free:
  CKPT=logs/resmamba_abl_gmd_deep/model/best_model_0.8682.pt \
  SCAN=grouped MAMBA_STAGES="2,3" BG_CHANNEL=3 python make_brats_figure.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.inferers import SlidingWindowInferer
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
from resmamba_model import ResMamba3D

WORK = os.path.expanduser("~/LW3DSegmentation/brats_segv2")
data_dir = f"{WORK}/data/fullres/train"
split_dir = f"{WORK}/splits"
CKPT = os.environ.get("CKPT", f"{WORK}/logs/resmamba_abl_gmd_deep/model/best_model_0.8682.pt")
SCAN = os.environ.get("SCAN", "grouped")
_ms = os.environ.get("MAMBA_STAGES", "2,3").strip()
MAMBA_STAGES = tuple(int(x) for x in _ms.split(",") if x.strip()) if _ms else ()
WANT = os.environ.get("NAME", "").strip()
OUT = os.environ.get("OUT", f"{WORK}/brats_gmd_qualitative.png")
BG_CHANNEL = int(os.environ.get("BG_CHANNEL", "3"))   # 0 t1n 1 t1c 2 t2w 3 t2f(FLAIR)
ALPHA = float(os.environ.get("ALPHA", "0.5"))
MARGIN = int(os.environ.get("MARGIN", "8"))
patch = [128, 128, 128]
device = "cuda:0"

# necrotic(1)=blue, edema(2)=red, enhancing(3)=cyan
CMAP = ListedColormap([(0.15, 0.45, 1.0), (0.90, 0.17, 0.17), (0.10, 0.95, 0.95)])


def read_ids(name):
    with open(os.path.join(split_dir, f"{name}.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def filt(sd):
    if "module" in sd:
        sd = sd["module"]
    return {(k[7:] if str(k).startswith("module") else str(k)): v for k, v in sd.items()}


def window(a):
    fg = a[a > a.min() + 1e-6]
    lo, hi = np.percentile(fg, 0.5), np.percentile(fg, 99.5)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def save_panel(bg, lab, out):
    """One clean panel: smooth grayscale MRI + optional translucent overlay."""
    h, w = bg.shape
    fig = plt.figure(figsize=(w / 80, h / 80), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(window(bg), cmap="gray", interpolation="bicubic")
    if lab is not None:
        # mask anything that is not a real tumour label (1/2/3); the GT volume's
        # out-of-brain voxels carry a fill value, not 0, so mask by validity
        m = np.ma.masked_where((lab < 1) | (lab > 3), lab)
        ax.imshow(m, cmap=CMAP, vmin=1, vmax=3, alpha=ALPHA, interpolation="nearest")
    fig.savefig(out, dpi=200, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    print("checkpoint:", CKPT, "scan:", SCAN, "stages:", MAMBA_STAGES)
    model = ResMamba3D(4, 4, init_filters=32, scan=SCAN, mamba_stages=MAMBA_STAGES,
                       deep_supervision=False)
    model.load_state_dict(filt(torch.load(CKPT, map_location="cpu")))
    model.eval().to(device)
    inferer = SlidingWindowInferer(roi_size=patch, sw_batch_size=2, overlap=0.5, mode="gaussian")

    _, test_ds = get_train_test_loader_from_test_list(data_dir, test_list=read_ids("test_ids"))

    chosen = None
    with torch.no_grad():
        for batch in test_ds:
            name = batch["properties"]["name"]
            if WANT and name != WANT:
                continue
            img = torch.as_tensor(batch["data"]).float()
            if img.ndim == 4:
                img = img.unsqueeze(0)
            seg = torch.as_tensor(batch["seg"])
            gt = (seg[0] if seg.ndim == 4 else seg).cpu().numpy().astype(np.int32)
            if gt.ndim == 4:
                gt = gt[0]
            logits = inferer(img.to(device), model)
            pred = logits.argmax(1)[0].cpu().numpy()
            vol = img[0, BG_CHANNEL].cpu().numpy()
            print("volume shape:", vol.shape)

            # DIAGNOSTIC: dump the plain MRI mid-slice for all three axes so we
            # can see which one is the true axial (top-down) view, then set AXIS.
            if os.environ.get("DIAG", "0") == "1":
                for ax in (0, 1, 2):
                    mid = vol.shape[ax] // 2
                    sl = np.rot90(np.take(vol, mid, axis=ax))
                    save_panel(sl, None, OUT.replace(".png", f"_plane{ax}.png"))
                print("saved _plane0/_plane1/_plane2 (plain MRI mid-slices)")
                chosen = name
                break

            # which array axis is axial (top-down). Overridable once known.
            axial = int(os.environ.get("AXIS", str(int(np.argmin(vol.shape)))))
            other = tuple(a for a in (0, 1, 2) if a != axial)
            prof = (gt == 3).sum(axis=other)
            if prof.max() == 0:
                prof = (gt > 0).sum(axis=other)
            s = int(prof.argmax())

            def take(a):
                return np.rot90(np.take(a, s, axis=axial))
            bg, gt_s, pr_s = take(vol), take(gt), take(pred)

            # tight square-ish crop around the brain
            ys, xs = np.where(bg > bg.min() + 1e-6)
            r0, r1 = max(ys.min() - MARGIN, 0), min(ys.max() + MARGIN, bg.shape[0])
            c0, c1 = max(xs.min() - MARGIN, 0), min(xs.max() + MARGIN, bg.shape[1])
            bg, gt_s, pr_s = bg[r0:r1, c0:c1], gt_s[r0:r1, c0:c1], pr_s[r0:r1, c0:c1]

            img_out = OUT.replace(".png", "_image.png")
            gt_out = OUT.replace(".png", "_gt.png")
            pred_out = OUT.replace(".png", "_pred.png")
            save_panel(bg, None,  img_out)
            save_panel(bg, gt_s,  gt_out)
            save_panel(bg, pr_s,  pred_out)
            print(f"saved {img_out}, {gt_out}, {pred_out}  "
                  f"(case {name}, axial axis={axial}, slice {s}, shape {bg.shape})")
            chosen = name
            break

    if chosen is None:
        print("no case matched — check NAME or test_ids.txt")
