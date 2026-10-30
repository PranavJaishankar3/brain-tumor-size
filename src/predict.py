"""Run the trained pipeline on one scan (.png/.jpg or Figshare .mat) and save an overlay.

python src/predict.py path/to/scan.png --run runs/seg_clahe
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import SIZE_NAMES, area_fraction, foreground_mask, get_device, load_meta, preprocess, size_class
from evaluate import keep_largest_in_fg
from models import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--run", default="runs/seg_clahe")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    meta = load_meta(args.data)
    ck = torch.load(Path(args.run) / "best.pt", map_location=device)
    model = UNet(base=ck["base"]).to(device).eval()
    model.load_state_dict(ck["model"])

    p = Path(args.image)
    if p.suffix == ".mat":
        from prepare_data import load_mat
        img = load_mat(p)[0]
    else:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    s = meta["img_size"]
    img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
    fg = foreground_mask(img)

    with torch.no_grad():
        x = torch.from_numpy(preprocess(img, fg, ck["preproc"]))[None, None].float().to(device)
        pred = (torch.sigmoid(model(x))[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    pred = keep_largest_in_fg(pred, fg)
    frac = area_fraction(pred, fg)
    cat = SIZE_NAMES[size_class(frac, meta["thresholds"])] if pred.sum() else "no tumor detected"

    out = Path(args.out or p.with_name(p.stem + "_pred.png"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(img, cmap="gray")
    ov = np.zeros((*pred.shape, 4)); ov[pred > 0] = (1, 0, 0, 0.45)
    ax[1].imshow(ov); ax[1].set_title(f"{cat}  ({100 * frac:.2f}% of head area)"); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close(fig)
    print(f"Size category: {cat}  | tumor area = {100 * frac:.2f}% of head region  | overlay: {out}")


if __name__ == "__main__":
    main()
