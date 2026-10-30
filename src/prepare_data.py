"""Convert raw .mat files to resized .npz samples, compute tumor size labels,
and make a patient-level train/val/test split (no patient appears in two splits)."""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

from common import SIZE_NAMES, TUMOR_TYPES, area_fraction, foreground_mask, size_class


def load_mat(path):
    import h5py
    try:
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            img = np.array(cj["image"]).T.astype(np.float32)
            mask = np.array(cj["tumorMask"]).T.astype(np.uint8)
            label = int(np.array(cj["label"]).item())
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).flatten())
    except OSError:  # older MATLAB format fallback
        from scipy.io import loadmat
        cj = loadmat(path)["cjdata"][0, 0]
        img = cj["image"].astype(np.float32)
        mask = cj["tumorMask"].astype(np.uint8)
        label = int(cj["label"].item())
        pid = str(cj["PID"].item())
    return img, mask, label, pid.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/mat")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.raw).glob("*.mat"), key=lambda p: (len(p.stem), p.stem))
    if not files:
        raise SystemExit(f"No .mat files in {args.raw}. Run src/download_data.py first.")

    rows = []
    for i, p in enumerate(files):
        img, mask, label, pid = load_mat(p)
        img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
        fg = foreground_mask(img)
        fg = np.maximum(fg, mask)  # tumor always counts as part of the head region
        np.savez_compressed(out / "samples" / f"{p.stem}.npz", image=img, mask=mask, fg=fg)
        rows.append({"id": p.stem, "pid": pid, "tumor_type": TUMOR_TYPES.get(label, str(label)),
                     "area_frac": area_fraction(mask, fg)})
        if (i + 1) % 250 == 0:
            print(f"  processed {i + 1}/{len(files)}")

    # patient-level split
    pids = sorted({r["pid"] for r in rows})
    random.Random(args.seed).shuffle(pids)
    n = len(pids)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    split_of = {p: "train" for p in pids[:n_tr]}
    split_of.update({p: "val" for p in pids[n_tr:n_tr + n_va]})
    split_of.update({p: "test" for p in pids[n_tr + n_va:]})
    for r in rows:
        r["split"] = split_of[r["pid"]]

    # size category thresholds = tertiles of TRAIN tumor area fraction (no leakage)
    tr = np.array([r["area_frac"] for r in rows if r["split"] == "train"])
    thresholds = [float(np.quantile(tr, 1 / 3)), float(np.quantile(tr, 2 / 3))]
    for r in rows:
        r["size_class"] = size_class(r["area_frac"], thresholds)

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {"thresholds": thresholds, "size_names": SIZE_NAMES, "img_size": args.size,
            "counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "val", "test"]},
            "patients": {s: sum(v == s for v in split_of.values()) for s in ["train", "val", "test"]}}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(rows)} samples to {out}")
    print(f"Split (images): {meta['counts']}  (patients): {meta['patients']}")
    print("Size thresholds (tumor area / head area): "
          f"small < {thresholds[0]:.4f} <= medium < {thresholds[1]:.4f} <= large")


if __name__ == "__main__":
    main()
