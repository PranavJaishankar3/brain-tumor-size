"""Evaluate trained runs on the held-out TEST patients and write a comparison table.

For segmentation runs (segment-then-measure):
  - Dice / IoU of the tumor region
  - size-category accuracy + macro-F1 using predicted tumor area / head area
  - area-fraction MAE
  - post-processing comparison: raw threshold vs. largest-component-inside-head
For classifier runs (direct baseline): accuracy + macro-F1.
For both: consistency under scan-intensity shifts (gamma/contrast perturbations) =
  % of test scans whose size category stays the same.

Usage: python src/evaluate.py runs/seg_raw runs/seg_zscore runs/seg_clahe runs/cls_clahe
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import SIZE_NAMES, area_fraction, get_device, load_meta, preprocess, size_class
from dataset import read_index
from models import UNet, size_classifier

PERTURB = [(0.7, 1.0), (1.4, 1.0), (1.0, 0.6), (0.8, 1.5)]  # (gamma, contrast scale)


def perturb(img, gamma, scale):
    x = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (x ** gamma) * scale * 1000.0


def keep_largest_in_fg(pred, fg):
    pred = (pred & (fg > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pred)
    if n <= 1:
        return pred
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def macro_f1(y, p, k=3):
    f = []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum()
        pr, rc = tp / max((p == c).sum(), 1), tp / max((y == c).sum(), 1)
        f.append(2 * pr * rc / max(pr + rc, 1e-8))
    return float(np.mean(f))


def confusion(y, p, k=3):
    m = np.zeros((k, k), int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m.tolist()


def load_run(run, device):
    ck = torch.load(Path(run) / "best.pt", map_location=device)
    if "base" in ck:
        m, kind = UNet(base=ck["base"]), "seg"
    else:
        m, kind = size_classifier(pretrained=False), "cls"
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), kind, ck["preproc"]


@torch.no_grad()
def infer(model, kind, img, fg, preproc, device):
    x = torch.from_numpy(preprocess(img, fg, preproc))[None, None].float().to(device)
    out = model(x)
    if kind == "seg":
        return (torch.sigmoid(out)[0, 0] > 0.5).cpu().numpy()
    return int(out.argmax(1).item())


def save_overlays(samples, path, n=8):
    n = min(n, len(samples))
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    axes = np.array(axes).reshape(2, n)
    for j, (img, gt, pr, ytrue, ypred) in enumerate(samples[:n]):
        for i, (m, col, title) in enumerate([(gt, (0, 1, 0), f"GT: {SIZE_NAMES[ytrue]}"),
                                              (pr, (1, 0, 0), f"Pred: {SIZE_NAMES[ypred]}")]):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray")
            ov = np.zeros((*m.shape, 4))
            ov[m > 0] = (*col, 0.45)
            ax.imshow(ov)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_run(run, data_dir, device, fig_dir):
    model, kind, preproc = load_run(run, device)
    meta = load_meta(data_dir)
    th = meta["thresholds"]
    rows = read_index(data_dir, "test")
    sdir = Path(data_dir) / "samples"

    y_true, preds_raw, preds_pp, consist = [], [], [], []
    dice, iou, mae, examples = [], [], [], []
    for r in rows:
        d = np.load(sdir / f"{r['id']}.npz")
        img, gt, fg = d["image"], d["mask"], d["fg"]
        yt = int(r["size_class"])
        y_true.append(yt)
        out = infer(model, kind, img, fg, preproc, device)
        if kind == "seg":
            pr_raw = out.astype(np.uint8)
            pr = keep_largest_in_fg(out.astype(np.uint8), fg)
            preds_raw.append(size_class(area_fraction(pr_raw, fg), th))
            frac = area_fraction(pr, fg)
            yp = size_class(frac, th)
            preds_pp.append(yp)
            inter = (pr & gt).sum()
            dice.append((2 * inter + 1) / (pr.sum() + gt.sum() + 1))
            iou.append((inter + 1) / ((pr | gt).sum() + 1))
            mae.append(abs(frac - float(r["area_frac"])))
            if len(examples) < 8:
                examples.append((preprocess(img, fg, "raw"), gt, pr, yt, yp))
            pert = [size_class(area_fraction(keep_largest_in_fg(
                infer(model, kind, perturb(img, g, s), fg, preproc, device).astype(np.uint8), fg), fg), th)
                for g, s in PERTURB]
        else:
            yp = out
            preds_pp.append(yp)
            pert = [infer(model, kind, perturb(img, g, s), fg, preproc, device) for g, s in PERTURB]
        consist.append(np.mean([p == yp for p in pert]))

    y = np.array(y_true)
    p = np.array(preds_pp)
    res = {"run": str(run), "kind": kind, "preproc": preproc, "n_test": len(rows),
           "size_acc": float((y == p).mean()), "size_macro_f1": macro_f1(y, p),
           "consistency": float(np.mean(consist)), "confusion": confusion(y, p)}
    if kind == "seg":
        pr_ = np.array(preds_raw)
        res.update({"dice": float(np.mean(dice)), "iou": float(np.mean(iou)), "area_frac_mae": float(np.mean(mae)),
                    "size_acc_no_postproc": float((y == pr_).mean()),
                    "size_macro_f1_no_postproc": macro_f1(y, pr_)})
        save_overlays(examples, fig_dir / f"{Path(run).name}_overlays.png")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    device = get_device()

    results = []
    for run in args.runs:
        if not (Path(run) / "best.pt").exists():
            print(f"skip {run}: no best.pt")
            continue
        print(f"Evaluating {run} ...")
        r = evaluate_run(run, args.data, device, out)
        results.append(r)
        with open(out / f"{Path(run).name}.json", "w") as f:
            json.dump(r, f, indent=2)

    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "–"
    lines = ["| Run | Method | Preproc | Dice | IoU | Area MAE | Size acc (no post-proc) | Size acc | Size macro-F1 | Consistency |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {Path(r['run']).name} | {'segment→measure' if r['kind'] == 'seg' else 'direct CNN'} | "
                     f"{r['preproc']} | {fmt(r.get('dice'))} | {fmt(r.get('iou'))} | {fmt(r.get('area_frac_mae'))} | "
                     f"{fmt(r.get('size_acc_no_postproc'))} | {fmt(r['size_acc'])} | {fmt(r['size_macro_f1'])} | "
                     f"{fmt(r['consistency'])} |")
    table = "\n".join(lines)
    (out / "summary.md").write_text(
        "# Test-set results (held-out patients)\n\n" + table +
        "\n\nConsistency = share of test scans whose predicted size category is unchanged under "
        "4 simulated scanner intensity shifts (gamma/contrast).\n")
    print("\n" + table)
    print(f"\nWrote {out / 'summary.md'} and overlay figures to {out}/")


if __name__ == "__main__":
    main()
