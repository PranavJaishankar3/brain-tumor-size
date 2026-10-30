"""Baseline: classify tumor size category directly from the image with a ResNet-18 (no segmentation).
Compared against the segment-then-measure pipeline in evaluate.py."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import PREPROC_MODES, get_device
from dataset import TumorDataset
from models import size_classifier


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for img, _, _, y, _ in loader:
        ps.append(model(img.to(device)).argmax(1).cpu())
        ys.append(y)
    ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    f1s = []
    for c in range(3):
        tp = ((ps == c) & (ys == c)).sum()
        prec = tp / max((ps == c).sum(), 1)
        rec = tp / max((ys == c).sum(), 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
    return float((ys == ps).mean()), float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--preproc", default="clahe", choices=PREPROC_MODES)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    out = Path(args.out or f"runs/cls_{args.preproc}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  preproc={args.preproc}  out={out}")

    tl = DataLoader(TumorDataset(args.data, "train", args.preproc, True), args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    vl = DataLoader(TumorDataset(args.data, "val", args.preproc, False), args.bs, num_workers=args.workers)

    model = size_classifier(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for img, _, _, y, _ in tl:
            img, y = img.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(img), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        acc, f1 = evaluate(model, vl, device)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_acc": acc, "val_macro_f1": f1})
        flag = ""
        if f1 > best:
            best = f1
            torch.save({"model": model.state_dict(), "preproc": args.preproc}, out / "best.pt")
            flag = "  *saved*"
        print(f"ep {ep:3d}  loss {np.mean(losses):.4f}  val_acc {acc:.3f}  val_F1 {f1:.3f}  ({time.time() - t0:.0f}s){flag}")
        with open(out / "history.json", "w") as f:
            json.dump({"args": vars(args), "best_val_macro_f1": best, "history": hist}, f, indent=2)


if __name__ == "__main__":
    main()
