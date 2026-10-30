"""Train the U-Net to separate the tumor region from surrounding tissue."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import PREPROC_MODES, get_device
from dataset import TumorDataset
from models import UNet, dice_bce_loss


@torch.no_grad()
def val_dice(model, loader, device):
    model.eval()
    scores = []
    for img, mask, fg, *_ in loader:
        img, mask = img.to(device), mask.to(device)
        pred = (torch.sigmoid(model(img)) > 0.5).float()
        inter = (pred * mask).sum((1, 2, 3))
        d = (2 * inter + 1) / (pred.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)
        scores.extend(d.cpu().tolist())
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--preproc", default="clahe", choices=PREPROC_MODES)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32, help="U-Net width; use 16 on CPU")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    out = Path(args.out or f"runs/seg_{args.preproc}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  preproc={args.preproc}  out={out}")

    tr = TumorDataset(args.data, "train", args.preproc, augment=True)
    va = TumorDataset(args.data, "val", args.preproc, augment=False)
    pin = device.type == "cuda"
    tl = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers, pin_memory=pin, drop_last=True)
    vl = DataLoader(va, args.bs, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = UNet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(tl))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for img, mask, *_ in tl:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = dice_bce_loss(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        vd = val_dice(model, vl, device)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_dice": vd})
        flag = ""
        if vd > best:
            best = vd
            torch.save({"model": model.state_dict(), "preproc": args.preproc, "base": args.base}, out / "best.pt")
            flag = "  *saved*"
        print(f"ep {ep:3d}  loss {np.mean(losses):.4f}  val_dice {vd:.4f}  ({time.time() - t0:.0f}s){flag}")
        with open(out / "history.json", "w") as f:
            json.dump({"args": vars(args), "best_val_dice": best, "history": hist}, f, indent=2)
    print(f"Best val Dice: {best:.4f}  ->  {out / 'best.pt'}")


if __name__ == "__main__":
    main()
