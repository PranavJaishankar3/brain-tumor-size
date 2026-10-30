import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import preprocess


def read_index(data_dir, split):
    with open(Path(data_dir) / "index.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class TumorDataset(Dataset):
    """Returns (image[1,H,W], tumor_mask[1,H,W], fg_mask[1,H,W], size_class, area_frac)."""

    def __init__(self, data_dir, split, preproc="clahe", augment=False):
        self.dir = Path(data_dir) / "samples"
        self.rows = read_index(data_dir, split)
        self.preproc = preproc
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _aug(self, img, mask, fg):
        h, w = img.shape
        if np.random.rand() < 0.5:
            img, mask, fg = img[:, ::-1], mask[:, ::-1], fg[:, ::-1]
        ang = np.random.uniform(-15, 15)
        sc = np.random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[:, 2] += np.random.uniform(-0.05, 0.05, 2) * [w, h]
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (w, h), flags=cv2.INTER_NEAREST)
        fg = cv2.warpAffine(np.ascontiguousarray(fg), M, (w, h), flags=cv2.INTER_NEAREST)
        # intensity jitter (contrast/brightness/gamma)
        g = np.random.uniform(0.8, 1.25)
        img = np.clip(img, 0, 1) ** g
        img = img * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
        return np.clip(img, 0, 1).astype(np.float32), mask, fg

    def __getitem__(self, i):
        r = self.rows[i]
        d = np.load(self.dir / f"{r['id']}.npz")
        img = preprocess(d["image"], d["fg"], self.preproc)
        mask, fg = d["mask"].astype(np.uint8), d["fg"].astype(np.uint8)
        if self.augment:
            img, mask, fg = self._aug(img, mask, fg)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()[None]
        return t(img), t(mask), t(fg), int(r["size_class"]), float(r["area_frac"])
