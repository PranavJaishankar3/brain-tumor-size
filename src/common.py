"""Shared preprocessing, size-category, and device helpers."""
import json
from pathlib import Path

import cv2
import numpy as np

SIZE_NAMES = ["small", "medium", "large"]
TUMOR_TYPES = {1: "meningioma", 2: "glioma", 3: "pituitary"}
PREPROC_MODES = ["raw", "zscore", "clahe"]


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def foreground_mask(img: np.ndarray) -> np.ndarray:
    """Head/brain region via Otsu + morphology + largest component + hole fill.
    Used as the denominator for relative tumor size and for intensity normalization."""
    x = img.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    u8 = (x * 255).astype(np.uint8)
    _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((th > 0).astype(np.uint8))
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        th = (lab == k).astype(np.uint8)
    else:
        th = (th > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(th)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    return filled


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(img: np.ndarray, fg: np.ndarray, mode: str) -> np.ndarray:
    """Return float32 image in roughly [0, 1].
    raw    : global min-max scaling
    zscore : z-score using only foreground pixels (removes scanner intensity offsets)
    clahe  : zscore + contrast-limited adaptive histogram equalization (sharpens tumor edges)
    """
    x = img.astype(np.float32)
    if mode == "raw":
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    m = fg > 0
    if m.sum() < 10:
        m = np.ones_like(m, dtype=bool)
    mu, sd = x[m].mean(), x[m].std() + 1e-8
    z = np.clip((x - mu) / sd, -3, 3)
    z = (z + 3) / 6.0
    z[~m] = 0.0
    if mode == "zscore":
        return z.astype(np.float32)
    if mode == "clahe":
        u8 = (z * 255).astype(np.uint8)
        out = _CLAHE.apply(u8).astype(np.float32) / 255.0
        out[~m] = 0.0
        return out
    raise ValueError(f"unknown preprocessing mode: {mode}")


def area_fraction(mask: np.ndarray, fg: np.ndarray) -> float:
    return float((mask > 0).sum()) / float(max((fg > 0).sum(), 1))


def size_class(frac: float, thresholds) -> int:
    t1, t2 = thresholds
    return 0 if frac < t1 else (1 if frac < t2 else 2)


def load_meta(data_dir="data/processed"):
    with open(Path(data_dir) / "meta.json") as f:
        return json.load(f)
