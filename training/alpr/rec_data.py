"""Datasets for the recogniser.

* labels file format (TSV, UTF-8): ``relative/path.jpg<TAB>text<TAB>colour``
* real samples are stored as a loose crop around the plate plus the 4 corners, so every
  epoch sees a *different* rectification (corner jitter = detector noise augmentation).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .charset import COLOR2IDX, encode
from .geometry import REC_H, REC_W, rectify, to_tensor_layout
from .synth import jitter_corners, photometric


def read_tsv(path: Path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel, text, color = line.split("\t")[:3]
            rows.append((rel, text, color))
    return rows


def imread(path) -> np.ndarray:
    """cv2.imread that also handles non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class RectifiedDataset(Dataset):
    """Already-rectified 168x48 images (synthetic set)."""

    def __init__(self, root: Path, labels: str = "labels.tsv", augment: bool = False):
        self.root = Path(root)
        self.rows = read_tsv(self.root / labels)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel, text, color = self.rows[i]
        img = imread(self.root / rel)
        if img.shape[:2] != (REC_H, REC_W):
            img = cv2.resize(img, (REC_W, REC_H))
        if self.augment:
            img = photometric(img, random.Random(), strength=0.3)
        return torch.from_numpy(to_tensor_layout(img)), text, COLOR2IDX[color]


class RealCropDataset(Dataset):
    """Loose crops + corners produced by 01_prepare_data.py."""

    def __init__(self, root: Path, labels: str, augment: bool = True):
        self.root = Path(root)
        self.items = [json.loads(l) for l in (self.root / labels).read_text(encoding="utf-8").splitlines() if l]
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = imread(self.root / it["image"])
        corners = np.array(it["corners"], np.float32)
        if self.augment:
            rng = random.Random()
            corners = jitter_corners(corners, rng, amount=0.05)
            img = photometric(img, rng, strength=0.6)
        plate = rectify(img, corners, REC_W, REC_H)
        return torch.from_numpy(to_tensor_layout(plate)), it["text"], COLOR2IDX[it["color"]]


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    targets = torch.tensor([i for t in texts for i in encode(t)], dtype=torch.long)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    colors = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, targets, lengths, colors, texts
