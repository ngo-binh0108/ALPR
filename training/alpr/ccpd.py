"""Parser for CCPD (Chinese City Parking Dataset) file names.

A CCPD file name encodes all annotations, e.g.
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
fields (split by '-'):
    0 area ratio, 1 tilt, 2 bbox "x1&y1_x2&y2",
    3 four vertices "RB_LB_LT_RT" (x&y each), 4 plate char indices,
    5 brightness, 6 blurriness
CCPD2019 plates are blue (7 chars); CCPD2020 (ccpd_green) plates are green (8 chars).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROV = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
         "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
_ALPHA = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
          "U", "V", "W", "X", "Y", "Z", "O"]
_ADS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]


@dataclass
class CcpdLabel:
    path: Path
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2
    corners: np.ndarray                  # (4, 2) float32, order TL, TR, BR, BL
    text: str
    color: str                           # "blue" | "green"


def parse(path: Path) -> CcpdLabel | None:
    """Return the label encoded in a CCPD file name, or None if it is unusable."""
    parts = path.stem.split("-")
    if len(parts) < 7:
        return None  # e.g. ccpd_np (no plate) images
    try:
        (x1, y1), (x2, y2) = [tuple(map(int, p.split("&"))) for p in parts[2].split("_")]
        rb, lb, lt, rt = [tuple(map(float, p.split("&"))) for p in parts[3].split("_")]
        idx = list(map(int, parts[4].split("_")))
    except ValueError:
        return None
    text = _PROV[idx[0]] + _ALPHA[idx[1]] + "".join(_ADS[i] for i in idx[2:])
    if "O" in text:
        return None  # "O" marks an unreadable character in CCPD
    corners = np.array([lt, rt, rb, lb], dtype=np.float32)
    color = "green" if len(text) == 8 else "blue"
    return CcpdLabel(path, (x1, y1, x2, y2), corners, text, color)


def scan(folder: Path) -> list[CcpdLabel]:
    labels = []
    for p in sorted(Path(folder).rglob("*.jpg")):
        lab = parse(p)
        if lab is not None:
            labels.append(lab)
    return labels
