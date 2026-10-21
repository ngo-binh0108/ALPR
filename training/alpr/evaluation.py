"""Accuracy of a pipeline (ONNX or PyTorch) against labelled images: CCPD file names or LabelMe JSON."""
from __future__ import annotations

import time

import numpy as np

from .rec_data import imread


def _box(corners):
    c = np.asarray(corners, np.float32)
    return (*c.min(0), *c.max(0))


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def evaluate(pipe, samples, conf_th: float = 0.4, iou_th: float = 0.5, keep_errors: int = 20) -> dict:
    """samples: [(image_path, [(corners, text, colour), ...]), ...].

    A ground-truth plate counts as *found* when a detection overlaps it with IoU > iou_th, and as
    *read* when that detection's text is identical. Labels starting with "?" (not yet reviewed)
    count for detection only.
    """
    n_gt = found = n_text = read = color_ok = 0
    times, errors = [], []
    for path, plates in samples:
        img = imread(path)
        if img is None:
            continue
        t = time.perf_counter()
        preds = pipe(img, conf_th=conf_th)
        times.append((time.perf_counter() - t) * 1000)
        used = set()
        for corners, text, color in plates:
            n_gt += 1
            gb = _box(corners)
            best, best_iou = None, iou_th
            for i, p in enumerate(preds):
                o = _iou(gb, p.box)
                if i not in used and o > best_iou:
                    best, best_iou = i, o
            checkable = bool(text) and not text.startswith("?")
            n_text += checkable
            if best is None:
                if checkable and len(errors) < keep_errors:
                    errors.append((str(path), text, "(not found)"))
                continue
            used.add(best)
            found += 1
            p = preds[best]
            if checkable:
                if p.text == text:
                    read += 1
                elif len(errors) < keep_errors:
                    errors.append((str(path), text, p.text))
                color_ok += p.color == color
    return {
        "images": len(times),
        "plates": n_gt,
        "detection_recall": found / max(n_gt, 1),
        "plate_accuracy": read / max(n_text, 1),      # end-to-end exact read of reviewed plates
        "colour_accuracy": color_ok / max(n_text, 1),
        "reviewed_plates": n_text,
        "ms_per_image": float(np.mean(times)) if times else 0.0,
        "errors": errors,
    }


def format_metrics(m: dict) -> str:
    return (f"plates found {m['detection_recall'] * 100:5.1f} %  |  read exactly {m['plate_accuracy'] * 100:5.1f} %  |  "
            f"colour {m['colour_accuracy'] * 100:5.1f} %  ({m['images']} images, {m['plates']} plates, "
            f"{m['reviewed_plates']} with reviewed text, {m['ms_per_image']:.0f} ms/image)")
