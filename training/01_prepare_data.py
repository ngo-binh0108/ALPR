"""Step 1 - build a *small* training set from CCPD and/or your own LabelMe annotations.

Outputs (default datasets/alpr):
    det/images/{train,val}/*.jpg  det/labels/{train,val}/*.txt  det/data.yaml   -> YOLO pose (4 corners)
    rec/real/{train,val}/*.jpg    rec/real/{train,val}.jsonl                     -> recogniser crops

Examples
    python 01_prepare_data.py ^
        --train ../datasets/raw/ccpd2019/trainn:600 --train ../datasets/raw/CCPD2020/ccpd_green/train:200 ^
        --val   ../datasets/raw/ccpd2019/valn:150   --val   ../datasets/raw/CCPD2020/ccpd_green/val:50

    # your own camera images annotated in LabelMe: one 4-point polygon per plate,
    # label = plate text, optionally "text|colour"  (e.g. 粤B12345|blue)
    python 01_prepare_data.py --train-labelme D:/my_site/train:0 --val-labelme D:/my_site/val:0
    (COUNT 0 = use every file)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpr import ccpd  # noqa: E402
from alpr.charset import COLORS, is_valid  # noqa: E402
from alpr.geometry import order_corners, rectify  # noqa: E402
from alpr.rec_data import imread  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def src_count(arg: str):
    path, _, n = arg.rpartition(":")
    if not path or not n.isdigit():   # a Windows drive letter such as "D:" is not a count
        return Path(arg), 0
    return Path(path), int(n)


def load_labelme(folder: Path):
    """Yield (image_path, [(corners, text, colour), ...]) from LabelMe json files."""
    for js in sorted(folder.rglob("*.json")):
        d = json.loads(js.read_text(encoding="utf-8"))
        img = js.parent / d["imagePath"]
        plates = []
        for sh in d.get("shapes", []):
            pts = np.array(sh["points"], np.float32)
            if len(pts) != 4:
                continue
            text, _, color = sh["label"].partition("|")
            text = text.strip()
            color = color.strip() or ("green" if len(text) == 8 else "blue")
            if color not in COLORS:
                print(f"  skip {js.name}: unknown colour {color}")
                continue
            plates.append((order_corners(pts), text, color))
        if plates and img.exists():
            yield img, plates


def gather(ccpd_args, labelme_args, rng):
    items = []   # (image_path, [(corners, text, colour)])
    for a in ccpd_args or []:
        folder, n = src_count(a)
        labs = ccpd.scan(folder)
        rng.shuffle(labs)
        labs = labs[:n] if n else labs
        items += [(l.path, [(l.corners, l.text, l.color)]) for l in labs]
        print(f"  {folder}: {len(labs)} images")
    for a in labelme_args or []:
        folder, n = src_count(a)
        lm = list(load_labelme(folder))
        rng.shuffle(lm)
        lm = lm[:n] if n else lm
        items += lm
        print(f"  {folder} (labelme): {len(lm)} images")
    return items


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(items, split: str, out: Path):
    img_dir = out / "det" / "images" / split
    lab_dir = out / "det" / "labels" / split
    crop_dir = out / "rec" / "real" / split
    for d in (img_dir, lab_dir, crop_dir):
        d.mkdir(parents=True, exist_ok=True)
    jsonl = []
    n_plates = 0
    for i, (img_path, plates) in enumerate(items):
        img = imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        name = f"{split}_{i:06d}"
        link_or_copy(Path(img_path), img_dir / f"{name}{Path(img_path).suffix.lower()}")
        lines = []
        for j, (c, text, color) in enumerate(plates):
            c = np.clip(c, 0, [w - 1, h - 1])
            x1, y1 = c.min(0)
            x2, y2 = c.max(0)
            kp = " ".join(f"{x / w:.6f} {y / h:.6f} 2" for x, y in c)
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {kp}")
            if not is_valid(text):
                continue   # still a detector sample, but not a recogniser sample
            # loose crop around the plate so training can re-rectify with jitter
            pw, ph = x2 - x1, y2 - y1
            cx1, cy1 = int(max(0, x1 - 0.3 * pw)), int(max(0, y1 - 0.6 * ph))
            cx2, cy2 = int(min(w, x2 + 0.3 * pw)), int(min(h, y2 + 0.6 * ph))
            crop = img[cy1:cy2, cx1:cx2]
            cc = (c - [cx1, cy1]).round(2).tolist()
            crop_name = f"{name}_{j}.jpg"
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(crop_dir / crop_name))
            jsonl.append(json.dumps({"image": f"{split}/{crop_name}", "corners": cc, "text": text, "color": color},
                                    ensure_ascii=False))
            n_plates += 1
        (lab_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "rec" / "real" / f"{split}.jsonl").write_text("\n".join(jsonl) + "\n", encoding="utf-8")
    print(f"  {split}: {len(items)} images, {n_plates} recogniser crops")


def preview(out: Path, n=16):
    rows = [json.loads(l) for l in (out / "rec/real/train.jsonl").read_text(encoding="utf-8").splitlines()[:n]]
    tiles = [rectify(imread(out / "rec/real" / r["image"]), np.array(r["corners"], np.float32)) for r in rows]
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imencode(".jpg", grid)[1].tofile(str(out / "rec_preview.jpg"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="append", help="CCPD folder[:count]")
    ap.add_argument("--val", action="append", help="CCPD folder[:count]")
    ap.add_argument("--train-labelme", action="append", help="LabelMe folder[:count]")
    ap.add_argument("--val-labelme", action="append", help="LabelMe folder[:count]")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "alpr"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.train or args.train_labelme):
        ap.error("give at least one --train or --train-labelme source")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(args.seed)
    print("train sources:")
    train = gather(args.train, args.train_labelme, rng)
    print("val sources:")
    val = gather(args.val, args.val_labelme, rng)
    if not val:  # hold out 15 % when no explicit validation set is given
        rng.shuffle(train)
        k = max(1, int(len(train) * 0.15))
        val, train = train[:k], train[k:]
    build_dataset(train, val, out)


def build_dataset(train, val, out: Path):
    """Write detector + recogniser datasets for the given (image, plates) items (also used by 07_finetune.py)."""
    write_split(train, "train", out)
    write_split(val, "val", out)
    # every source image used for training/validation -> 05_evaluate.py --exclude keeps tests honest
    used = sorted(str(Path(p).resolve()) for p, _ in train + val)
    (out / "used_files.txt").write_text("\n".join(used) + "\n", encoding="utf-8")

    (out / "det" / "data.yaml").write_text(
        f"path: {(out / 'det').as_posix()}\n"
        "train: images/train\nval: images/val\n"
        "kpt_shape: [4, 3]\n"
        "flip_idx: [1, 0, 3, 2]\n"
        "names:\n  0: plate\n", encoding="utf-8")
    preview(out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
