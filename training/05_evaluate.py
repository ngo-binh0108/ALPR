"""Step 5 - end-to-end evaluation of the exported ONNX models (same pipeline as the C++ app).

Reports detection recall, end-to-end plate accuracy, colour accuracy and latency.

Example
    python 05_evaluate.py --images ../datasets/raw/ccpd2019/valn --limit 500
    python 05_evaluate.py --images ../datasets/alpr/det/images/val --save-vis vis_out
    python 05_evaluate.py --labelme D:/site_val                           # your own LabelMe-labelled photos
    python 05_evaluate.py --labelme D:/site_val --det runs/det/plate/weights/best.pt --rec runs/rec_more/best.pt
"""
import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpr import ccpd  # noqa: E402
from alpr.evaluation import evaluate, format_metrics  # noqa: E402
from alpr.pipeline import AlprPipeline, TorchPipeline  # noqa: E402
from alpr.rec_data import imread  # noqa: E402

HERE = Path(__file__).resolve().parent


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def draw(img, plates):
    for p in plates:
        x1, y1, x2, y2 = map(int, p.box)
        cv2.polylines(img, [p.corners.astype(np.int32)], True, (0, 255, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(img, f"{p.conf:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(HERE.parent / "models"))
    ap.add_argument("--images", default="", help="folder of CCPD-named images (labels from file name)")
    ap.add_argument("--labelme", default="", help="folder of LabelMe-labelled photos (ground truth from .json)")
    ap.add_argument("--det", default="", help="PyTorch detector checkpoint instead of ONNX (with --rec)")
    ap.add_argument("--rec", default="", help="PyTorch recogniser checkpoint instead of ONNX (with --det)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save-vis", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=str(HERE.parent / "datasets/alpr/used_files.txt"),
                    help="list of images used in training/validation (skipped)")
    args = ap.parse_args()

    if args.det and args.rec:
        pipe = TorchPipeline(Path(args.det), Path(args.rec))
    else:
        pipe = AlprPipeline(Path(args.models), threads=args.threads)

    if args.labelme:   # own photos: ground truth from LabelMe (labels with "?" count for detection only)
        import importlib.util
        spec = importlib.util.spec_from_file_location("prepare_data", HERE / "01_prepare_data.py")
        prep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prep)
        samples = list(prep.load_labelme(Path(args.labelme)))
        random.Random(args.seed).shuffle(samples)
        if args.limit:
            samples = samples[: args.limit]
        m = evaluate(pipe, samples, conf_th=args.conf)
        print(format_metrics(m))
        for path, gt, got in m["errors"][:15]:
            print(f"  miss: {Path(path).name[:40]:40s} {gt:10s} -> {got}")
        return
    if not args.images:
        ap.error("give --images (CCPD) or --labelme")
    labels = ccpd.scan(Path(args.images))
    if args.exclude and Path(args.exclude).exists():
        used = set(Path(args.exclude).read_text(encoding="utf-8").splitlines())
        before = len(labels)
        labels = [l for l in labels if str(l.path.resolve()) not in used]
        print(f"excluded {before - len(labels)} images that were used for training/validation")
    # CCPD names start with the plate-area ratio, so sorted order = smallest plates first:
    # take a random (seeded) sample instead of the first N files
    random.Random(args.seed).shuffle(labels)
    if args.limit:
        labels = labels[: args.limit]
    vis_dir = Path(args.save_vis) if args.save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    n = found = exact = color_ok = 0
    det_ms, rec_ms, total_ms = [], [], []
    wrong = []
    for lab in labels:
        img = imread(lab.path)
        t = time.perf_counter()
        plates = pipe(img, conf_th=args.conf)
        total_ms.append((time.perf_counter() - t) * 1000)
        det_ms.append(pipe.timing["det_ms"])
        if plates:
            rec_ms.append(pipe.timing["rec_ms"])
        n += 1
        best = max(plates, key=lambda p: iou(p.box, lab.bbox), default=None)
        if best is not None and iou(best.box, lab.bbox) > 0.5:
            found += 1
            exact += best.text == lab.text
            color_ok += best.color == lab.color
            if best.text != lab.text:
                wrong.append((lab.path.name[:40], lab.text, best.text, round(best.conf, 3)))
        if vis_dir:
            cv2.imencode(".jpg", draw(img.copy(), plates))[1].tofile(str(vis_dir / f"{n:05d}.jpg"))

    print(f"images               : {n}")
    print(f"detection recall     : {found / max(n, 1):.4f}  (IoU > 0.5)")
    print(f"end-to-end plate acc : {exact / max(n, 1):.4f}  (exact string match)")
    print(f"recognition acc      : {exact / max(found, 1):.4f}  (on detected plates)")
    print(f"colour acc           : {color_ok / max(found, 1):.4f}")
    print(f"latency  det {np.mean(det_ms):.1f} ms | rec {np.mean(rec_ms) if rec_ms else 0:.1f} ms | "
          f"total {np.mean(total_ms):.1f} ms/frame  ({args.threads} threads)")
    for w in wrong[:15]:
        print("  miss:", w)


if __name__ == "__main__":
    main()
