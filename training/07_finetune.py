"""Step 7 - fine-tune the current models on your own labelled photos, in one command.

    python 07_finetune.py --train-labelme D:/site_photos

What it does
  1. builds a dataset from your LabelMe folder(s) (e.g. made with 00_pre_label.py + LabelMe review)
     plus a small "replay" sample of the original CCPD data, so the model keeps what it knew
  2. measures the CURRENT models on your validation photos        (before)
     and on a fixed sample of the ORIGINAL data it was trained for  (to catch forgetting)
  3. fine-tunes the detector   from runs/det/plate/weights/best.pt  (low LR, no warm-up)
  4. fine-tunes the recogniser from runs/rec_more/best.pt           (real crops + synthetic replay,
     every plate also cut with the new detector's own corners)
  5. measures the NEW models on the same two sets                  (after)
  6. exports ONNX to ../models_<name>; with --install (and only if it is not worse on your photos and
     loses at most --max-forget points on the original data) it replaces ../models after backing it up
     to ../models_backup_<time>

Examples
    python 07_finetune.py --train-labelme D:/site_photos                        # everything, CPU
    python 07_finetune.py --train-labelme D:/site/train --val-labelme D:/site/val --device 0 --install
    python 07_finetune.py --train-labelme D:/site_photos --what rec             # recogniser only
    python 07_finetune.py --train-labelme D:/tiny_set --freeze 10 --epochs-det 15  # < 100 photos
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from alpr.evaluation import evaluate, format_metrics  # noqa: E402
from alpr.pipeline import TorchPipeline  # noqa: E402


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_data", HERE / "01_prepare_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode})")


def add_detector_crops(det_weights, items, real_dir: Path, imgsz, device, conf=0.25):
    """Detector-in-the-loop: cut every labelled plate a second time using the corners the detector
    predicts (not the hand-labelled ones) and add it to the recogniser's training set with the true
    text. The recogniser then learns exactly the corner errors it will see at run time."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from alpr.charset import is_valid
    from alpr.evaluation import _box, _iou
    from alpr.rec_data import imread

    model = YOLO(str(det_weights))
    out_dir = real_dir / "train_det"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines, n = [], 0
    for i, (path, plates) in enumerate(items):
        img = imread(path)
        if img is None:
            continue
        r = model.predict(img, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy()
        h, w = img.shape[:2]
        for j, (corners, text, color) in enumerate(plates):
            if not is_valid(text):
                continue                     # "?" = not reviewed yet
            ious = [_iou(_box(corners), b) for b in boxes]
            k = int(np.argmax(ious))
            if ious[k] < 0.5:
                continue
            c = kpts[k].astype(np.float32)
            x1, y1 = c.min(0)
            x2, y2 = c.max(0)
            pw, ph = x2 - x1, y2 - y1
            cx1, cy1 = int(max(0, x1 - 0.3 * pw)), int(max(0, y1 - 0.6 * ph))
            cx2, cy2 = int(min(w, x2 + 0.3 * pw)), int(min(h, y2 + 0.6 * ph))
            name = f"det_{i:06d}_{j}.jpg"
            cv2.imencode(".jpg", img[cy1:cy2, cx1:cx2], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(out_dir / name))
            lines.append(json.dumps({"image": f"train_det/{name}", "corners": (c - [cx1, cy1]).round(2).tolist(),
                                     "text": text, "color": color}, ensure_ascii=False))
            n += 1
    if lines:
        with open(real_dir / "train.jsonl", "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(f"  added {n} recogniser crops cut with the detector's own corners")
    return n


def sample_original(prep, srcs, rng, skip):
    """Fixed sample of the ORIGINAL (CCPD) data, never used for training, to measure forgetting."""
    from alpr import ccpd

    used = set(skip)
    for d in ("alpr", "alpr_more"):
        f = ROOT / "datasets" / d / "used_files.txt"
        if f.exists():
            used |= set(f.read_text(encoding="utf-8").splitlines())
    items = []
    for a in srcs:
        folder, n = prep.src_count(a)
        labs = [l for l in ccpd.scan(folder) if str(l.path.resolve()) not in used]
        rng.shuffle(labs)
        labs = labs[:n] if n else labs
        items += [(l.path, [(l.corners, l.text, l.color)]) for l in labs]
        print(f"  original data check: {folder}: {len(labs)} images")
    return items


def measure(tag, det, rec, sets, imgsz, device):
    pipe = TorchPipeline(Path(det), Path(rec), imgsz=imgsz, device="cpu" if device == "cpu" else "cuda")
    out = {}
    for name, samples in sets.items():
        if samples:
            out[name] = evaluate(pipe, samples)
            print(f"[{tag}] {name:13s} {format_metrics(out[name])}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-labelme", action="append", required=True, help="LabelMe folder[:count] with your photos")
    ap.add_argument("--val-labelme", action="append", help="separate validation folder (default: 15 %% of your photos)")
    ap.add_argument("--replay", action="append",
                    help="CCPD folder:count mixed in against forgetting (default: 300 blue + 100 green)")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--original", action="append",
                    help="CCPD folder:count measured before/after to catch forgetting "
                         "(default: 200 blue + 100 green validation images)")
    ap.add_argument("--no-original", action="store_true")
    ap.add_argument("--max-forget", type=float, default=2.0,
                    help="--install refuses if plate accuracy on the original data drops by more points than this")
    ap.add_argument("--base-det", default=str(HERE / "runs/det/plate/weights/best.pt"))
    ap.add_argument("--base-rec", default=str(HERE / "runs/rec_more/best.pt"))
    ap.add_argument("--what", choices=["both", "det", "rec"], default="both")
    ap.add_argument("--epochs-det", type=int, default=20)
    ap.add_argument("--epochs-rec", type=int, default=15)
    ap.add_argument("--steps-rec", type=int, default=80, help="recogniser iterations per epoch")
    ap.add_argument("--lr-det", type=float, default=5e-4, help="low LR: the detector is already trained")
    ap.add_argument("--lr-rec", type=float, default=1e-3, help="(stage B uses half of it)")
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N detector layers (10 = backbone)")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu | 0 (NVIDIA GPU)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--name", default=time.strftime("ft_%Y%m%d_%H%M"))
    ap.add_argument("--install", action="store_true", help="replace ../models if the result is not worse")
    ap.add_argument("--no-det-crops", action="store_true",
                    help="do not add recogniser crops cut with the detector's predicted corners")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    py = sys.executable
    prep = load_prepare_module()
    rng = random.Random(args.seed)
    ds = ROOT / "datasets" / "finetune" / args.name
    runs = HERE / "runs" / "finetune"
    if ds.exists():
        shutil.rmtree(ds)

    # ------------------------------------------------------------------ 1. dataset
    print("== 1/6 dataset")
    site = prep.gather(None, args.train_labelme, rng)
    if not site:
        raise SystemExit("no LabelMe annotations found (json next to the images, 4-point polygons)")
    val = prep.gather(None, args.val_labelme, rng) if args.val_labelme else []
    if not val:                                   # validation = your photos only, never replay data
        rng.shuffle(site)
        k = max(1, int(len(site) * 0.15))
        val, site = site[:k], site[k:]
    replay_src = [] if args.no_replay else (args.replay or [
        s for s in [f"{ROOT / 'datasets/raw/ccpd2019/trainn'}:300", f"{ROOT / 'datasets/raw/CCPD2020/ccpd_green/train'}:100"]
        if Path(s.rpartition(":")[0]).exists()])
    replay = prep.gather(replay_src, None, rng) if replay_src else []
    print(f"  your photos: {len(site)} train / {len(val)} val   |   replay: {len(replay)}")
    prep.build_dataset(site + replay, val, ds)
    original_src = [] if args.no_original else (args.original or [
        s for s in [f"{ROOT / 'datasets/raw/ccpd2019/valn'}:200", f"{ROOT / 'datasets/raw/CCPD2020/ccpd_green/val'}:100"]
        if Path(s.rpartition(":")[0]).exists()])
    original = sample_original(prep, original_src, random.Random(args.seed + 1),   # own RNG: split stays the same
                               {str(Path(p).resolve()) for p, _ in replay}) if original_src else []
    sets = {"your photos": [(p, plates) for p, plates in val], "original data": original}

    det_new, rec_new = Path(args.base_det), Path(args.base_rec)

    # ------------------------------------------------------------------ 2. before
    print("\n== 2/6 current models on your validation photos (+ original data)")
    before = measure("before", args.base_det, args.base_rec, sets, args.imgsz, args.device)

    # ------------------------------------------------------------------ 3. detector
    if args.what in ("both", "det"):
        print("\n== 3/6 fine-tune detector")
        cmd = [py, HERE / "02_train_detector.py", "--data", ds / "det/data.yaml", "--model", args.base_det,
               "--epochs", args.epochs_det, "--imgsz", args.imgsz, "--batch", args.batch, "--device", args.device,
               "--project", runs, "--name", f"{args.name}_det", "--lr0", args.lr_det, "--warmup", 0,   # no warm-up: its high bias LR would wreck a trained head
               "--close-mosaic", max(1, args.epochs_det // 3), "--patience", max(5, args.epochs_det // 2)]
        if args.freeze:
            cmd += ["--freeze", args.freeze]
        if args.threads:
            cmd += ["--threads", args.threads]
        run(cmd)
        det_new = runs / f"{args.name}_det" / "weights" / "best.pt"

    # ------------------------------------------------------------------ 4. recogniser
    if args.what in ("both", "rec"):
        print("\n== 4/6 fine-tune recogniser")
        if not args.no_det_crops:
            add_detector_crops(det_new, site + replay, ds / "rec/real", args.imgsz,
                               "cpu" if args.device == "cpu" else args.device)
        cmd = [py, HERE / "03_train_recognizer.py", "--real", ds / "rec/real",
               "--synth-dir", ROOT / "datasets/alpr/rec/synth", "--synth", 40000, "--skip-a",
               "--init", args.base_rec, "--epochs-b", args.epochs_rec, "--steps-b", args.steps_rec,
               "--lr", args.lr_rec, "--out", runs / f"{args.name}_rec",
               "--device", "cpu" if args.device == "cpu" else "cuda"]
        if args.threads:
            cmd += ["--threads", args.threads]
        run(cmd)
        rec_new = runs / f"{args.name}_rec" / "best.pt"

    # ------------------------------------------------------------------ 5. after
    print("\n== 5/6 fine-tuned models on the same photos")
    after = measure("after ", det_new, rec_new, sets, args.imgsz, args.device)

    # ------------------------------------------------------------------ 6. export (+ install)
    print("\n== 6/6 export")
    out_models = ROOT / f"models_{args.name}"
    run([py, HERE / "04_export_onnx.py", "--det", det_new, "--rec", rec_new, "--imgsz", args.imgsz, "--out", out_models])
    report = {"name": args.name, "before": before, "after": after, "detector": str(det_new),
              "recogniser": str(rec_new), "models": str(out_models),
              "train_photos": len(site), "val_photos": len(val), "replay": len(replay)}
    s0, s1 = before["your photos"], after["your photos"]
    better = (s1["plate_accuracy"], s1["detection_recall"]) >= (s0["plate_accuracy"], s0["detection_recall"])
    forget = 0.0
    if "original data" in before:
        forget = 100 * (before["original data"]["plate_accuracy"] - after["original data"]["plate_accuracy"])
        report["forgetting_points"] = round(forget, 2)
    (out_models / "finetune_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    for name in before:
        print(f"{name:13s} before : {format_metrics(before[name])}")
        print(f"{'':13s} after  : {format_metrics(after[name])}")
    print(f"models : {out_models}")
    if forget > args.max_forget:
        print(f"WARNING: {forget:.1f} points lost on the original data (> --max-forget {args.max_forget}): "
              "the models forget plates that are not in your photos. Use more --replay or fewer epochs")
    if args.install:
        if not better:
            print("NOT installed: the fine-tuned models are worse on your validation photos than the current ones")
        elif forget > args.max_forget:
            print("NOT installed: too much forgetting (raise --max-forget if your cameras never see those plates)")
        else:
            backup = ROOT / time.strftime("models_backup_%Y%m%d_%H%M%S")
            shutil.copytree(ROOT / "models", backup)
            for f in out_models.iterdir():
                if f.suffix in {".onnx", ".json"} and f.name != "finetune_report.json":
                    shutil.copy2(f, ROOT / "models" / f.name)
            print(f"installed into {ROOT / 'models'}  (previous models backed up to {backup})")
    else:
        print("to use them: copy the files of that folder into ../models (or run again with --install)")
    for path, gt, got in s1["errors"][:10]:
        print(f"  still wrong: {Path(path).name[:40]:40s} {gt:10s} -> {got}")


if __name__ == "__main__":
    main()
