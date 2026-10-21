"""Test the trained models on your own data - image, folder, video, USB camera or RTSP stream.

Uses the exported ONNX models (exactly the pipeline of ALPR Studio) or, with --det/--rec, the
PyTorch checkpoints straight from training (no export needed).

Examples
    python infer.py --source car.jpg --show                        # one image in a window
    python infer.py --source D:/photos --save out_vis --csv out.csv # folder -> annotated images + CSV
    python infer.py --source ../datasets/test/ccpd_test_clip.mp4 --save out.mp4 --show
    python infer.py --source 0 --show                               # USB camera 0
    python infer.py --source rtsp://user:pw@192.168.1.64:554/Streaming/Channels/101 --show --every 2
    python infer.py --source D:/photos --roi "0.2,0.3 0.8,0.3 0.9,1 0.1,1"   # only plates inside this polygon
    python infer.py --source D:/photos --det runs/det/plate/weights/best.pt --rec runs/rec_more/best.pt

Keys in the window: q / Esc = quit, space = pause, s = save the current frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpr.pipeline import AlprPipeline, TorchPipeline  # noqa: E402
from alpr.rec_data import imread  # noqa: E402

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLATE_RGB = {"blue": (80, 160, 255), "green": (90, 220, 110), "yellow": (250, 215, 40),
             "white": (240, 240, 240), "black": (160, 160, 160)}


# ----------------------------------------------------------------------------- sources
def open_source(src: str):
    """Yields (frame_bgr, name, timestamp_s). Handles image, folder, video, camera, stream."""
    p = Path(src)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        for i, f in enumerate(files):
            img = imread(f)
            if img is not None:
                yield img, str(f.relative_to(p)), float(i)
        return
    if p.is_file() and p.suffix.lower() in IMG_EXT:
        yield imread(p), p.name, 0.0
        return
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    live = src.isdigit() or "://" in src
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() if live else i / fps
        yield frame, f"frame_{i:06d}", t
        i += 1
    cap.release()


# ----------------------------------------------------------------------------- drawing
def load_font(size):
    for f in ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def draw(frame, plates, accepted, roi_px, fps_text, font):
    vis = frame.copy()
    if roi_px is not None:
        cv2.polylines(vis, [roi_px.astype(np.int32)], True, (110, 203, 108), 2, cv2.LINE_AA)
    for p, ok in zip(plates, accepted):
        c = (255, 205, 96) if ok else (0, 225, 252)          # BGR: accepted = accent blue, else amber
        cv2.polylines(vis, [p.corners.astype(np.int32)], True, c, 2, cv2.LINE_AA)
    img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for p, ok in zip(plates, accepted):
        x1, y1 = int(p.box[0]), int(p.box[1])
        label = f"{p.text}  {p.conf * 100:.0f}%" if p.text else "?"
        tw = d.textlength(label, font=font)
        th = font.size + 8
        y = y1 - th - 4 if y1 - th - 4 > 0 else int(p.box[3]) + 4
        bg = (96, 205, 255) if ok else (40, 40, 40)
        d.rounded_rectangle([x1, y, x1 + tw + 14, y + th], radius=5, fill=bg)
        d.text((x1 + 7, y + 3), label, font=font, fill=(0, 0, 0) if ok else PLATE_RGB.get(p.color, (255, 255, 255)))
    if fps_text:
        d.rounded_rectangle([8, 8, 16 + d.textlength(fps_text, font=font) + 8, 16 + font.size + 8], radius=5, fill=(32, 32, 32))
        d.text((16, 12), fps_text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------- main
def parse_roi(s: str, w: int, h: int):
    pts = [tuple(map(float, xy.split(","))) for xy in s.split()]
    if len(pts) < 3:
        raise SystemExit("--roi needs at least 3 points: 'x,y x,y x,y' (normalised 0..1)")
    return np.array([[x * w, y * h] for x, y in pts], np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image | folder | video | camera index | rtsp/http url")
    ap.add_argument("--models", default=str(HERE.parent / "models"), help="folder with the exported ONNX models")
    ap.add_argument("--det", default="", help="PyTorch detector checkpoint (use instead of ONNX)")
    ap.add_argument("--rec", default="", help="PyTorch recogniser checkpoint (use instead of ONNX)")
    ap.add_argument("--imgsz", type=int, default=512, help="detector input size for --det")
    ap.add_argument("--conf", type=float, default=0.4, help="detector threshold")
    ap.add_argument("--accept", type=float, default=0.8, help="min. recognition confidence for a valid read")
    ap.add_argument("--any-format", action="store_true", help="do not require a valid plate format")
    ap.add_argument("--roi", default="", help="normalised polygon 'x,y x,y x,y ...' - only plates inside")
    ap.add_argument("--every", type=int, default=1, help="analyse every N-th frame (video / stream)")
    ap.add_argument("--cooldown", type=float, default=10.0, help="seconds before the same plate is reported again")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="show a window with the results")
    ap.add_argument("--save", default="", help="folder for annotated images, or .mp4/.avi for an annotated video")
    ap.add_argument("--json", default="", help="write every read to this JSON Lines file")
    ap.add_argument("--csv", default="", help="write accepted plates (events) to this CSV")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.det or args.rec:
        if not (args.det and args.rec):
            ap.error("--det and --rec must be given together")
        pipe = TorchPipeline(Path(args.det), Path(args.rec), imgsz=args.imgsz)
        backend = f"PyTorch ({Path(args.det).name}, {Path(args.rec).name})"
    else:
        pipe = AlprPipeline(Path(args.models), threads=args.threads)
        backend = f"ONNX Runtime ({args.models})"
    plate_re = re.compile(pipe.meta["recognizer"].get("plate_regex", ".*"))
    print(f"backend: {backend}")

    font = load_font(22)
    json_f = open(args.json, "w", encoding="utf-8") if args.json else None
    csv_f = open(args.csv, "w", newline="", encoding="utf-8-sig") if args.csv else None
    csv_w = csv.writer(csv_f) if csv_f else None
    if csv_w:
        csv_w.writerow(["time_s", "source", "plate", "color", "confidence", "x1", "y1", "x2", "y2"])
    save = Path(args.save) if args.save else None
    video_out = None
    if save and save.suffix.lower() not in {".mp4", ".avi"}:
        save.mkdir(parents=True, exist_ok=True)

    last_seen: dict[str, float] = {}
    n_frames = n_reads = n_events = 0
    lat = []
    paused = False
    still = Path(args.source).is_dir() or Path(args.source).suffix.lower() in IMG_EXT
    t_start = time.perf_counter()
    for idx, (frame, name, ts) in enumerate(open_source(args.source)):
        if args.max_frames and idx >= args.max_frames:
            break
        if not still and idx % max(1, args.every):
            continue
        h, w = frame.shape[:2]
        roi_px = parse_roi(args.roi, w, h) if args.roi else None

        t0 = time.perf_counter()
        plates = pipe(frame, conf_th=args.conf)
        lat.append((time.perf_counter() - t0) * 1000)
        if roi_px is not None:
            plates = [p for p in plates if cv2.pointPolygonTest(
                roi_px, ((p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2), False) >= 0]
        accepted = [p.conf >= args.accept and (args.any_format or bool(plate_re.match(p.text))) for p in plates]
        n_frames += 1
        n_reads += len(plates)

        for p, ok in zip(plates, accepted):
            if not ok:
                continue
            new = still or ts - last_seen.get(p.text, -1e9) > args.cooldown
            last_seen[p.text] = ts
            if new:
                n_events += 1
                print(f"[{name}] {p.text:10s} {p.color:6s} conf {p.conf:.3f}  det {p.score:.2f}")
                if csv_w:
                    csv_w.writerow([f"{ts:.3f}", name, p.text, p.color, f"{p.conf:.4f}", *[int(v) for v in p.box]])
        if json_f:
            json_f.write(json.dumps({"source": name, "time_s": round(ts, 3), "plates": [
                {"text": p.text, "conf": round(p.conf, 4), "color": p.color, "det_score": round(p.score, 4),
                 "valid": ok, "box": [round(v, 1) for v in p.box], "corners": p.corners.round(1).tolist()}
                for p, ok in zip(plates, accepted)]}, ensure_ascii=False) + "\n")

        if args.show or save:
            fps_text = f"{1000.0 / max(1e-3, np.mean(lat[-20:])):.1f} fps  |  {lat[-1]:.0f} ms"
            vis = draw(frame, plates, accepted, roi_px, fps_text, font)
            if save:
                if save.suffix.lower() in {".mp4", ".avi"}:
                    if video_out is None:
                        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if save.suffix.lower() == ".mp4" else "XVID"))
                        video_out = cv2.VideoWriter(str(save), fourcc, 25.0 / max(1, args.every), (w, h))
                    video_out.write(vis)
                else:
                    out = save / (Path(name).stem + ".jpg")
                    cv2.imencode(".jpg", vis)[1].tofile(str(out))
            if args.show:
                view = vis
                if max(w, h) > 1280:
                    s = 1280 / max(w, h)
                    view = cv2.resize(vis, (int(w * s), int(h * s)))
                cv2.imshow("ALPR test", view)
                key = cv2.waitKey(0 if still else 1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    paused = not paused
                while paused:
                    k = cv2.waitKey(50) & 0xFF
                    if k == ord(" "):
                        paused = False
                    elif k in (27, ord("q")):
                        paused = False
                        break
                if key == ord("s"):
                    cv2.imencode(".jpg", vis)[1].tofile(f"snapshot_{int(time.time())}.jpg")

    wall = time.perf_counter() - t_start
    if video_out is not None:
        video_out.release()
    for f in (json_f, csv_f):
        if f:
            f.close()
    if args.show:
        cv2.destroyAllWindows()
    if lat:
        print(f"\nframes {n_frames} | plate reads {n_reads} | plates reported {n_events} | "
              f"model {np.mean(lat):.1f} ms/frame ({1000 / np.mean(lat):.1f} fps) | wall {wall:.1f} s")


if __name__ == "__main__":
    main()
