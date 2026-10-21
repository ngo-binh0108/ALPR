"""Step 0 - turn a folder of vehicle photos into LabelMe annotations *automatically*, so you only
review and correct instead of labelling from scratch.

For every image the current models (../models) find each plate and read it. A LabelMe JSON is
written next to the image with one 4-point polygon per plate:

    label = "粤B12345|blue"     confident read (>= --accept, valid plate format)
    label = "?粤B1234|blue"     uncertain read -> check it, fix the text and delete the leading "?"

Images are listed in review_list.csv, worst first, so the uncertain ones get checked first.
01_prepare_data.py uses a "?" plate for the detector (the corners are fine), but not for the
recogniser until you remove the "?".

Example
    python 00_pre_label.py --images D:/site_photos
    label_tool.bat D:/site_photos                     (LabelMe review; see README "1b. Your own vehicle photos")
    python 01_prepare_data.py --train-labelme D:/site_photos --out ../datasets/site
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpr.charset import PLATE_REGEX  # noqa: E402
from alpr.pipeline import AlprPipeline  # noqa: E402
from alpr.rec_data import imread  # noqa: E402

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with vehicle photos (searched recursively)")
    ap.add_argument("--models", default=str(HERE.parent / "models"))
    ap.add_argument("--conf", type=float, default=0.25, help="detector threshold (low: better to review extra boxes)")
    ap.add_argument("--accept", type=float, default=0.97, help="reads above this (and valid format) need no '?'")
    ap.add_argument("--overwrite", action="store_true", help="replace existing .json files (default: keep your edits)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    pipe = AlprPipeline(Path(args.models), threads=args.threads)
    plate_re = re.compile(pipe.meta["recognizer"].get("plate_regex", PLATE_REGEX))
    root = Path(args.images)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    review = []
    n_plates = n_sure = skipped = 0
    for i, path in enumerate(images, 1):
        js = path.with_suffix(".json")
        if js.exists() and not args.overwrite:
            skipped += 1
            continue
        img = imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        plates = pipe(img, conf_th=args.conf)
        shapes, worst = [], 1.0
        for p in plates:
            sure = p.conf >= args.accept and bool(plate_re.match(p.text))
            label = f"{'' if sure else '?'}{p.text}|{p.color}"
            shapes.append({"label": label, "points": [[round(float(x), 1), round(float(y), 1)] for x, y in p.corners],
                           "group_id": None, "description": f"det {p.score:.2f} rec {p.conf:.2f}",
                           "shape_type": "polygon", "flags": {}, "mask": None})
            n_plates += 1
            n_sure += sure
            worst = min(worst, p.conf if sure else p.conf * 0.5)
        if not plates:
            worst = 0.0   # nothing found: maybe a missed plate -> review
        doc = {"version": "5.5.0", "flags": {"needs_review": worst < args.accept}, "shapes": shapes,
               "imagePath": path.name, "imageData": None, "imageHeight": h, "imageWidth": w}
        js.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        if worst < args.accept:
            review.append((round(worst, 3), str(path.relative_to(root)), len(plates),
                           " ".join(s["label"] for s in shapes) or "(no plate found)"))
        if i % 50 == 0:
            print(f"  {i}/{len(images)} images")

    review.sort()
    with open(root / "review_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["score", "image", "plates", "labels"])
        wr.writerows(review)
    print(f"images {len(images)} (kept {skipped} existing .json) | plates {n_plates} | confident {n_sure} | "
          f"to review {len(review)} -> {root / 'review_list.csv'}")


if __name__ == "__main__":
    main()
