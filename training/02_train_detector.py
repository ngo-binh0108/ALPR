"""Step 2 - fine-tune a YOLO pose model to find plates *and* their 4 corners.

Why pose instead of plain boxes: the 4 corner key-points let us perspective-rectify
tilted plates before recognition, which is the single biggest accuracy win for
Chinese plates seen at an angle.

Few-data recipe (a few hundred images is enough):
  * start from COCO-pretrained yolo26n-pose (transfer learning, NMS-free-friendly head)
  * strong but plate-safe augmentation (mosaic, scale, HSV, small rotation/perspective,
    horizontal flip with flip_idx so corner order stays TL,TR,BR,BL)
  * cache images in RAM, early stopping (patience)

Example
    python 02_train_detector.py --epochs 60 --device cpu      # CPU
    python 02_train_detector.py --epochs 100 --device 0 --batch 32   # NVIDIA GPU
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "datasets/alpr/det/data.yaml"))
    ap.add_argument("--model", default=str(ROOT / "packages/weights/yolo26n-pose.pt"),
                    help="pretrained weights (yolo26n-pose.pt / yolo11n-pose.pt / a previous best.pt)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu | 0 | 0,1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--project", default=str(ROOT / "training/runs/det"))
    ap.add_argument("--name", default="plate")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    # fine-tuning controls (defaults = Ultralytics defaults, used when training from COCO weights)
    ap.add_argument("--lr0", type=float, default=None, help="initial learning rate (fine-tune: ~0.002)")
    ap.add_argument("--warmup", type=float, default=None, help="warm-up epochs (fine-tune: 0 = none)")
    ap.add_argument("--freeze", type=int, default=None, help="freeze the first N layers (10 = backbone, for tiny datasets)")
    ap.add_argument("--close-mosaic", type=int, default=10, help="last N epochs without mosaic")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if args.threads:
        torch.set_num_threads(args.threads)

    extra = {k: v for k, v in {"lr0": args.lr0, "warmup_epochs": args.warmup, "freeze": args.freeze}.items()
             if v is not None}
    if args.lr0 is not None:
        extra["optimizer"] = "AdamW"   # optimizer="auto" would ignore lr0
    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, project=args.project, name=args.name, exist_ok=True, resume=args.resume,
        patience=args.patience, cache="ram", pretrained=True, seed=0, deterministic=False,
        # augmentation tuned for plates
        mosaic=1.0, close_mosaic=min(args.close_mosaic, args.epochs), mixup=0.0, degrees=5.0, translate=0.1, scale=0.5, shear=2.0,
        perspective=0.0005, fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        # loss weights: key-points matter as much as the box for rectification
        pose=12.0, kobj=1.0, plots=True, **extra,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nbest detector weights: {best}")
    metrics = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=False)
    print(f"box mAP50={metrics.box.map50:.4f}  pose mAP50={metrics.pose.map50:.4f}")


if __name__ == "__main__":
    main()
