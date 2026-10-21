"""Step 3 - train the plate recogniser (CTC) with very little real data.

Stage A  synthetic pre-training : rendered plates for all provinces / colours / 2-row plates
Stage B  real fine-tuning       : few real crops (re-rectified with random corner jitter every
                                  epoch) mixed 1:1 with synthetic plates to avoid forgetting

Example
    python 03_train_recognizer.py                         # full recipe (A + B)
    python 03_train_recognizer.py --synth 100000 --epochs-a 10 --epochs-b 40
    python 03_train_recognizer.py --skip-a --init runs/rec/best.pt   # fine-tune only (new site data)
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpr.charset import CHARS, COLORS, ctc_greedy_decode  # noqa: E402
from alpr.rec_data import RealCropDataset, RectifiedDataset, collate  # noqa: E402
from alpr.rec_model import PlateNet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cpu")


# ----------------------------------------------------------------- synthetic set
def _gen_chunk(job):
    from alpr.synth import synth_sample
    start, count, seed, out = job
    rng = random.Random(seed)
    rows = []
    for i in range(start, start + count):
        img, text, color = synth_sample(rng)
        rel = f"{i // 10000:02d}/{i:07d}.jpg"
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(Path(out) / rel))
        rows.append(f"{rel}\t{text}\t{color}")
    return rows


def build_synth(out: Path, n: int, seed: int, workers: int):
    labels = out / "labels.tsv"
    if labels.exists() and sum(1 for _ in labels.open(encoding="utf-8")) >= n:
        print(f"synthetic set ready: {out} ({n})")
        return
    for k in range(n // 10000 + 1):
        (out / f"{k:02d}").mkdir(parents=True, exist_ok=True)
    chunk = 1000
    jobs = [(s, min(chunk, n - s), seed * 100003 + s, str(out)) for s in range(0, n, chunk)]
    t = time.time()
    rows = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(_gen_chunk, jobs)):
            rows += r
            print(f"\r  rendering synthetic plates {len(rows)}/{n}", end="", flush=True)
    labels.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n  done in {time.time() - t:.0f}s -> {out}")


# ----------------------------------------------------------------- train / eval
def _worker_init(_):
    # one thread per loader process, otherwise OpenMP/OpenCV pools oversubscribe the CPU
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def ctc_loss(logits, targets, lengths):
    log_probs = logits.log_softmax(-1).permute(1, 0, 2)          # T,B,C
    T, B = log_probs.shape[:2]
    in_len = torch.full((B,), T, dtype=torch.long, device=log_probs.device)
    return F.ctc_loss(log_probs, targets, in_len, lengths, blank=0, zero_infinity=True)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ok = char_ok = char_all = color_ok = n = 0
    for imgs, _, _, colors, texts in loader:
        logits, color = model(imgs.to(DEVICE))
        probs = logits.softmax(-1).cpu().numpy()
        pred_color = color.argmax(1).cpu()
        for p, t, pc, gc in zip(probs, texts, pred_color, colors):
            s, _ = ctc_greedy_decode(p)
            ok += s == t
            char_ok += sum(a == b for a, b in zip(s, t)) if len(s) == len(t) else 0
            char_all += len(t)
            color_ok += int(pc == gc)
            n += 1
    model.train()
    return ok / max(n, 1), char_ok / max(char_all, 1), color_ok / max(n, 1)


def run_stage(name, model, train_ds, val_loader, epochs, lr, batch, workers, steps_per_epoch, out_dir, best_acc):
    sampler = RandomSampler(train_ds, replacement=True, num_samples=steps_per_epoch * batch)
    loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=workers, collate_fn=collate,
                        persistent_workers=workers > 0, drop_last=True, worker_init_fn=_worker_init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * steps_per_epoch
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    for ep in range(1, epochs + 1):
        t0, run_loss = time.time(), 0.0
        for it, (imgs, targets, lengths, colors, _) in enumerate(loader, 1):
            imgs, targets, lengths, colors = imgs.to(DEVICE), targets.to(DEVICE), lengths.to(DEVICE), colors.to(DEVICE)
            logits, color_logits = model(imgs)
            loss = ctc_loss(logits, targets, lengths) + 0.1 * F.cross_entropy(color_logits, colors)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
            if it % 20 == 0:
                print(f"\r  [{name}] ep {ep}/{epochs} it {it}/{steps_per_epoch} loss {run_loss / it:.4f}", end="", flush=True)
        acc, char_acc, col_acc = evaluate(model, val_loader)
        print(f"\r  [{name}] ep {ep}/{epochs} loss {run_loss / steps_per_epoch:.4f} | real-val plate acc {acc:.4f} "
              f"char acc {char_acc:.4f} colour acc {col_acc:.4f} | {time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict(), "chars": CHARS, "colors": COLORS}, out_dir / "last.pt")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "chars": CHARS, "colors": COLORS, "acc": acc}, out_dir / "best.pt")
            print(f"  -> new best {acc:.4f} saved")
    return best_acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default=str(ROOT / "datasets/alpr/rec/real"), help="output of 01_prepare_data.py")
    ap.add_argument("--synth-dir", default=str(ROOT / "datasets/alpr/rec/synth"))
    ap.add_argument("--synth", type=int, default=60000, help="number of synthetic plates to render")
    ap.add_argument("--epochs-a", type=int, default=6)
    ap.add_argument("--epochs-b", type=int, default=30)
    ap.add_argument("--steps-a", type=int, default=0, help="iterations per stage-A epoch (0 = one pass)")
    ap.add_argument("--steps-b", type=int, default=150, help="iterations per stage-B epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = auto)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--init", default="", help="start from these weights")
    ap.add_argument("--out", default=str(ROOT / "training/runs/rec"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    global DEVICE
    DEVICE = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"device: {DEVICE}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    build_synth(Path(args.synth_dir), args.synth, args.seed, args.workers or 1)
    synth = RectifiedDataset(Path(args.synth_dir), augment=True)
    real_train = RealCropDataset(Path(args.real), "train.jsonl", augment=True)
    real_val = RealCropDataset(Path(args.real), "val.jsonl", augment=False)
    val_loader = DataLoader(real_val, batch_size=256, num_workers=0, collate_fn=collate)
    print(f"synthetic {len(synth)} | real train {len(real_train)} | real val {len(real_val)}")

    model = PlateNet().to(DEVICE)
    if args.init:
        # load every tensor whose shape matches: a checkpoint with another charset / colour list still
        # provides the whole backbone, only the character classifier / colour head start fresh
        src = torch.load(args.init, map_location=DEVICE)["model"]
        own = model.state_dict()
        keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        skipped = sorted(set(own) - set(keep))
        print(f"initialised from {args.init} ({len(keep)}/{len(own)} tensors"
              + (f"; new: {', '.join(skipped)}" if skipped else "") + ")")
    print(f"PlateNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    best = -1.0
    if not args.skip_a:
        steps = args.steps_a or len(synth) // args.batch
        best = run_stage("A synth", model, synth, val_loader, args.epochs_a, args.lr, args.batch, args.workers,
                         steps, out, best)
    # stage B: real crops repeated to match a synthetic slice of equal size (1:1 mix)
    reps = max(1, len(synth) // max(1, len(real_train)) // 4)
    mixed = ConcatDataset([Subset(real_train, list(range(len(real_train))) * reps),
                           Subset(synth, random.sample(range(len(synth)), min(len(synth), len(real_train) * reps)))])
    best = run_stage("B real", model, mixed, val_loader, args.epochs_b, args.lr * 0.5, args.batch, args.workers,
                     args.steps_b, out, best)
    print(f"\nbest real-val plate accuracy {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
