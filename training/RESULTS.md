# Reference training run — measured results

Machine: AWS VM, CPU only (reported as Ryzen 9 5950X, ~8 effective cores), no GPU, 2026-09-28.
Everything below was produced by the scripts in this folder, with the default arguments in `run_all.bat`.

## 1. Few-data recipe (800 training images)

| Data | Images |
|---|---|
| Train | 600 CCPD2019 blue + 200 CCPD2020 green |
| Validation (model selection) | 150 blue + 50 green |
| Synthetic plates (recogniser stage A) | 40 000 |

### Training curves
| Stage | Epochs | Time (CPU) | Result on real validation data |
|---|---|---|---|
| Recogniser A — synthetic only | 3 | 42 min | exact plate 55.0 %, characters 88.9 % |
| Recogniser B — real + synthetic | 25 | 29 min | **exact plate 97.0 %**, characters 98 %, colour 99.5 % |
| Detector — YOLO26n-pose 512 px | 30 | 1 h 52 min | box mAP50 **0.994**, corner (pose) mAP50-95 **0.993** |

Synthetic pre-training alone already reads 55 % of real plates correctly. A single epoch of real fine-tuning then lifts it to 90 %.

### End-to-end on held-out images (never used for training or model selection)
`python 05_evaluate.py --images <folder> --limit 1000` (seeded random sample, `used_files.txt` excluded)

| Test set | Detection recall | Exact plate (end-to-end) | Colour |
|---|---|---|---|
| CCPD2019 blue, 1 000 images | **100 %** | **91.7 %** | 100 % |
| CCPD2020 green **test split**, 1 000 images | **98.9 %** | **82.3 %** | 97.8 % |

Where the errors come from:
* **The detector is not the bottleneck.** On green test plates, recognition from the detector's corners (85.6 %) is as good as from the hand-labelled ground-truth corners (85.0 %).
  (This holds for this recogniser. A recogniser fine-tuned only on hand-labelled corners of a small site set does develop a gap; see section 4.)
* **Blue plates:** most errors are the first Chinese character (province, e.g. 皖↔新, 闽↔鄂). `ccpd2019balance` spreads 31 provinces evenly, so 600 images give only ~20 real examples per province.
* **Green plates:** length errors (a character dropped or doubled in the dense 8-character layout) and D/B/0 confusions. 200 real green examples are few.

Both are data problems, not recipe problems. See section 2.

## 2. Same recipe, more real crops (recogniser only)

`01_prepare_data.py --out ../datasets/alpr_more --train ccpd2019/trainn:2000 --train ccpd_green/train:2000`, then
`03_train_recognizer.py --skip-a --init runs/rec/best.pt --epochs-b 20 --steps-b 100` (the detector is unchanged).

Stage B, 20 epochs × 100 steps, took 33 min on the CPU. Evaluated on the same held-out random samples
(images used by either run are excluded):

| Test set (1 000 images each) | 800 images (section 1) | 4 000 real crops for the recogniser |
|---|---|---|
| CCPD2019 blue, exact plate | 91.7 % | **93.9 %** |
| CCPD2020 green test, exact plate | 82.3 % | **85.5 %** |
| Colour (green / blue) | 97.8 % / 100 % | 99.5 % / 100 % |

Five times more real recogniser data gives +2 to +3 points, at identical speed. **`models/` ships this recogniser**
together with the 800-image detector (detection is already at 99–100 % recall). The few-data recogniser stays in
`runs/rec/best.pt`; re-export it with `04_export_onnx.py --rec runs/rec/best.pt` if you want the strict few-data model.
For your own site, 200–500 labelled frames from your cameras (README step 1b) are worth more than extra CCPD data.

## 3. Speed (C++ engine, `ALPRStudio.exe --bench`, same VM, idle CPU)

| Configuration | Result |
|---|---|
| 1 channel, 8 threads | **31 ms/frame** (26 ms detect + 3 ms recognise) → 32 fps |
| 4 channels (concurrent) | 60 frames/s total → 15 LPR fps per channel |
| 12 channels (concurrent) | 59 frames/s total → 5 LPR fps per channel |
| Python reference pipeline, 1 stream | 57 ms/frame (the C++ engine is ~2× faster) |

On this VM total throughput levels off at about 60 analysed frames/s. A normal 8–16-core desktop CPU
roughly doubles to quadruples that, and CUDA far more. Display always runs at 25 fps, independently of LPR.

**C++ ↔ Python parity** on 90 held-out frames: identical boxes and corners (0.000 px difference), identical
text for 81/82 plates. The single difference was a low-confidence read (0.74 vs 0.69) below the 0.80 acceptance threshold.

**INT8 detector (`06_quantize.py`):** 0.90× speed (slower) on this Zen 3 CPU, recall 99.8 %, plate accuracy 91.2 % vs 91.7 %.
Only worth it on CPUs with VNNI (Intel 11th gen+ / AMD Zen 4+); measure with `--bench` before switching.

## 4. Fine-tuning on "site" photos (`07_finetune.py`)

Simulated site: 400 CCPD2020 green **test** photos with true LabelMe labels, split by the script into 340 train /
60 validation. Replay: 150 CCPD2019 blue + 50 green training images. Everything on the CPU:
detector 6 epochs (14 min, early-stopped, best epoch 1, box mAP50 0.994), recogniser 10 × 60 steps (13 min).

| Variant | Site validation, 60 plates | Held-out green, 1 000 | Held-out blue, 1 000 |
|---|---|---|---|
| A. current `models/` | 85.0 % | 84.5 % | 93.9 % |
| B. fine-tuned detector + recogniser on hand-labelled corners | 85.0 % | 85.4 % | 86.8 % |
| C. as B, plus 538 crops cut with the detector's own corners (**default**) | **93.3 %** | **90.6 %** | 92.8 % |

End-to-end exact plate; detection recall is 98.8–100 % everywhere. Held-out = `05_evaluate.py --limit 1000`
with every image used by any run excluded (the 400 site photos included); these images were never used for
training or model selection. The 60-photo numbers are slightly optimistic, because the recogniser checkpoint is
chosen on crops of those photos.

* **The fine-tuned detector alone changes nothing** (fine-tuned detector + original recogniser: 85.0 %). It
  was already at 99 % recall.
* **Hand-labelled corners mislead the recogniser.** B reads 98.3 % of the validation plates cut with the
  hand-labelled corners, but only 85.0 % end-to-end: at run time the crops come from the detector's corners, which
  differ by a few pixels. Adding crops cut with the detector's corners (C) closes most of that gap, +6.1 points on
  1 000 unseen green plates.
* **Forgetting.** B lost 7.1 points on blue plates, which the site photos do not contain, and the site's own
  validation photos cannot show it. C lost 1.1 points; its detector crops also cover the replay images, which doubles
  the real blue data. `07_finetune.py` therefore also measures 200 blue + 100 green original validation images
  before and after, and `--install` refuses a model that loses more than `--max-forget` (2) points there. The
  default replay (300 + 100) is twice what this run used.
