# Training — Chinese licence plates with few data

This folder trains the two small networks that ALPR Studio runs:

| Model | What it does | Architecture | Input | Size |
|---|---|---|---|---|
| `plate_det.onnx` | Finds plates **and their 4 corners** | YOLO26n-pose (1 class, 4 key-points), fine-tuned from COCO | 512×512 RGB | ~10 MB |
| `plate_rec.onnx` | Reads the characters + plate colour | PlateNet: CNN + CTC (1.8 M params, 0.36 GMAC) | 168×48 BGR | ~7 MB |

```
frame ──► ROI crop ──► detector (box + 4 corners) ──► perspective rectify (+ 2-row merge)
      ──► recogniser (CTC) ──► "粤B12345" + colour ──► tracker vote ──► event (CSV / JSON / DB)
```

### The two models in detail

**Detector: YOLO26n-pose** (Ultralytics YOLO26, *nano* size, *pose* variant)
* 2.74 M parameters, 4.2 GFLOPs at 512×512. Starts from COCO-pretrained `yolo26n-pose.pt` (human pose) and is
  re-headed to **1 class (`plate`) + 4 key-points**. The key-points are the plate corners TL, TR, BR, BL.
* Output `[1, 17, 5376]`: for each of 5 376 anchor positions (strides 8/16/32), `cx cy w h` + plate score + 4 × (x, y, visibility).
  The app applies score threshold + NMS in C++ (`PlateDetector.cpp`).
* Why pose instead of a plain box detector: the corners allow a perspective warp, so tilted or angled plates
  reach the recogniser straight. Why nano: it runs on a CPU at ~25 ms per frame.

**Recogniser: PlateNet** (written for this project, `alpr/rec_model.py`): a CRNN-style CNN with CTC, no RNN
* 1.84 M parameters, 0.36 GMAC. Input: the rectified plate, 168×48 BGR.
* Conv+BN+ReLU stack with 2 residual blocks: 48×168 → 3×21 feature map, collapsed to a sequence of 21 steps (256-d),
  2 × 1-D conv context layers, then a per-step classifier over **77 classes** (76 characters + CTC blank).
  The characters: 31 provinces, 使领警学港澳挂民航应急, 24 letters (no I/O), 10 digits.
* A second small head predicts the **plate colour** (blue, green, yellow, white, black).
* Decoding: greedy CTC (take the best class per step, merge repeats, drop blanks). Confidence = mean probability
  of the emitted characters. Two-row plates are handled before the network: the upper row is placed left of the lower row.
* Why CTC: no character segmentation is needed, 7- and 8-character plates are handled by the same model,
  and it takes ~3 ms per plate on a CPU.

Why two stages instead of one big OCR model: each stage is tiny and fast on a CPU, the corner
key-points let us straighten tilted plates (the biggest accuracy win), and the recogniser can be
pre-trained on **unlimited synthetic plates**, which is what makes training with few real images work.

---

## 0. Environment (once)

```bat
REM Anaconda Prompt
cd training
setup_env.bat              & REM creates env "alpr" (Python 3.13) from ..\packages\python (offline)
conda activate alpr
```

`packages\python` holds CPU wheels. For an NVIDIA GPU, download CUDA wheels first:
`..\packages\download_python_packages.bat cu128`, then run `setup_env.bat`.

| Package | Version tested |
|---|---|
| Python | 3.13.15 |
| torch / torchvision | 2.14.0 / 0.29.0 |
| ultralytics | 8.4.164 |
| onnx / onnxruntime / onnxslim | 1.23 / 1.30 / 0.1.96 |
| opencv-python | 5.0.0 |

---

## 1. Data — how little is enough?

The recipe needs **~800 real images for training and ~200 for validation**. That amount already
gives a usable model because:

1. the detector starts from COCO-pretrained YOLO weights (transfer learning), so it only has to learn "what a plate looks like";
2. the recogniser learns the characters from **40 000 synthetic plates** covering all 31 provinces, 5 colours,
   new-energy 8-character plates, 学/警/挂/港/澳/领 suffixes and two-row yellow plates, then needs real crops only to close the synthetic-to-real gap;
3. every real crop is re-rectified with **random corner jitter every epoch**, so 800 plates act like tens of thousands of different training samples.

### 1a. Public data (already downloaded)

| Folder | Content | Used for |
|---|---|---|
| `datasets/raw/ccpd2019/trainn` (8 900) | CCPD 2019 blue plates, balanced subset | train sample |
| `datasets/raw/ccpd2019/valn` (8 874) | CCPD 2019 blue plates | val sample / test |
| `datasets/raw/CCPD2020/ccpd_green/{train,val,test}` | new-energy green plates | train / val / test |

CCPD encodes all labels in the file name (box, 4 corners, characters), so no annotation work is needed.
CCPD is licensed for **non-commercial research**. For a commercial product, train on your own images.

### 1b. Your own vehicle photos → training data (recommended for your site)

Photos from your own cameras matter most. 200–500 of them, added to the CCPD sample, usually fix most errors
caused by your viewing angle, lighting and resolution. You never have to type every plate by hand: the current
model labels the photos, and you only correct its mistakes.

**What each label contains** (one LabelMe polygon per plate):

| Field | Value |
|---|---|
| shape | polygon with exactly **4 points** on the plate corners (any order; sorted to TL, TR, BR, BL automatically) |
| label | plate text, then `|` and the colour: `粤B12345|blue`, `粤BD12345|green`, `粤B1234学|yellow` |
| colours | `blue green yellow white black` (without a colour: 8 characters → green, otherwise blue) |
| two-row plates | label the outer corners of the whole plate and type the text as one line: `粤B12345|yellow` |

**Step by step**

1. **Collect photos.** Put the vehicle photos (or frames exported from your videos) in one folder, e.g. `D:\site_photos`.
   Aim for variety: day and night, rain, every camera and lane, near and far, tilted plates, several vehicle types.
   Skip photos where the plate is unreadable even for a human.

2. **Pre-label automatically.**
   ```bat
   python 00_pre_label.py --images D:\site_photos
   ```
   A `.json` is written next to every photo, with polygons and text from the current model:
   * `粤B12345|blue`: confident read (≥ 0.97, valid format). In our test on 300 unseen photos, **98.7 %** of these were correct.
   * `?粤B1234|blue`: uncertain. Check it, fix the text, delete the leading `?`.
   * `review_list.csv`: photos that need attention, worst first (uncertain reads and photos where nothing was found).
   * Existing `.json` files are never overwritten (your corrections are safe) unless you pass `--overwrite`.

3. **Review in LabelMe.**
   ```bat
   label_tool.bat D:\site_photos
   ```
   (The first run creates a `labelme` conda env from `packages\python_labelme`, offline.)
   * **A/D** (or *Prev/Next Image*) walks through the photos. Changes are saved automatically.
   * Wrong text: double-click the entry in *Shape List*, correct it, remove `?`.
   * Missed plate: *Create Polygons* (Ctrl+N), click the **4 corners**, type `plate|colour`.
   * Wrong corners: drag the points onto the real plate corners. Tight corners give straight crops.
   * Not a plate (false detection): select the polygon and press Delete.
   * Start with the photos at the top of `review_list.csv`, then skim the rest quickly.

4. **Fine-tune in one command** (see [Step 7](#step-7--fine-tune-on-your-own-photos-07_finetunepy)):
   ```bat
   python 07_finetune.py --train-labelme D:\site_photos --install
   ```
   It builds the dataset (your photos + a little CCPD "replay" so nothing is forgotten), measures the current
   models on your validation photos, fine-tunes detector and recogniser, measures again, exports, and with
   `--install` replaces `../models` only if the result is not worse (the old models are backed up).

5. **Check** `datasets/finetune/<name>/rec_preview.jpg` and the before/after lines, then copy `models/` next to
   `ALPRStudio.exe` (or rebuild the app, which copies it).

6. **Repeat.** Pre-label the next batch with the improved model. Each round needs fewer corrections.

(`folder:N` samples N images from a folder; `folder` alone uses all of them.)

---

## 2. Step by step

Run every command from `training/` inside `conda activate alpr`. `run_all.bat` runs steps 1–5 in order
(`run_all.bat 0` uses GPU 0).

### Step 1 – build the small dataset
```bat
python 01_prepare_data.py ^
  --train ../datasets/raw/ccpd2019/trainn:600 --train ../datasets/raw/CCPD2020/ccpd_green/train:200 ^
  --val   ../datasets/raw/ccpd2019/valn:150   --val   ../datasets/raw/CCPD2020/ccpd_green/val:50
```
This writes to `datasets/alpr/`:
* `det/` – YOLO-pose labels, one line per plate:
  `0 cx cy w h  x1 y1 2  x2 y2 2  x3 y3 2  x4 y4 2` (normalised, corners TL, TR, BR, BL), plus `data.yaml` with
  `kpt_shape: [4,3]` and `flip_idx: [1,0,3,2]`, so horizontal flips keep the corner order correct.
* `rec/real/` – loose crops around each plate plus corners (`train.jsonl`, `val.jsonl`).
* `rec_preview.jpg` – rectified examples. Open it: the plates must look straight and readable.

### Step 2 – recogniser (synthetic pre-training → real fine-tuning)
```bat
python 03_train_recognizer.py --synth 40000 --epochs-a 3 --epochs-b 25 --steps-b 60
```
* **Stage A (synthetic):** renders 40 000 plates once (`datasets/alpr/rec/synth`, ~2 min), then trains
  PlateNet on them. It uses the same geometry path as inference: random perspective → jittered corners → rectify.
* **Stage B (real):** mixes real crops (re-rectified with ±5 % corner jitter and camera degradations each epoch)
  1:1 with synthetic plates, so rare provinces and colours are not forgotten.
* Loss: CTC over 21 time steps + 0.1 × colour cross-entropy. AdamW, warm-up + cosine LR, gradient clipping.
* The best checkpoint by **exact plate accuracy on real validation crops** is saved to `runs/rec/best.pt`.

Preview the synthetic generator with `python -m alpr.synth` (writes `synth_preview.jpg`).

### Step 3 – detector (plate + 4 corners)
```bat
python 02_train_detector.py --epochs 30 --imgsz 512 --device cpu      & REM or --device 0 on a GPU
```
* Fine-tunes `packages/weights/yolo26n-pose.pt` (COCO pretrained).
* Plate-safe augmentation: mosaic (off for the last 10 epochs), scale ±50 %, rotation ±5°, shear 2°,
  small perspective, HSV, horizontal flip with `flip_idx`. No vertical flip, no mixup.
* The pose loss weight is 12, because corner accuracy drives rectification quality.
* Early stopping (`--patience`), images cached in RAM. Output: `runs/det/plate/weights/best.pt`.

### Step 4 – export to ONNX
```bat
python 04_export_onnx.py --imgsz 512
```
Writes `../models/plate_det.onnx`, `../models/plate_rec.onnx` and `../models/alpr_models.json`.
The JSON is the contract with the C++ app: input sizes, output format (`yolo_raw` = `[1, 4+nc+12, N]`),
number of key-points, normalisation, the two-row threshold, **the full charset** and the colour names.
The C++ app hard-codes none of these. The script also checks that ONNX and PyTorch outputs match (<1e-5).

### Step 5 – evaluate exactly like the app
```bat
python 05_evaluate.py --images ../datasets/raw/CCPD2020/ccpd_green/test --limit 300
python 05_evaluate.py --images ../datasets/raw/ccpd2019/valn --limit 300 --save-vis vis_out
```
Reports detection recall, end-to-end exact-match accuracy, colour accuracy and latency, and lists wrong reads.
`alpr/pipeline.py` is the reference implementation that `inference/src/core/*` mirrors step by step.

### Test the models on your own images / videos: `infer.py`
```bat
python infer.py --source D:\photos --save out_vis --csv plates.csv          & REM folder -> annotated images + CSV
python infer.py --source clip.mp4 --show --save out.mp4                       & REM video, live window + annotated video
python infer.py --source 0 --show                                             & REM USB camera
python infer.py --source rtsp://user:pw@192.168.1.64:554/Streaming/Channels/101 --show --every 2
python infer.py --source D:\photos --roi "0.2,0.3 0.8,0.3 0.9,1 0.1,1"       & REM only plates inside the polygon
python infer.py --source D:\photos --det runs/det/plate/weights/best.pt --rec runs/rec_more/best.pt   & REM PyTorch, before export
```
It uses the same pipeline as ALPR Studio (ONNX by default, or the PyTorch checkpoints with `--det/--rec`) and prints each
plate once per `--cooldown` seconds. `--json` writes every read, `--csv` the reported plates. In the window:
`q`/Esc quits, space pauses, `s` saves a frame.

### Step 7 – fine-tune on your own photos: `07_finetune.py`
Use this whenever you have new labelled photos (from `00_pre_label.py` + LabelMe review, README section 1b):
```bat
python 07_finetune.py --train-labelme D:\site_photos                     & REM CPU, 15 % of the photos become validation
python 07_finetune.py --train-labelme D:\site\train --val-labelme D:\site\val --device 0 --install
python 07_finetune.py --train-labelme D:\site_photos --what rec          & REM only the recogniser (text errors)
python 07_finetune.py --train-labelme D:\tiny_set --freeze 10            & REM < 100 photos: keep the backbone fixed
```
| Step | What happens |
|---|---|
| 1 dataset | your photos (train/val) + replay: 300 CCPD blue + 100 green (change with `--replay folder:N`, off with `--no-replay`). Validation is always your photos only |
| 2 before | current `runs/det/plate` + `runs/rec_more` models measured on your validation photos, and on a fixed sample of the **original data** (200 CCPD blue + 100 green validation images never used for training; change with `--original folder:N`, off with `--no-original`) to detect forgetting |
| 3 detector | starts from the trained detector, AdamW, **low LR 5e-4, no warm-up** (the warm-up's high bias LR damages an already trained head), mosaic off for the last third |
| 4 recogniser | starts from the trained recogniser, real crops (re-rectified with corner jitter each epoch) mixed 1:1 with synthetic plates. Every training plate is added a second time, **cut with the (new) detector's own predicted corners**, so the recogniser learns the corner errors it meets at run time (off with `--no-det-crops`) |
| 5 after | same measurement as step 2 |
| 6 export | ONNX + `alpr_models.json` into `../models_<name>`, plus `finetune_report.json`. `--install`: replaces `../models` only if plate accuracy and recall on your photos are not worse **and** the original data lost at most `--max-forget` points (default 2) |

Measure any model on your photos at any time: `python 05_evaluate.py --labelme D:\site_val`
(add `--det … --rec …` to test PyTorch checkpoints).

**Measured example** (details in `RESULTS.md` section 4): 400 green-plate "site" photos (340 train / 60 val) +
200 replay images, CPU. Recogniser fine-tune 10 × 60 steps: 13 min (+ 2 min to cut the detector crops).

| 1 000 held-out images each | Current models | Fine-tuned, hand-labelled corners only | Fine-tuned **with detector crops** (default) |
|---|---|---|---|
| Green plates (the site's plate type) | 84.5 % | 85.4 % | **90.6 %** |
| Blue plates (not in the site photos) | 93.9 % | 86.8 % | 92.8 % |

Two lessons are built into the script's defaults:
* **Detector crops matter most.** The recogniser trained on hand-labelled corners read 98 % of the site's validation
  crops, but end-to-end nothing improved, because at run time it gets the detector's slightly different corners.
  With the detector's own crops added, the site's validation photos went from 85.0 % to 93.3 %.
* **Watch forgetting.** The version trained without detector crops lost 7 points on blue plates, which never
  appeared in the site photos, and the site photos alone would not show it. That is why step 2/5 also measures the
  original data and `--install` refuses such a model. More `--replay` (the default is 300 + 100, twice what this
  example used) reduces forgetting further.

### Step 6 (optional) – INT8 detector
```bat
python 06_quantize.py          & REM writes ../models/plate_det_int8.onnx and prints FP32 vs INT8 speed
python 06_quantize.py --use    & REM also switches alpr_models.json to the INT8 detector
```
This only helps on CPUs with VNNI int8 instructions. On the reference Zen 3 VM it was slower (0.9×), so the default stays FP32.

---

## 3. Results of the reference run

The measured numbers from this machine (CPU only) are in **[RESULTS.md](RESULTS.md)**. In short, with 800 training images:
blue plates reach 100 % detection and 91.7 % exact end-to-end reads, green plates 98.9 % and 82.3 %, and the C++ engine runs at 31 ms/frame.

---

## 4. Improving the model

| Symptom | Fix |
|---|---|
| Plates missed at your site | Label 200–500 frames from your cameras (step 1b), then fine-tune: `02_train_detector.py --model runs/det/plate/weights/best.pt --epochs 30` |
| Characters confused (e.g. 8/B, 0/D) | Add real crops from your site, then `03_train_recognizer.py --skip-a --init runs/rec/best.pt --epochs-b 20` |
| Rare provinces wrong | Raise `--synth` (e.g. 100 000) and `--epochs-a` |
| Small, distant plates | Draw a tighter ROI in the app (plates look bigger to the detector), or export with `--imgsz 640` |
| Two-row (yellow truck) plates | Already handled (aspect < 2.3 → top row placed left of bottom row); add real examples via LabelMe |
| Fonts look wrong in `synth_preview.jpg` | Put GA36 plate fonts into `training/fonts/` (`platech.ttf` Chinese, `platechar.ttf` letters/digits) |

**GPU:** an NVIDIA GPU is roughly 20–50× faster. Use `--device 0 --batch 32` for the detector; the recogniser uses
the GPU automatically when you install CUDA torch (`download_python_packages.bat cu128`).

**CPU tip:** on this VM, PyTorch slowed down about 20× above 8 threads and two training jobs at once thrashed each
other. Run one job at a time and use `--threads` (e.g. 8) if your CPU behaves the same way.

## 5. Other plate styles and languages
Yes, this works. See **[EXTENDING.md](EXTENDING.md)**. Characters, plate-format rule, colours, input size and two-row
layout all live in `alpr/charset.py` / `alpr/geometry.py`. They are exported into `alpr_models.json`, so ALPR Studio
needs no code change for a new language.

## 6. Licences
* Ultralytics YOLO is **AGPL-3.0**. For closed-source commercial use you need an Ultralytics Enterprise licence,
  or you can swap in another pose detector that produces the same `[1, 4+nc+12, N]` output.
* CCPD: research use only.
* PlateNet, the scripts and the C++ app are yours.
