# Extending ALPR to other plate styles and other languages

**Yes, you can.** Nothing in the pipeline is tied to Chinese except *data and configuration*:

* The **detector** only learns "a plate-shaped rectangle with 4 corners", which works for any country.
* The **recogniser** (CTC) reads any sequence of symbols. Chinese, Hangul, Latin, Arabic digits and so on are just
  entries in a character list.
* Everything language-specific is written into **`models/alpr_models.json`** at export time, and the C++ app reads it
  from there. **ALPR Studio needs no code change and no recompile** for a new language. You copy the new `models/` folder next to the exe.

| What | Where you change it | Exported to `alpr_models.json` as | Used by |
|---|---|---|---|
| Characters | `alpr/charset.py` (`PROVINCES`, `SPECIALS`, `LETTERS`, `DIGITS` → `CHARS`) | `charset` | recogniser output decoding (C++ + Python) |
| Valid plate formats | `alpr/charset.py` → `PLATE_REGEX` | `plate_regex` | "Only accept valid plate formats", pre-labelling |
| Plate colours / types | `alpr/charset.py` → `COLORS` | `colors` | colour head, results `color` field |
| Recogniser input size | `alpr/geometry.py` → `REC_W`, `REC_H` | `input_width/height` | rectification + network input |
| Two-row plates | `alpr/geometry.py` → `DOUBLE_LAYER_RATIO`, `UPPER_ROW_END`, `LOWER_ROW_START` | `double_layer_ratio`, `double_layer_split` | row merge before recognition |
| Synthetic plates | `alpr/synth.py` (`random_plate`, `render_plate`), fonts in `training/fonts/` | – | recogniser pre-training |
| Real examples | LabelMe labels (any Unicode text, `text|colour`) | – | detector + recogniser fine-tuning |

---

## A. A new *Chinese* plate style

Examples: motorcycle, tractor, 应急 emergency, 民航, cross-border 粤Z…港/澳, new-energy trucks.

1. **Characters:** most are already in `SPECIALS` (`使领警学港澳挂民航应急`). Add any missing ones there.
2. **Format:** add an alternative to `PLATE_REGEX`, e.g. motorcycles `f"|{_P}{_A}{_X}{{5}}"` are already covered.
3. **Synthetic layout:** in `synth.py` add a branch to `random_plate()` (text + colour + style) and, if the geometry
   differs, a layout to `render_plate()` (character boxes in mm, background colour, text colour).
   Check the result with `python -m alpr.synth`, which writes `synth_preview.jpg`.
4. **Real data:** 100–300 photos of the style, labelled with `00_pre_label.py` + LabelMe (README section 1b).
5. **Train.** If only real photos are new, one command does it: `python 07_finetune.py --train-labelme <folder> --install`.
   If you also added a synthetic style, run the recogniser with stage A:
   ```bat
   python 03_train_recognizer.py --real ../datasets/site/rec/real --init runs/rec_more/best.pt --epochs-a 2 --epochs-b 20 --out runs/rec_style
   ```
   Fine-tune the detector only if the plate *shape* is new (e.g. square motorcycle plates):
   `02_train_detector.py --model runs/det/plate/weights/best.pt --name style`.
6. **Export** (`04_export_onnx.py`) and copy `models/` to the app.

## B. Another country / language

Example: Korean plates `12가3456` (Hangul). Latin (EU, US), Japanese, Arabic and others follow the same steps.

1. **Characters.** Make `alpr/charset.py` describe the new alphabet (keep `BLANK` at index 0):
   ```python
   PROVINCES = ""                                    # no region prefix in modern Korean plates
   SPECIALS  = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호배"
   LETTERS   = ""
   DIGITS    = "0123456789"
   COLORS    = ["white", "yellow", "green", "blue"]
   ```
   For several countries in one model, put all symbols into the lists (union) and all formats into the regex.
2. **Format rule:** `PLATE_REGEX = r"^\d{2,3}[가-힣]\d{4}$"`.
3. **Input size.** The default 168×48 suits plates of aspect about 3.5 (China). Korea/EU 520×110 is 4.7 → use
   `REC_W, REC_H = 224, 48`. Only change the **width**: the network collapses exactly 48 px of height.
   Width/8 gives the number of CTC time steps (168 → 21, 224 → 28), which should stay ≥ 2 × the plate length.
4. **Two-row plates** (older Korean/Japanese plates, motorcycles): adjust `DOUBLE_LAYER_RATIO` and the split
   `UPPER_ROW_END` / `LOWER_ROW_START` to where the rows really are. The app reads them from the JSON.
5. **Fonts + synthetic layout.** Put the country's plate font (or e.g. Malgun Gothic for Hangul) into `training/fonts/`,
   then write the layout in `synth.py`: plate size in mm, character boxes, colours. Aim for `synth_preview.jpg` to look like real plates.
6. **Data.** For a new script, plan on about 500–2 000 labelled real images. Label text is written in the native script,
   in visual left-to-right order (also for Arabic). Start labelling by hand. Once a first model exists, `00_pre_label.py`
   does most of the work.
7. **Train.**
   * Recogniser: new charset → run stage A (synthetic) and stage B. Warm-start from the Chinese model anyway:
     `--init runs/rec_more/best.pt` reuses the whole backbone (66 of 70 tensors) and only the classifier and colour head
     start fresh. That converges much faster than from scratch.
   * Detector: fine-tune from `runs/det/plate/weights/best.pt`. Plates look alike worldwide, so 200–500 images are enough.
8. **Export → copy `models/` → done.** ALPR Studio reads the new charset, format rule, colours, size and split.
   Plate text is drawn with the system font fallback (YaHei → Malgun Gothic / Segoe UI), so Hangul, Latin and Arabic display correctly.

## Checklist when something looks wrong
* `rec_preview.jpg` from `01_prepare_data.py`: every crop must be one straight, complete plate.
* `synth_preview.jpg`: synthetic plates should resemble real ones (layout, font weight, colours).
* Reads are correct but always rejected → `PLATE_REGEX` is too strict (or untick *Only accept valid plate formats*).
* Characters dropped at the edges → corners too tight. Label the corners on the plate's outer edge.
* Changing `CHARS` means old recogniser checkpoints no longer match the classifier size. Always export the recogniser
  trained with the new charset, never an old one.
