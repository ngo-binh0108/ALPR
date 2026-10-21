"""ONNX Runtime reference pipeline - the C++ app (src/core/*) implements exactly these steps."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .geometry import rectify


@dataclass
class Plate:
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 in frame pixels
    score: float
    corners: np.ndarray                         # (4, 2) TL, TR, BR, BL
    text: str = ""
    conf: float = 0.0
    color: str = ""
    crop: np.ndarray | None = field(default=None, repr=False)


def letterbox(img, w, h, pad=114):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    px, py = (w - nw) // 2, (h - nh) // 2
    out = np.full((h, w, 3), pad, np.uint8)
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, s, px, py


def nms(boxes, scores, iou_th):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][iou < iou_th]
    return keep


class AlprPipeline:
    def __init__(self, model_dir: Path, threads: int = 0):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "alpr_models.json").read_text(encoding="utf-8"))
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ["CPUExecutionProvider"]
        d, r = self.meta["detector"], self.meta["recognizer"]
        self.det = ort.InferenceSession(str(model_dir / d["file"]), so, providers=prov)
        self.rec = ort.InferenceSession(str(model_dir / r["file"]), so, providers=prov)
        self.dw, self.dh = d["input_width"], d["input_height"]
        self.nc, self.nk = len(d["class_names"]), d["num_keypoints"]
        self.fmt = d["format"]
        self.rw, self.rh = r["input_width"], r["input_height"]
        self.chars, self.colors = r["charset"], r["colors"]
        self.double_ratio = r["double_layer_ratio"]
        self.split = tuple(r.get("double_layer_split", (5 / 12, 1 / 3)))
        self.timing = {"det_ms": 0.0, "rec_ms": 0.0}

    # ------------------------------------------------------------- detection
    def detect(self, img, conf_th=0.4, iou_th=0.5) -> list[Plate]:
        lb, s, px, py = letterbox(img, self.dw, self.dh)
        blob = (lb[..., ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        t = time.perf_counter()
        out = self.det.run(None, {self.det.get_inputs()[0].name: np.ascontiguousarray(blob)})[0][0]
        self.timing["det_ms"] = (time.perf_counter() - t) * 1000
        if self.fmt == "yolo_raw":
            out = out.T                                     # anchors x (4 + nc + nk*3)
            scores = out[:, 4:4 + self.nc].max(1)
            m = scores > conf_th
            out, scores = out[m], scores[m]
            cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            kpts = out[:, 4 + self.nc:].reshape(-1, self.nk, 3) if self.nk else None
            keep = nms(boxes, scores, iou_th) if len(boxes) else []
        else:                                               # end-to-end: x1 y1 x2 y2 score cls kpts
            scores = out[:, 4]
            m = scores > conf_th
            out, scores = out[m], scores[m]
            boxes = out[:, :4]
            kpts = out[:, 6:].reshape(-1, self.nk, 3) if self.nk else None
            keep = list(range(len(boxes)))
        plates = []
        for i in keep:
            b = (boxes[i] - [px, py, px, py]) / s
            if kpts is not None and self.nk == 4:
                c = (kpts[i, :, :2] - [px, py]) / s
            else:
                c = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
            plates.append(Plate(tuple(b.tolist()), float(scores[i]), c.astype(np.float32)))
        return plates

    # ----------------------------------------------------------- recognition
    def recognize(self, img, plates: list[Plate]):
        if not plates:
            return plates
        crops = [rectify(img, p.corners, self.rw, self.rh, self.double_ratio, self.split) for p in plates]
        blob = np.stack([c.astype(np.float32) / 127.5 - 1.0 for c in crops]).transpose(0, 3, 1, 2)
        t = time.perf_counter()
        probs, color = self.rec.run(None, {"images": np.ascontiguousarray(blob)})
        self.timing["rec_ms"] = (time.perf_counter() - t) * 1000
        for p, pr, co, crop in zip(plates, probs, color, crops):
            best = pr.argmax(1)
            text, confs, prev = [], [], 0
            for t_, k in enumerate(best):
                if k != 0 and k != prev:
                    text.append(self.chars[k])
                    confs.append(pr[t_, k])
                prev = k
            p.text = "".join(text)
            p.conf = float(np.mean(confs)) if confs else 0.0
            p.color = self.colors[int(co.argmax())]
            p.crop = crop
        return plates

    def __call__(self, img, conf_th=0.4, iou_th=0.5):
        return self.recognize(img, self.detect(img, conf_th, iou_th))


class TorchPipeline:
    """Same interface as AlprPipeline but runs the PyTorch checkpoints directly
    (ultralytics best.pt + PlateNet best.pt) - handy to test a model right after training,
    before exporting it to ONNX."""

    def __init__(self, det_weights: Path, rec_weights: Path, imgsz: int = 512, device: str = "cpu"):
        import torch
        from ultralytics import YOLO

        from .charset import CHARS, COLORS, PLATE_REGEX
        from .rec_model import PlateNet

        self.torch = torch
        self.device = device
        self.imgsz = imgsz
        self.det = YOLO(str(det_weights))
        ck = torch.load(str(rec_weights), map_location=device)
        self.chars = ck.get("chars", CHARS)
        self.colors = ck.get("colors", COLORS)
        self.rec = PlateNet(num_classes=len(self.chars), num_colors=len(self.colors)).to(device).eval()
        self.rec.load_state_dict(ck["model"])
        from .geometry import DOUBLE_LAYER_RATIO, REC_H, REC_W
        self.rw, self.rh, self.double_ratio = REC_W, REC_H, DOUBLE_LAYER_RATIO
        self.meta = {"recognizer": {"plate_regex": PLATE_REGEX}}
        self.timing = {"det_ms": 0.0, "rec_ms": 0.0}

    def detect(self, img, conf_th=0.4, iou_th=0.5) -> list[Plate]:
        t = time.perf_counter()
        r = self.det.predict(img, conf=conf_th, iou=iou_th, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        self.timing["det_ms"] = (time.perf_counter() - t) * 1000
        plates = []
        if r.boxes is None or len(r.boxes) == 0:
            return plates
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
        for i, b in enumerate(boxes):
            if kpts is not None and kpts.shape[1] == 4:
                c = kpts[i].astype(np.float32)
            else:
                c = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32)
            plates.append(Plate(tuple(float(v) for v in b), float(scores[i]), c))
        return plates

    def recognize(self, img, plates: list[Plate]):
        if not plates:
            return plates
        from .charset import ctc_greedy_decode
        torch = self.torch
        crops = [rectify(img, p.corners, self.rw, self.rh, self.double_ratio) for p in plates]
        x = np.stack([c.astype(np.float32) / 127.5 - 1.0 for c in crops]).transpose(0, 3, 1, 2)
        t = time.perf_counter()
        with torch.no_grad():
            logits, color = self.rec(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
            probs = logits.softmax(-1).cpu().numpy()
            color = color.softmax(-1).cpu().numpy()
        self.timing["rec_ms"] = (time.perf_counter() - t) * 1000
        for p, pr, co, crop in zip(plates, probs, color, crops):
            p.text, p.conf = ctc_greedy_decode(pr, self.chars)
            p.color = self.colors[int(co.argmax())]
            p.crop = crop
        return plates

    def __call__(self, img, conf_th=0.4, iou_th=0.5):
        return self.recognize(img, self.detect(img, conf_th, iou_th))
