"""Step 6 (optional) - INT8 static quantisation of the detector for faster CPU inference.

Only Conv layers are quantised (QDQ, per-channel weights); the YOLO decode math stays FP32 so
box / key-point coordinates keep their precision.  Calibration uses real validation images.

Example
    python 06_quantize.py                      # writes ../models/plate_det_int8.onnx
    python 06_quantize.py --use                # also points alpr_models.json at the INT8 model
Then compare with 05_evaluate.py (accuracy) and ALPRStudio.exe --bench (speed) before keeping it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpr.pipeline import letterbox  # noqa: E402
from alpr.rec_data import imread  # noqa: E402

HERE = Path(__file__).resolve().parent


class Reader(CalibrationDataReader):
    def __init__(self, files, size, name):
        self.files, self.size, self.name = list(files), size, name

    def get_next(self):
        if not self.files:
            return None
        img = imread(self.files.pop())
        lb, *_ = letterbox(img, self.size, self.size)
        x = (lb[..., ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return {self.name: np.ascontiguousarray(x)}


def bench(path, x, n=30):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    for _ in range(3):
        s.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(n):
        s.run(None, {name: x})
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(HERE.parent / "models"))
    ap.add_argument("--calib", default=str(HERE.parent / "datasets/alpr/det/images/val"))
    ap.add_argument("--num", type=int, default=100, help="calibration images")
    ap.add_argument("--use", action="store_true", help="switch alpr_models.json to the INT8 detector")
    args = ap.parse_args()

    models = Path(args.models)
    meta_path = models / "alpr_models.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    src = models / "plate_det.onnx"
    pre = models / "plate_det_pre.onnx"
    dst = models / "plate_det_int8.onnx"
    size = meta["detector"]["input_width"]

    quant_pre_process(str(src), str(pre))
    name = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    files = sorted(Path(args.calib).glob("*.jpg"))[: args.num]
    quantize_static(str(pre), str(dst), Reader(files, size, name), quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=["Conv"], per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)

    x = np.random.rand(1, 3, size, size).astype(np.float32)
    fp, q = bench(src, x), bench(dst, x)
    print(f"FP32 {fp:.1f} ms  ->  INT8 {q:.1f} ms  ({fp / q:.2f}x)   [{dst.name}, {dst.stat().st_size / 1e6:.1f} MB]")
    if args.use:
        meta["detector"]["file"] = dst.name
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("alpr_models.json now uses the INT8 detector")


if __name__ == "__main__":
    main()
