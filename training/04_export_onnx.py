"""Step 4 - export both models to ONNX and write the model contract for the C++ app.

Produces (default ../models):
    plate_det.onnx      YOLO pose detector   input images[1,3,S,S] RGB/255
    plate_rec.onnx      PlateNet recogniser  input images[B,3,48,168] BGR [-1,1]
    alpr_models.json    everything the C++ app needs (sizes, formats, charset, colours)

Example
    python 04_export_onnx.py --det runs/det/plate/weights/best.pt --rec runs/rec/best.pt --imgsz 512
"""
import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpr.charset import CHARS, COLORS, PLATE_REGEX  # noqa: E402
from alpr.geometry import DOUBLE_LAYER_RATIO, LOWER_ROW_START, REC_H, REC_W, UPPER_ROW_END  # noqa: E402
from alpr.rec_model import PlateNet, PlateNetExport  # noqa: E402

HERE = Path(__file__).resolve().parent


def export_detector(weights: Path, imgsz: int, out: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False))
    shutil.copy2(onnx_path, out / "plate_det.onnx")
    sess = ort.InferenceSession(str(out / "plate_det.onnx"), providers=["CPUExecutionProvider"])
    shape = sess.get_outputs()[0].shape
    names = model.names
    nc = len(names)
    # ultralytics writes the key-point layout into the ONNX metadata, e.g. "[4, 3]"
    meta = sess.get_modelmeta().custom_metadata_map
    kpt_shape = ast.literal_eval(meta["kpt_shape"]) if "kpt_shape" in meta else [0, 3]
    nk = int(kpt_shape[0])
    # raw ultralytics layout: [1, 4 + nc + nk*3, anchors];  end-to-end layout: [1, max_det, 6 + nk*3]
    fmt = "yolo_raw" if shape[1] == 4 + nc + nk * 3 else "yolo_e2e"
    print(f"detector output {shape} -> format {fmt}, classes {names}, keypoints {nk}")
    return {"file": "plate_det.onnx", "input_width": imgsz, "input_height": imgsz, "format": fmt,
            "class_names": [names[i] for i in range(nc)], "num_keypoints": nk,
            "input_layout": "NCHW_RGB_0_1", "letterbox_pad": 114}


def export_recognizer(weights: Path, out: Path) -> dict:
    ck = torch.load(str(weights), map_location="cpu")
    assert ck["chars"] == CHARS, "charset of checkpoint differs from alpr/charset.py"
    net = PlateNet()
    net.load_state_dict(ck["model"])
    net.eval()
    model = PlateNetExport(net).eval()
    dummy = torch.zeros(1, 3, REC_H, REC_W)
    path = out / "plate_rec.onnx"
    torch.onnx.export(model, dummy, str(path), input_names=["images"], output_names=["chars", "color"],
                      dynamic_axes={"images": {0: "batch"}, "chars": {0: "batch"}, "color": {0: "batch"}},
                      opset_version=17, dynamo=False)
    try:
        import onnx
        import onnxslim
        onnx.save(onnxslim.slim(onnx.load(str(path))), str(path))
    except Exception as e:  # slimming is optional
        print(f"onnxslim skipped: {e}")
    # parity check torch vs onnxruntime
    x = torch.randn(4, 3, REC_H, REC_W)
    with torch.no_grad():
        ref = model(x)[0].numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": x.numpy()})[0]
    print(f"recogniser ONNX max abs diff vs torch: {np.abs(ref - got).max():.2e}, output {got.shape}")
    return {"file": "plate_rec.onnx", "input_width": REC_W, "input_height": REC_H, "time_steps": int(got.shape[1]),
            "input_layout": "NCHW_BGR_m1_1", "double_layer_ratio": DOUBLE_LAYER_RATIO,
            "double_layer_split": [UPPER_ROW_END, LOWER_ROW_START],
            "charset": CHARS, "colors": COLORS, "plate_regex": PLATE_REGEX, "val_accuracy": float(ck.get("acc", 0.0))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", default=str(HERE / "runs/det/plate/weights/best.pt"))
    ap.add_argument("--rec", default=str(HERE / "runs/rec/best.pt"))
    ap.add_argument("--imgsz", type=int, default=512, help="detector input size (512 fast, 640 small plates)")
    ap.add_argument("--out", default=str(HERE.parent / "models"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = {"version": 1, "detector": export_detector(Path(args.det), args.imgsz, out),
            "recognizer": export_recognizer(Path(args.rec), out)}
    (out / "alpr_models.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"models + alpr_models.json written to {out}")


if __name__ == "__main__":
    main()
