"""Plate rectification helpers.  The C++ app (PlateRecognizer.cpp) mirrors these exactly."""
import cv2
import numpy as np

REC_W, REC_H = 168, 48          # recogniser input size (width, height)
DOUBLE_LAYER_RATIO = 2.3        # plate width/height below this => two-row plate
UPPER_ROW_END = 5 / 12          # two-row plates: upper row = [0, 5/12) of the height
LOWER_ROW_START = 1 / 3         #                 lower row = [1/3, 1] (rows overlap slightly)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def plate_aspect(corners: np.ndarray) -> float:
    tl, tr, br, bl = corners
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(w / max(h, 1e-6))


def merge_double_layer(img: np.ndarray, split=(UPPER_ROW_END, LOWER_ROW_START)) -> np.ndarray:
    """Place the upper row of a two-row plate to the left of the lower row."""
    h, w = img.shape[:2]
    upper = img[: max(1, int(h * split[0]))]
    lower = img[min(h - 1, int(h * split[1])):]
    upper = cv2.resize(upper, (lower.shape[1], lower.shape[0]))
    return np.hstack([upper, lower])


def rectify(image: np.ndarray, corners: np.ndarray, out_w: int = REC_W, out_h: int = REC_H,
            double_ratio: float = DOUBLE_LAYER_RATIO, split=(UPPER_ROW_END, LOWER_ROW_START)) -> np.ndarray:
    """Perspective-warp the plate quadrilateral to the recogniser input size."""
    corners = np.asarray(corners, np.float32)
    double = plate_aspect(corners) < double_ratio
    # warp double-layer plates to a taller canvas first so both rows keep resolution
    wh = (out_w, out_w // 2) if double else (out_w, out_h)
    dst = np.array([[0, 0], [wh[0] - 1, 0], [wh[0] - 1, wh[1] - 1], [0, wh[1] - 1]], np.float32)
    m = cv2.getPerspectiveTransform(corners, dst)
    plate = cv2.warpPerspective(image, m, wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if double:
        plate = merge_double_layer(plate, split)
    if plate.shape[1] != out_w or plate.shape[0] != out_h:
        plate = cv2.resize(plate, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return plate


def to_tensor_layout(plate_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> float32 3xHxW in [-1, 1] (same normalisation as C++)."""
    x = plate_bgr.astype(np.float32) * (1.0 / 127.5) - 1.0
    return x.transpose(2, 0, 1)
