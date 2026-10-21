"""Synthetic Chinese licence plate generator.

Few real images cannot cover 31 provinces x 5 colours x double-row plates, so the
recogniser is pre-trained on rendered plates that follow the GA36-2018 layout and
then fine-tuned on the (few) real crops.

Every sample goes through the *same* path as inference:
render plate -> random perspective into a noisy scene -> jittered corners
(simulating detector key-point error) -> geometry.rectify() -> 168x48.

Put a real plate font into training/fonts/ (e.g. a GA36 "platech.ttf" for Chinese
and "platechar.ttf" for letters/digits) to make the renders even closer to reality;
otherwise Windows system fonts are used.
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .charset import DIGITS, LETTERS, PROVINCES
from .geometry import REC_H, REC_W, rectify

_FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
_WIN = Path("C:/Windows/Fonts")
_CN_FONTS = [_FONT_DIR / "platech.ttf", _WIN / "simhei.ttf", _WIN / "msyhbd.ttc", _WIN / "msyh.ttc",
             _WIN / "simsun.ttc", Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")]
_EN_FONTS = [_FONT_DIR / "platechar.ttf", _WIN / "ARIALNB.TTF", _WIN / "bahnschrift.ttf", _WIN / "arialbd.ttf",
             Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf")]

ALNUM = LETTERS + DIGITS


def _first_existing(paths):
    found = [p for p in paths if p.exists()]
    if not found:
        raise FileNotFoundError(f"no usable font among {paths}; copy one into {_FONT_DIR}")
    return found


@lru_cache(maxsize=None)
def _fonts():
    cn = [str(p) for p in _first_existing(_CN_FONTS)][:2]
    en = [str(p) for p in _first_existing(_EN_FONTS)][:2]
    return cn, en


@lru_cache(maxsize=4096)
def _glyph(ch: str, font_path: str) -> np.ndarray:
    """Tight grayscale mask (0..255) of one character."""
    font = ImageFont.truetype(font_path, 160)
    img = Image.new("L", (240, 240), 0)
    ImageDraw.Draw(img).text((20, 10), ch, fill=255, font=font)
    a = np.array(img)
    ys, xs = np.nonzero(a > 40)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _is_cn(ch: str) -> bool:
    return ord(ch) > 127


def _draw_char(canvas: np.ndarray, ch: str, box, color, rng: random.Random):
    """Stretch the glyph of ``ch`` into box=(x, y, w, h) and alpha-blend it."""
    cn_fonts, en_fonts = _fonts()
    fp = rng.choice(cn_fonts if _is_cn(ch) else en_fonts)
    g = _glyph(ch, fp)
    x, y, w, h = [int(round(v)) for v in box]
    if ch in "1":  # keep the digit one thin instead of stretching it into a bar
        nw = max(2, int(w * 0.35))
        x += (w - nw) // 2
        w = nw
    m = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
    roi = canvas[y:y + h, x:x + w]
    roi[:] = roi * (1 - m) + np.array(color, np.float32) * m


# ------------------------------------------------------------------ plate texts
def _prov(rng):
    return rng.choice(PROVINCES)


def _tail(rng, n, max_letters=2):
    s = [rng.choice(DIGITS) for _ in range(n)]
    for i in rng.sample(range(n), rng.randint(0, max_letters)):
        s[i] = rng.choice(LETTERS)
    return "".join(s)


def random_plate(rng: random.Random):
    """Returns (text, colour, style) where style selects the physical layout."""
    r = rng.random()
    p, L = _prov(rng), rng.choice(LETTERS)
    if r < 0.38:
        return p + L + _tail(rng, 5), "blue", "single7"
    if r < 0.58:
        if rng.random() < 0.8:   # small new-energy car: 3rd char D/F(+new letters)
            t = p + L + rng.choice("DFDFDFABCEGHJK") + rng.choice(ALNUM) + "".join(rng.choice(DIGITS) for _ in range(4))
        else:                    # large new-energy vehicle: suffix D/F
            t = p + L + "".join(rng.choice(DIGITS) for _ in range(5)) + rng.choice("DF")
        return t, "green", "single8"
    if r < 0.68:
        suffix = rng.random()
        if suffix < 0.15:
            return p + L + _tail(rng, 4) + "学", "yellow", "single7"
        if suffix < 0.30:
            return p + L + _tail(rng, 4) + "挂", "yellow", "single7"
        return p + L + _tail(rng, 5), "yellow", "single7"
    if r < 0.80:
        t = p + L + (_tail(rng, 4) + "挂" if rng.random() < 0.2 else _tail(rng, 5))
        return t, "yellow", "double"
    if r < 0.90:
        return p + L + _tail(rng, 4) + "警", "white", "single7"
    k = rng.random()
    if k < 0.6:
        return "粤Z" + _tail(rng, 4) + rng.choice("港澳"), "black", "single7"
    if k < 0.8:
        return p + L + _tail(rng, 4) + "领", "black", "single7"
    return "使" + "".join(rng.choice(DIGITS) for _ in range(6)), "black", "single7"


# ------------------------------------------------------------------ rendering
_BG = {  # RGB background, RGB text
    "blue": ((12, 60, 168), (245, 245, 245)),
    "yellow": ((238, 186, 22), (15, 15, 15)),
    "white": ((236, 236, 236), (15, 15, 15)),
    "black": ((18, 18, 20), (240, 240, 240)),
}


def _jitter_color(c, rng, amt=25):
    return tuple(int(np.clip(v + rng.randint(-amt, amt), 0, 255)) for v in c)


def render_plate(text: str, color: str, style: str, rng: random.Random) -> np.ndarray:
    """Render a frontal plate as RGB uint8 at 1 px / mm."""
    if style == "single8":
        W, H = 480, 140
        top = np.array([236, 250, 238], np.float32)
        bot = np.array(_jitter_color((70, 190, 100), rng), np.float32)
        t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        img = np.broadcast_to(top * (1 - t) + bot * t, (H, W, 3)).copy()
        fg = (10, 10, 10)
        xs = [15, 67, 162, 214, 266, 318, 370, 422]
        boxes = [(x, 25, 43, 90) for x in xs]
        dots = [(129, 70)]
    elif style == "double":
        W, H = 440, 220
        bg, fg = _BG[color]
        img = np.full((H, W, 3), _jitter_color(bg, rng), np.float32)
        boxes = [(110, 15, 80, 60), (250, 15, 80, 60)] + [(27.5 + i * 80, 90, 65, 110) for i in range(5)]
        dots = [(220, 45)]
    else:
        W, H = 440, 140
        bg, fg = _BG[color]
        img = np.full((H, W, 3), _jitter_color(bg, rng), np.float32)
        xs = [15.5, 72.5, 151.5, 208.5, 265.5, 322.5, 379.5]
        boxes = [(x, 25, 45, 90) for x in xs]
        dots = [(134.5, 70)]
    fg = _jitter_color(fg, rng, 15)

    # frame line close to the edge
    cv2.rectangle(img, (3, 3), (W - 4, H - 4), fg, rng.choice([1, 2, 3]))
    # the separator dot between the region code and the serial
    if text[0] != "使":
        for cx, cy in dots:
            cv2.circle(img, (int(cx), int(cy)), 5, fg, -1)
    if style == "single8" and len(text) != 8:
        raise ValueError(text)
    for i, ch in enumerate(text[: len(boxes)]):
        col = (200, 20, 20) if ch == "警" else fg
        _draw_char(img, ch, boxes[i], col, rng)
    # rivets
    if rng.random() < 0.6:
        for cx in (W * 0.25, W * 0.75):
            cv2.circle(img, (int(cx), 12), 5, _jitter_color((150, 150, 150), rng), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _motion_blur(img, rng):
    k = rng.choice([3, 5, 7])
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def photometric(img_bgr: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    """Camera-like degradation shared by synthetic and real samples."""
    img = img_bgr.astype(np.float32)
    # brightness / contrast / gamma
    img = img * rng.uniform(1 - 0.45 * strength, 1 + 0.3 * strength) + rng.uniform(-35, 35) * strength
    img = np.clip(img, 0, 255)
    g = rng.uniform(1 - 0.4 * strength, 1 + 0.5 * strength)
    img = 255.0 * (img / 255.0) ** g
    # illumination gradient / shadow
    if rng.random() < 0.35 * strength:
        h, w = img.shape[:2]
        ramp = np.linspace(rng.uniform(0.4, 1), rng.uniform(0.8, 1.2), w, dtype=np.float32)
        img *= ramp[None, :, None] if rng.random() < 0.5 else ramp[::-1][None, :, None]
    # colour cast
    img *= np.array([rng.uniform(0.85, 1.15) for _ in range(3)], np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    r = rng.random()
    if r < 0.25 * strength:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.6))
    elif r < 0.40 * strength:
        img = _motion_blur(img, rng)
    if rng.random() < 0.4 * strength:
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 14), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5 * strength:
        q = rng.randint(25, 90)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img


def jitter_corners(corners: np.ndarray, rng: random.Random, amount: float = 0.06) -> np.ndarray:
    """Perturb 4 corners by up to ``amount`` of the plate size (detector key-point noise)."""
    c = corners.astype(np.float32).copy()
    w = np.linalg.norm(c[1] - c[0])
    h = np.linalg.norm(c[3] - c[0])
    for i in range(4):
        c[i, 0] += rng.uniform(-amount, amount) * w * 0.5
        c[i, 1] += rng.uniform(-amount, amount) * h * 1.2
    return c


def synth_sample(rng: random.Random):
    """One training sample: (168x48 BGR uint8, text, colour)."""
    text, color, style = random_plate(rng)
    plate = cv2.cvtColor(render_plate(text, color, style, rng), cv2.COLOR_RGB2BGR)
    ph, pw = plate.shape[:2]

    # place into a scene: random plate size + perspective
    target_w = rng.uniform(70, 260)
    s = target_w / pw
    cw, ch = int(target_w * 1.8), int(ph * s * 3.0)
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    ox, oy = (cw - target_w) / 2, (ch - ph * s) / 2
    dst = src * s + np.array([ox, oy], np.float32)
    k = 0.12
    dst += np.array([[rng.uniform(-k, k) * target_w, rng.uniform(-k, k) * ph * s] for _ in range(4)], np.float32)
    ang = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    center = dst.mean(0)
    dst = (dst - center) @ rot.T + center

    bg = np.full((ch, cw, 3), [rng.randint(0, 255) for _ in range(3)], np.uint8)
    bg = cv2.add(bg, np.random.default_rng(rng.randint(0, 2**31)).integers(0, 60, bg.shape, dtype=np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, m, (cw, ch))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), m, (cw, ch))
    scene = np.where(mask[..., None] > 127, warped, bg)
    scene = photometric(scene, rng)

    out = rectify(scene, jitter_corners(dst, rng), REC_W, REC_H)
    return out, text, color


if __name__ == "__main__":  # quick visual check: python -m alpr.synth
    rng = random.Random(0)
    tiles = [synth_sample(rng)[0] for _ in range(24)]
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 24, 4)])
    cv2.imwrite("synth_preview.jpg", grid)
    print("wrote synth_preview.jpg")
