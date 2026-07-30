"""Optional OCR of axis tick labels (needs the Tesseract engine installed).

This is the one piece that a self-contained tool cannot do reliably on its
own: reading the *numbers* printed on a raster figure's axes. A real OCR engine
does it well, so this module uses ``pytesseract`` **if, and only if,** the
Tesseract binary is actually present on the system. When it is not, every entry
point degrades to "unavailable" and the caller falls back to the reliable
manual flow (auto-detected tick positions + zoomed label crops the user reads).

Install to enable (user side, one-time):
  * Windows:  install "Tesseract at UB Mannheim", then it's on PATH, or set
              pytesseract.pytesseract.tesseract_cmd to the .exe path.
  * conda:    conda install -c conda-forge tesseract
  * apt/brew: apt-get install tesseract-ocr  /  brew install tesseract

Safety: OCR results are only ever *applied* when all required labels parse to
numbers; a partial/low-confidence read returns None so a wrong tick can never
silently corrupt a calibration. Even when applied, the calibration is flagged
"ocr (verify)".
"""
from __future__ import annotations

import os
import re
import shutil

import numpy as np

_AVAILABLE: bool | None = None
_NUM = re.compile(r"[+-]?\d*\.?\d+")

# Common Tesseract binary locations to probe when it isn't on PATH (the
# UB-Mannheim Windows installer does not add itself to PATH by default).
_CANDIDATE_BINARIES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
]


def _locate_binary() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for path in _CANDIDATE_BINARIES:
        if os.path.isfile(path):
            return path
    return None


def available() -> bool:
    """True if pytesseract is importable AND the Tesseract binary is present.

    Also points pytesseract at the binary when it is installed but not on PATH
    (typical of the Windows installer), so OCR works with no manual setup.
    """
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import pytesseract
            binary = _locate_binary()
            if binary and binary.lower() != "tesseract":
                pytesseract.pytesseract.tesseract_cmd = binary
            pytesseract.get_tesseract_version()
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def _preprocess(crop_rgb: np.ndarray, upscale: int = 4) -> "np.ndarray":
    import cv2
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    # upscale + Otsu threshold to black text on white — what Tesseract likes
    big = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() < 127:          # ensure dark text on light background
        th = 255 - th
    return cv2.copyMakeBorder(th, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def _leading_minus(crop_rgb: np.ndarray) -> bool:
    """Detect a leading minus sign that Tesseract commonly drops.

    A minus is a short, wide, vertically-centred ink mark to the left of the
    digits — distinct from a digit (nearly full height) or a decimal point
    (small and low). We take the left-most connected ink component and check
    that geometry, so a decimal point or a digit stroke won't be misread as a
    sign.
    """
    import cv2
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return False
    H = crop_rgb.shape[0]
    comps = [stats[i] for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] >= 3]
    if not comps:
        return False
    left = min(comps, key=lambda s: s[cv2.CC_STAT_LEFT])
    x, y, w, h, _ = left
    cy = y + h / 2
    return (w >= 1.4 * h                     # wider than tall
            and h < 0.45 * H                 # short (not a digit)
            and 0.25 * H < cy < 0.75 * H)    # vertically centred (not a '.')


def read_number(crop_rgb: np.ndarray) -> tuple[float | None, float]:
    """Read a single numeric label from a tick-label crop.

    Returns ``(value, confidence)``; ``value`` is None when OCR is unavailable
    or the text does not parse to exactly one number. ``confidence`` is
    Tesseract's mean word confidence in 0..1 (0 when unavailable).
    """
    if not available() or crop_rgb is None or crop_rgb.size == 0:
        return None, 0.0
    import pytesseract
    img = _preprocess(crop_rgb)
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789.-"
    try:
        data = pytesseract.image_to_data(img, config=cfg,
                                         output_type=pytesseract.Output.DICT)
    except Exception:
        return None, 0.0
    texts, confs = [], []
    for t, c in zip(data.get("text", []), data.get("conf", [])):
        t = t.strip()
        if t:
            texts.append(t)
            try:
                confs.append(max(0.0, float(c)) / 100.0)
            except ValueError:
                pass
    joined = _normalize_minus("".join(texts))
    nums = _NUM.findall(joined)
    if len(nums) != 1:
        return None, (float(np.mean(confs)) if confs else 0.0)
    value = nums[0]
    # Recover a leading minus if Tesseract dropped it (very common on axis
    # labels). Detected geometrically from the image, so it is reliable even
    # when the character-level OCR confidence for the sign is poor.
    conf = float(np.mean(confs)) if confs else 0.0
    try:
        val = float(value)
    except ValueError:
        return None, conf
    if val >= 0 and not value.startswith("-"):
        try:
            if _leading_minus(crop_rgb):
                val = -val
                conf = max(conf, 0.75)   # geometry is independent evidence
        except Exception:
            pass
    return val, conf


def _normalize_minus(s: str) -> str:
    for m in ("−", "–", "—"):
        s = s.replace(m, "-")
    return s


def read_axis_values(crops: dict, *, min_conf: float = 0.4
                     ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Read the four outer tick labels into ``(x_lo, x_hi)`` and ``(y_lo, y_hi)``.

    ``crops`` is the dict from :func:`plotframe.crop_tick_labels`
    (keys x_lo/x_hi/y_lo/y_hi). Returns ``(x_pair, y_pair)``; a pair is None
    unless *both* of its labels read as numbers above ``min_conf`` — so an axis
    is calibrated by OCR only when fully and confidently read.
    """
    def _pair(k_lo, k_hi):
        if k_lo not in crops or k_hi not in crops:
            return None
        v_lo, c_lo = read_number(crops[k_lo])
        v_hi, c_hi = read_number(crops[k_hi])
        if v_lo is None or v_hi is None or v_lo == v_hi:
            return None
        if min(c_lo, c_hi) < min_conf:
            return None
        return (v_lo, v_hi)

    return _pair("x_lo", "x_hi"), _pair("y_lo", "y_hi")
