"""Automatic axis calibration from the PDF *text layer* (vector figures).

Many vector figures (matplotlib output, Elsevier production PDFs, ...) keep
their axis tick labels as real, selectable text with exact positions —
``page.get_text("words")`` returns them with bounding boxes. Each numeric
label is centred on its tick, so the label row under a plot IS the x-axis
calibration and the label column left of it IS the y-axis calibration: fit a
straight line through (label centre px, label value) pairs and the full
pixel->data map falls out. No OCR, no user input.

Not every PDF qualifies: some figures (e.g. the rizo ACS paper) outline their
text into vector glyph paths, leaving nothing in the text layer — detection
then simply fails and the caller falls back to a manual calibration JSON or
normalised output. All acceptance checks below err on the side of returning
nothing rather than a wrong calibration, because a silently-wrong axis is the
worst possible failure mode for digitized data.

Geometry conventions match the rest of the package: PDF points, y down.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz
import numpy as np

from .calibrate import Calibration

# numeric tick label, after normalising unicode minus signs. Anchored, short.
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")
# characters that render as a minus in figures
_MINUSES = {"−", "–", "—"}


def _normalize(text: str) -> str:
    for m in _MINUSES:
        text = text.replace(m, "-")
    return text.strip()


@dataclass
class TickLabel:
    value: float
    cx: float           # bbox centre x (PDF points)
    cy: float           # bbox centre y
    bbox: tuple[float, float, float, float]
    height: float = field(init=False)

    def __post_init__(self):
        self.height = self.bbox[3] - self.bbox[1]


@dataclass
class AxisLabelSet:
    """One detected row (x-axis) or column (y-axis) of tick labels."""

    orientation: str                 # "x" | "y"
    labels: list[TickLabel]
    slope: float                     # data units per PDF point
    intercept: float
    residual: float                  # max |fit error| / value range
    title: str = ""                  # nearby axis-title text, best effort

    @property
    def pixel_span(self) -> tuple[float, float]:
        pos = [l.cx if self.orientation == "x" else l.cy for l in self.labels]
        return min(pos), max(pos)

    @property
    def line_coord(self) -> float:
        """The row's y (for x-axis) or column's x (for y-axis)."""
        if self.orientation == "x":
            return float(np.median([l.cy for l in self.labels]))
        return float(np.median([l.cx for l in self.labels]))


def _numeric_words(page) -> list[TickLabel]:
    out = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, raw = w[0], w[1], w[2], w[3], w[4]
        t = _normalize(raw)
        if len(t) > 8 or not _NUM.match(t):
            continue
        out.append(TickLabel(value=float(t), cx=(x0 + x1) / 2, cy=(y0 + y1) / 2,
                             bbox=(x0, y0, x1, y1)))
    return out


def _cluster(vals_words: list[tuple[float, TickLabel]], tol: float) -> list[list[TickLabel]]:
    """Group words whose key coordinate is within ``tol`` of the cluster mean."""
    groups: list[list[tuple[float, TickLabel]]] = []
    for key, w in sorted(vals_words, key=lambda kw: kw[0]):
        placed = False
        for g in groups:
            if abs(key - np.mean([k for k, _ in g])) <= tol:
                g.append((key, w))
                placed = True
                break
        if not placed:
            groups.append([(key, w)])
    return [[w for _, w in g] for g in groups]


def _fit_axis(labels: list[TickLabel], orientation: str,
              max_residual: float = 0.03) -> AxisLabelSet | None:
    """Least-squares fit value ~ pixel for one candidate label group.

    Accepts only if: >=3 labels, values not all equal, pixel positions strictly
    monotonic once sorted by value (an axis can't fold back), and the worst
    fit error is under ``max_residual`` of the value range. Real tick labels
    sit on a nearly perfect line; body text that happens to contain numbers
    essentially never does.
    """
    if len(labels) < 3:
        return None
    labels = sorted(labels, key=lambda l: l.value)
    vals = np.array([l.value for l in labels])
    # tick labels never repeat a value on one axis; a duplicate means we are
    # looking at caption/body text that happens to contain numbers.
    if len(np.unique(vals)) != len(vals):
        return None
    pos = np.array([l.cx if orientation == "x" else l.cy for l in labels])
    if np.ptp(vals) == 0 or np.ptp(pos) < 1e-6:
        return None
    d = np.diff(pos)
    if not (np.all(d > 0) or np.all(d < 0)):
        return None
    A = np.column_stack([pos, np.ones_like(pos)])
    (slope, intercept), *_ = np.linalg.lstsq(A, vals, rcond=None)
    fit = slope * pos + intercept
    residual = float(np.max(np.abs(fit - vals)) / np.ptp(vals))
    if residual > max_residual:
        return None
    return AxisLabelSet(orientation, labels, float(slope), float(intercept), residual)


def find_axis_label_sets(pdf_path: str, page_number: int) -> list[AxisLabelSet]:
    """All plausible x-axis rows and y-axis columns of tick labels on a page."""
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    words = _numeric_words(page)
    all_words = page.get_text("words")
    doc.close()
    if not words:
        return []

    med_h = float(np.median([w.height for w in words])) or 8.0
    sets: list[AxisLabelSet] = []

    # x-axis candidates: rows (same cy)
    for group in _cluster([(w.cy, w) for w in words], tol=med_h * 0.6):
        fitted = _fit_axis(group, "x")
        if fitted:
            sets.append(fitted)
    # y-axis candidates: columns (same cx). Tick labels are right-aligned to
    # the axis so their *right* edge lines up better than the centre; cluster
    # on the right edge but keep centre for the fit position.
    for group in _cluster([(w.bbox[2], w) for w in words], tol=med_h * 0.9):
        fitted = _fit_axis(group, "y")
        if fitted:
            sets.append(fitted)

    _attach_titles(sets, all_words)
    return sets


def _attach_titles(sets: list[AxisLabelSet], all_words) -> None:
    """Best-effort: join non-numeric text just beyond each label set into a
    title string (e.g. "E vs. RHE / V"). Purely informational metadata."""
    for s in sets:
        lo, hi = s.pixel_span
        line = s.line_coord
        parts = []
        for w in all_words:
            x0, y0, x1, y1, raw = w[0], w[1], w[2], w[3], w[4]
            t = _normalize(raw)
            if _NUM.match(t):
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if s.orientation == "x":
                # within the row's x span, up to ~3 line-heights below it
                if lo - 20 <= cx <= hi + 20 and line + 5 < cy <= line + 40:
                    parts.append((cx, t))
            else:
                # left of the column, vertically within its span. Rotated
                # y-titles read bottom-to-top, i.e. descending cy = reading
                # order, so sort those descending.
                if lo - 20 <= cy <= hi + 20 and line - 45 <= cx < line - 5:
                    parts.append((-cy, t))
        s.title = " ".join(t for _, t in sorted(parts))[:60]


def match_calibration(
    sets: list[AxisLabelSet],
    bbox: tuple[float, float, float, float],
    *,
    max_gap: float = 60.0,
) -> Calibration | None:
    """Pick the label row below and column left of ``bbox`` -> Calibration.

    ``bbox`` is the curve group's extent (x0, y0, x1, y1) in PDF points. The
    x-axis labels for that plot sit slightly below its bottom edge and overlap
    it horizontally; the y-axis labels sit slightly left of its left edge and
    overlap vertically. Nearest qualifying set per axis wins — this per-curve
    matching handles multiple independent figures on one page for free.
    """
    x0, y0, x1, y1 = bbox

    def _overlap(alo, ahi, blo, bhi):
        lo, hi = max(alo, blo), min(ahi, bhi)
        return max(0.0, hi - lo) / max(1e-9, min(ahi - alo, bhi - blo))

    best_x, best_y = None, None
    for s in sets:
        lo, hi = s.pixel_span
        if s.orientation == "x":
            gap = s.line_coord - y1
            if -5 <= gap <= max_gap and _overlap(lo, hi, x0, x1) >= 0.5:
                if best_x is None or gap < best_x[0]:
                    best_x = (gap, s)
        else:
            gap = x0 - s.line_coord
            if -5 <= gap <= max_gap and _overlap(lo, hi, y0, y1) >= 0.5:
                if best_y is None or gap < best_y[0]:
                    best_y = (gap, s)
    if best_x is None or best_y is None:
        return None
    sx, sy = best_x[1], best_y[1]

    # Express the two linear fits as two-anchor Calibrations at the bbox edges.
    def _at(s: AxisLabelSet, p: float) -> float:
        return s.slope * p + s.intercept

    return Calibration(
        x1=x0, ex1=_at(sx, x0), x2=x1, ex2=_at(sx, x1),
        y1=y0, jy1=_at(sy, y0), y2=y1, jy2=_at(sy, y1),
        x_unit=(sx.title or "auto (unverified)"),
        y_unit=(sy.title or "auto (unverified)"),
    )


def autocalibrate(pdf_path: str, page_number: int,
                  bbox: tuple[float, float, float, float]) -> Calibration | None:
    """Convenience wrapper: label sets for the page, matched to one bbox."""
    sets = find_axis_label_sets(pdf_path, page_number)
    if not sets:
        return None
    return match_calibration(sets, bbox)


# --------------------------------------------------------------------------- #
# Assisted calibration for vector pages whose figure text is outlined (no
# text layer): render the page and reuse the raster branch's frame + tick
# detection, so the user supplies only the two outermost tick VALUES per axis
# (same UX as raster figures) instead of hand-building a JSON of pixel anchors.
# --------------------------------------------------------------------------- #
def detect_ticks_for_bbox(pdf_path: str, page_number: int,
                          bbox: tuple[float, float, float, float],
                          *, zoom: float = 3.0) -> dict | None:
    """Frame + tick pixel positions for the plot containing ``bbox``.

    Renders the page at ``zoom``, finds every axes frame, picks the one that
    best overlaps the curves' bbox, and detects tick marks along its axes.
    Returns ``{image, frame_px, ticks, zoom}`` (render-pixel space) or None.
    """
    import cv2

    from .ingest import render_page
    from .raster_extract import detect_all_frames, detect_axis_ticks

    img = render_page(pdf_path, page_number, zoom=zoom)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frames = detect_all_frames(gray)
    if not frames:
        return None

    bx0, by0, bx1, by1 = (v * zoom for v in bbox)

    def _iou(f):
        fx0, fy0, fx1, fy1 = f
        ix0, iy0 = max(fx0, bx0), max(fy0, by0)
        ix1, iy1 = min(fx1, bx1), min(fy1, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = ((fx1 - fx0) * (fy1 - fy0) + (bx1 - bx0) * (by1 - by0) - inter) or 1
        return inter / union

    frame = max(frames, key=_iou)
    if _iou(frame) < 0.2:
        return None
    ticks = detect_axis_ticks(gray, frame)
    if len(ticks["x_ticks"]) < 2 or len(ticks["y_ticks"]) < 2:
        return None
    return {"image": img, "frame_px": frame, "ticks": ticks, "zoom": zoom}


def assisted_tick_calibration(detection: dict,
                              x_vals: tuple[float, float],
                              y_vals: tuple[float, float],
                              *, x_unit: str = "", y_unit: str = "") -> Calibration:
    """Calibration from detected tick positions + user-supplied outer values.

    ``detection`` comes from :func:`detect_ticks_for_bbox`; positions are
    render pixels and get divided by the zoom so the Calibration operates in
    PDF points, matching the vector pipeline's coordinates.
    """
    z = detection["zoom"]
    xt = detection["ticks"]["x_ticks"]
    yt = detection["ticks"]["y_ticks"]
    return Calibration(
        x1=xt[0] / z, ex1=x_vals[0], x2=xt[-1] / z, ex2=x_vals[1],
        y1=yt[0] / z, jy1=y_vals[0], y2=yt[-1] / z, jy2=y_vals[1],
        x_unit=x_unit, y_unit=y_unit,
    )
