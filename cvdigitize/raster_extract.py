"""Raster (scanned/rasterized figure) curve extraction (M2).

Used when a figure has no usable vector geometry — e.g. the whole plot was
flattened into a single embedded bitmap when the PDF was produced (common
when figure panels are assembled/exported as one image). We recover the curve
directly from pixels instead:

1. ``detect_frame_bbox``   find the plot's axes box via long straight lines
   (Hough transform) — robust even when the curve/fill/ticks are the same
   dark colour as the frame, because the frame lines are far longer than any
   of those.
2. ``mask_dark_curve``     isolate the curve by *darkness* (HSV value), not
   colour saturation — anti-aliasing blends a black line into a coloured fill
   with elevated saturation, so saturation-based masks fail; brightness does
   not have this problem.
3. ``largest_component``   drop tick marks / arrows / stray marks, keeping the
   single largest connected blob (the curve is always the biggest structure).
4. ``skeletonize_curve``   thin the blob to a 1-pixel-wide trace.
5. ``skeleton_to_polyline`` prune short spurs (tick-mark stubs touching the
   curve) and walk the remaining pixel graph into an ordered polyline, exactly
   analogous to ``postprocess.order_curve`` for the vector branch.

Output coordinates are in PDF-point units (dividing by the render zoom and
adding the crop offset), so the raster branch plugs into the same
``calibrate.Calibration`` / CLI pipeline as the vector branch.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .ingest import render_page


@dataclass
class ImageRegion:
    """An embedded raster image on a page, in PDF-point coordinates."""

    xref: int
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1) PDF points


def _merge_tiling(regions: list["ImageRegion"], gap: float = 4.0) -> list["ImageRegion"]:
    """Union image rects that tile one figure (PDF producers often slice a
    single figure into several abutting strips). Two rects merge when they
    share one axis span (within ``gap``) and touch along the other."""
    boxes = [list(r.bbox) for r in regions]
    xrefs = [r.xref for r in regions]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                same_x = abs(a[0] - b[0]) <= gap and abs(a[2] - b[2]) <= gap
                same_y = abs(a[1] - b[1]) <= gap and abs(a[3] - b[3]) <= gap
                touch_v = min(abs(a[3] - b[1]), abs(b[3] - a[1])) <= gap
                touch_h = min(abs(a[2] - b[0]), abs(b[2] - a[0])) <= gap
                if (same_x and touch_v) or (same_y and touch_h):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j], xrefs[j]
                    changed = True
                    break
            if changed:
                break
    return [ImageRegion(x, tuple(b)) for x, b in zip(xrefs, boxes)]


def find_image_regions(pdf_path: str, page_number: int,
                       *, min_area_frac: float = 0.05) -> list[ImageRegion]:
    """Embedded images on a page (tiling strips merged), largest first."""
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    regions = []
    for img in page.get_images(full=True):
        xref = img[0]
        for r in page.get_image_rects(xref):
            if abs(r.width * r.height) / page_area >= 0.01:
                regions.append(ImageRegion(xref, (r.x0, r.y0, r.x1, r.y1)))
    doc.close()
    regions = _merge_tiling(regions)
    regions = [rg for rg in regions
               if ((rg.bbox[2] - rg.bbox[0]) * (rg.bbox[3] - rg.bbox[1])
                   / page_area) >= min_area_frac]
    regions.sort(key=lambda rg: (rg.bbox[2] - rg.bbox[0]) * (rg.bbox[3] - rg.bbox[1]),
                reverse=True)
    return regions


def render_region(pdf_path: str, page_number: int,
                  bbox: tuple[float, float, float, float],
                  zoom: float = 4.0) -> np.ndarray:
    """Render just ``bbox`` (PDF points) of a page to an RGB array at ``zoom``."""
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    rect = fitz.Rect(*bbox)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    doc.close()
    return img[:, :, :3].copy()


# --------------------------------------------------------------------------- #
# 1. Axes-frame detection
# --------------------------------------------------------------------------- #
def detect_frame_bbox(gray: np.ndarray, *, dark_thresh: int = 220,
                      min_len_frac: float = 0.35,
                      min_size_frac: float = 0.15) -> tuple[int, int, int, int] | None:
    """Find the plot's rectangular axes box via long straight Hough lines.

    Works even when the curve, ticks and frame share the same dark colour: the
    frame's four border lines are by far the longest straight dark segments in
    the image, so filtering by minimum length isolates them from the (locally
    curvy, shorter) data curve. Returns ``(x0, y0, x1, y1)`` in pixel
    coordinates, or ``None`` if no confident frame was found.

    A genuine frame's four sides come from *matching* long lines, but each
    side's own minimum-length check doesn't guarantee the sides are far apart
    from each other: two near-duplicate long horizontal lines (anti-aliasing,
    or an unrelated long-straight stretch of curve/ink) can sit only a few
    rows apart and still each individually pass ``min_len_frac``, yielding a
    box too thin to be a real frame -- and too thin for ``frame_inset_px`` to
    safely inset without producing an empty interior slice downstream. Reject
    that case here (return None, so callers fall back to a margin box)
    instead of handing out a box no caller can safely use.
    """
    h, w = gray.shape
    dark = (gray < dark_thresh).astype(np.uint8) * 255
    minlen = int(min_len_frac * min(h, w))
    lines = cv2.HoughLinesP(dark, 1, np.pi / 180, threshold=80,
                            minLineLength=minlen, maxLineGap=8)
    if lines is None:
        return None

    horiz, vert = [], []
    for l in lines:
        x1, y1, x2, y2 = l.reshape(-1)
        if abs(y1 - y2) < 3 and abs(x1 - x2) >= minlen:
            horiz.append((int(round((y1 + y2) / 2)), min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) < 3 and abs(y1 - y2) >= minlen:
            vert.append((int(round((x1 + x2) / 2)), min(y1, y2), max(y1, y2)))
    if not horiz or not vert:
        return None

    # keep only lines close to the longest span found (the true frame edges)
    def _filter_longest(cands, lo_idx, hi_idx):
        spans = [c[hi_idx] - c[lo_idx] for c in cands]
        best = max(spans)
        return [c for c, s in zip(cands, spans) if s >= 0.8 * best]

    horiz = _filter_longest(horiz, 1, 2)
    vert = _filter_longest(vert, 1, 2)

    top = min(c[0] for c in horiz)
    bottom = max(c[0] for c in horiz)
    left = min(c[0] for c in vert)
    right = max(c[0] for c in vert)
    if right <= left or bottom <= top:
        return None
    if (right - left) < min_size_frac * w or (bottom - top) < min_size_frac * h:
        return None
    return left, top, right, bottom


def _long_runs(arr: np.ndarray, minlen: int) -> list[tuple[int, int]]:
    """All ``(start, end)`` (end exclusive) runs of ``True`` at least ``minlen`` long."""
    padded = np.r_[0, arr.astype(np.int8), 0]
    diffs = np.diff(padded)
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= minlen]


def _line_segments(dark: np.ndarray, minlen: int, *, horizontal: bool
                   ) -> list[tuple[int, int, int]]:
    """Every long dark run in each row (or column): ``(position, span0, span1)``."""
    n = dark.shape[0] if horizontal else dark.shape[1]
    segs = []
    for i in range(n):
        line = dark[i, :] if horizontal else dark[:, i]
        for s, e in _long_runs(line, minlen):
            segs.append((i, s, e))
    return segs


def _merge_collinear_segments(segs: list[tuple[int, int, int]], *, pos_tol: int = 2,
                              iou_thresh: float = 0.5) -> list[tuple[int, int, int]]:
    """Merge segments at adjacent positions into one line, by span overlap.

    Anti-aliasing spreads a single border line over a few adjacent rows/cols;
    those must merge into one line. An unrelated feature that happens to sit a
    row or two away (e.g. a different panel's content) must *not* merge just
    because it is nearby — so segments only merge when their spans actually
    overlap substantially (high IoU), not merely by position proximity.
    """
    segs = sorted(segs)
    used = [False] * len(segs)
    lines = []
    for i, (p, a0, a1) in enumerate(segs):
        if used[i]:
            continue
        group = [(p, a0, a1)]
        used[i] = True
        cur_p = p
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            p2, b0, b1 = segs[j]
            if p2 - cur_p > pos_tol:
                break
            inter = max(0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0)
            if union > 0 and inter / union >= iou_thresh:
                group.append((p2, b0, b1))
                used[j] = True
                cur_p = p2
        ps = [g[0] for g in group]
        a0s = [g[1] for g in group]
        a1s = [g[2] for g in group]
        lines.append((int(round(sum(ps) / len(ps))), int(np.median(a0s)), int(np.median(a1s))))
    return lines


def detect_all_frames(gray: np.ndarray, *, dark_thresh: int = 220,
                      min_len_frac: float = 0.08,
                      min_size_frac: float = 0.05,
                      span_tol_frac: float = 0.0) -> list[tuple[int, int, int, int]]:
    """Find every rectangular axes box in a (possibly multi-panel) raster image.

    Candidate border lines are found by direct run-length scanning (every long
    contiguous dark run in each row/column), then merged across the few rows a
    single anti-aliased line spans — matched by span overlap, not just
    position, so an unrelated feature a row away never merges with a real
    border. Rectangles are formed from (top, bottom) x (left, right) line
    combinations whose spans are mutually consistent, then any rectangle that
    strictly contains another valid one is dropped (that only happens when
    collinear borders of adjacent stacked/side-by-side panels coincidentally
    validate their union as a rectangle too).
    """
    h, w = gray.shape
    dark = gray < dark_thresh
    minlen = max(15, int(min_len_frac * min(h, w)))

    hlines = _merge_collinear_segments(_line_segments(dark, minlen, horizontal=True))
    vlines = _merge_collinear_segments(_line_segments(dark, minlen, horizontal=False))
    if not hlines or not vlines:
        return []

    min_w, min_h = min_size_frac * w, min_size_frac * h
    tol_w, tol_h = span_tol_frac * w, span_tol_frac * h
    rects = []
    for ti, (top, tx0, tx1) in enumerate(hlines):
        for bottom, bx0, bx1 in hlines[ti + 1:]:
            if bottom - top < min_h:
                continue
            for li, (left, ly0, ly1) in enumerate(vlines):
                for right, ry0, ry1 in vlines[li + 1:]:
                    if right - left < min_w:
                        continue
                    if not (tx0 <= left + tol_w and tx1 >= right - tol_w):
                        continue
                    if not (bx0 <= left + tol_w and bx1 >= right - tol_w):
                        continue
                    if not (ly0 <= top + tol_h and ly1 >= bottom - tol_h):
                        continue
                    if not (ry0 <= top + tol_h and ry1 >= bottom - tol_h):
                        continue
                    rects.append((left, top, right, bottom))

    if not rects:
        # Fallback: L-shaped (despined) axes — just a left spine and a bottom
        # spine, the default matplotlib style. A horizontal line whose LEFT
        # end meets a vertical line's BOTTOM end forms the corner; the frame's
        # top/right bounds are simply the spines' far ends.
        tol = max(8, minlen // 6)
        for r, hx0, hx1 in hlines:
            for c, vy0, vy1 in vlines:
                if abs(hx0 - c) <= tol and abs(vy1 - r) <= tol:
                    if (hx1 - c) >= min_w and (r - vy0) >= min_h:
                        rects.append((c, vy0, hx1, r))

    # Collinear/abutting borders between adjacent panels can make a *union* of
    # several panels also pass the consistency test. Such a union always
    # strictly contains the smaller, true panel rectangles, so discard any
    # candidate that contains another candidate.
    def _contains(outer, inner) -> bool:
        return (outer != inner and outer[0] <= inner[0] and outer[1] <= inner[1]
                and outer[2] >= inner[2] and outer[3] >= inner[3])

    atomic = [r for r in rects if not any(_contains(r, s) for s in rects)]

    atomic.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for r in atomic:
        if not any(all(abs(a - b) <= 6 for a, b in zip(r, k)) for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: (round(r[1] / 20), r[0]))
    return kept


def _cluster_peaks(positions: np.ndarray, gap: int = 3) -> list[int]:
    """Midpoints of runs of consecutive (within ``gap``) integer positions."""
    if len(positions) == 0:
        return []
    positions = sorted(int(p) for p in positions)
    runs = [[positions[0]]]
    for p in positions[1:]:
        if p - runs[-1][-1] <= gap:
            runs[-1].append(p)
        else:
            runs.append([p])
    return [int(round(sum(r) / len(r))) for r in runs]


def _regular_subset(positions: list[int], tol_frac: float = 0.2) -> list[int]:
    """Largest subset of positions forming an (approximate) arithmetic progression.

    Axis ticks are evenly spaced; stray dark marks in the scan band (a curve
    dipping close to the border, noise specks) are not. Tries every pair as
    progression generators and keeps the one with the most on-grid inliers.
    Returns positions unchanged when fewer than 4 (nothing to vote with).
    """
    n = len(positions)
    if n < 4:
        return positions
    pos = sorted(positions)
    best: list[int] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            step = pos[j] - pos[i]
            if step <= 2:
                continue
            tol = max(2.0, tol_frac * step)
            inliers = [p for p in pos
                       if abs((p - pos[i]) - round((p - pos[i]) / step) * step) <= tol]
            if len(inliers) > len(best):
                best = inliers
    return best if len(best) >= 3 else positions


def detect_axis_ticks(gray: np.ndarray, frame_px: tuple[int, int, int, int], *,
                      dark_thresh: int = 220, tick_band: int = 6,
                      min_count: int = 3) -> dict[str, list[int]]:
    """Find tick-mark pixel positions along the bottom and left axes of a frame.

    Looks for short dark tick marks just outside the frame border (below the
    bottom edge for x-ticks, left of the left edge for y-ticks); if an axis
    yields fewer than 2 there, falls back to a band just INSIDE the border
    (inward-pointing ticks — common in ACS/print styles), where an arithmetic-
    progression vote rejects stray marks from the curve itself grazing the
    border. Positions are returned in the **same pixel space as** ``frame_px``
    (absolute in ``gray``, matching ``polyline_px`` elsewhere). This turns
    manual calibration from "find two exact pixel coordinates" into "read off
    two tick labels": pixel positions are known once ticks are detected; the
    user only supplies the two corresponding data values.
    """
    left, top, right, bottom = frame_px
    dark = gray < dark_thresh
    edge_margin = tick_band + 2  # drop positions this close to a corner (border bleed)

    def _xscan(row0, row1):
        band = dark[row0:row1, left:right]
        pos = _cluster_peaks(np.where(band.sum(axis=0) >= min_count)[0])
        return [p + left for p in pos
                if edge_margin <= p <= (right - left) - edge_margin]

    def _yscan(col0, col1):
        band = dark[top:bottom, col0:col1]
        pos = _cluster_peaks(np.where(band.sum(axis=1) >= min_count)[0])
        return [p + top for p in pos
                if edge_margin <= p <= (bottom - top) - edge_margin]

    def _majors_only(positions, axis: str) -> list[int]:
        """Keep full-depth (major) ticks; minor ticks are shorter strokes and
        carry no printed label, so they must not become calibration anchors."""
        if len(positions) < 3:
            return positions
        deep = 2 * tick_band + 4
        depths = []
        for p in positions:
            if axis == "x":
                col = dark[max(0, bottom - deep):bottom - 1, p - 1:p + 2]
                depths.append(int(col.any(axis=1).sum()))
            else:
                row = dark[p - 1:p + 2, left + 1:left + 1 + deep]
                depths.append(int(row.any(axis=0).sum()))
        mx = max(depths) or 1
        kept = [p for p, d in zip(positions, depths) if d >= 0.65 * mx]
        return kept if len(kept) >= 2 else positions

    x_positions = _xscan(bottom + 2, bottom + 2 + tick_band)
    if len(x_positions) < 2:
        x_positions = _regular_subset(
            _majors_only(_regular_subset(_xscan(bottom - 2 - tick_band, bottom - 2)), "x"))

    y_positions = _yscan(max(0, left - 2 - tick_band), left - 2)
    if len(y_positions) < 2:
        y_positions = _regular_subset(
            _majors_only(_regular_subset(_yscan(left + 2, left + 2 + tick_band)), "y"))

    return {"x_ticks": x_positions, "y_ticks": y_positions}


def crop_tick_labels(image: np.ndarray, frame_px: tuple[int, int, int, int],
                     ticks: dict[str, list[int]], *, zoom: float = 4.0
                     ) -> dict[str, np.ndarray]:
    """Crop zoomed thumbnails of the four outermost axis tick labels.

    Turns the one remaining manual step (calibration) into "read four numbers
    off zoomed crops" rather than hunting them in the full figure. Returns
    ``{'x_lo','x_hi','y_lo','y_hi'}`` -> small RGB image (any may be absent).
    Label boxes are sized generously in render-pixel units so a multi-digit,
    possibly-negative number fits.
    """
    left, top, right, bottom = frame_px
    h, w = image.shape[:2]
    # Generous boxes: label offsets from the border vary a lot between styles
    # (outward ticks push labels further out; inward ticks leave a plain gap).
    # A too-large crop is still perfectly readable; a too-small one is blank.
    lw = int(55 * zoom / 4)    # half-width of an x-label box
    lh = int(34 * zoom / 4)    # height of an x-label box
    yw = int(70 * zoom / 4)    # width of a y-label box
    yh = int(15 * zoom / 4)    # half-height of a y-label box
    out: dict[str, np.ndarray] = {}

    def _clip(y0, y1, x0, x1):
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, w and x0), min(w, x1)
        return image[y0:y1, x0:x1] if (y1 > y0 and x1 > x0) else None

    xt = ticks.get("x_ticks", [])
    if xt:
        for key, xp in (("x_lo", xt[0]), ("x_hi", xt[-1])):
            crop = _clip(bottom + 2, bottom + 2 + lh, xp - lw, xp + lw)
            if crop is not None:
                out[key] = crop
    yt = ticks.get("y_ticks", [])
    if yt:
        for key, yp in (("y_lo", yt[0]), ("y_hi", yt[-1])):
            crop = _clip(yp - yh, yp + yh, left - 2 - yw, left - 2)
            if crop is not None:
                out[key] = crop
    return out


# --------------------------------------------------------------------------- #
# Line vs. axis width (CV Studio Step 3 — don't confuse curve ink with axis ink)
# --------------------------------------------------------------------------- #
def _run_length_through(line: np.ndarray, pos: int) -> int:
    """Length of the contiguous True run in 1D boolean ``line`` that contains
    index ``pos`` (0 if ``pos`` is out of range or not True there)."""
    n = len(line)
    if pos < 0 or pos >= n or not line[pos]:
        return 0
    lo = pos
    while lo > 0 and line[lo - 1]:
        lo -= 1
    hi = pos
    while hi < n - 1 and line[hi + 1]:
        hi += 1
    return hi - lo + 1


def _axis_thickness(dark: np.ndarray, *, fixed_index: int, span: tuple[int, int],
                    along_columns: bool, n_samples: int = 40) -> float:
    """Robust thickness of a straight axis line at ``fixed_index``.

    ``along_columns=True`` measures a HORIZONTAL axis (fixed row): at each
    sampled column, extrapolate outward in both directions from that row
    until the ink ends (the first white pixel) -- ``_run_length_through``
    -- to get the local dark-run thickness. ``along_columns=False`` measures
    a VERTICAL axis (fixed column) the same way, one row at a time.

    A curve crossing the axis makes that ONE sample's run much longer (it's
    reading through the crossing curve's ink, not just the thin axis line),
    but never shorter -- so unlike a percentile (which is still pulled up
    once more than that fraction of samples land near a crossing, e.g. a CV
    with several crossings across a short span), the MINIMUM over many
    samples is safe: it only takes a single clean sample anywhere along the
    span to reveal the true axis-only thickness.
    """
    lo, hi = sorted(span)
    if hi - lo < 4:
        return 1.0
    margin = max(2, int(0.03 * (hi - lo)))
    lo2, hi2 = (lo + margin, hi - margin) if hi - lo > 2 * margin else (lo, hi)
    positions = np.unique(np.linspace(lo2, hi2, min(n_samples, hi2 - lo2 + 1)).astype(int))
    runs = []
    for p in positions:
        line = dark[:, p] if along_columns else dark[p, :]
        r = _run_length_through(line, fixed_index)
        if r > 0:
            runs.append(r)
    return float(np.min(runs)) if runs else 1.0


def frame_border_thickness(dark: np.ndarray, frame: tuple[int, int, int, int]) -> float:
    """Measure a detected frame's OWN border thickness (all 4 sides, worst
    case) -- not to be assumed equal to the plotted axis lines' thickness
    (they can differ, e.g. a thin 2px axis inside a thicker 6px outer
    frame). Used to size the inset that keeps the frame border out of the
    curve-ink mask (see :func:`~cvdigitize.studio.pipeline.autoextract_crop`).
    """
    x0, y0, x1, y1 = frame
    top = _axis_thickness(dark, fixed_index=y0, span=(x0, x1), along_columns=True)
    bottom = _axis_thickness(dark, fixed_index=y1, span=(x0, x1), along_columns=True)
    left = _axis_thickness(dark, fixed_index=x0, span=(y0, y1), along_columns=False)
    right = _axis_thickness(dark, fixed_index=x1, span=(y0, y1), along_columns=False)
    return float(max(top, bottom, left, right))


def frame_border_mask(shape: tuple, frame: tuple[int, int, int, int], thickness: float) -> np.ndarray:
    """Boolean mask covering a band of ``thickness`` px straddling all 4
    sides of ``frame`` (a ring, not a filled rectangle) -- for painting the
    frame's own border out of the ink before tracing. Belt-and-suspenders
    alongside ``_extract_from_frame``'s inset: an inset sized from a
    *measured* thickness can still be a pixel or two short of a real,
    slightly irregular/anti-aliased border, and this guarantees the border
    can never survive into the ink mask regardless of that margin.
    """
    h, w = shape[:2]
    x0, y0, x1, y1 = frame
    t = int(np.ceil(thickness))
    m = np.zeros((h, w), bool)
    m[max(0, y0 - t):min(h, y0 + t + 1), max(0, x0 - t):min(w, x1 + t + 1)] = True   # top
    m[max(0, y1 - t):min(h, y1 + t + 1), max(0, x0 - t):min(w, x1 + t + 1)] = True   # bottom
    m[max(0, y0 - t):min(h, y1 + t + 1), max(0, x0 - t):min(w, x0 + t + 1)] = True   # left
    m[max(0, y0 - t):min(h, y1 + t + 1), max(0, x1 - t):min(w, x1 + t + 1)] = True   # right
    return m


def strip_border_fill_mask(gray: np.ndarray, *, dark_thresh: int = 220,
                          max_stroke_width: float = 9.0) -> np.ndarray:
    """Boolean mask of dark regions that are both (a) connected to the
    image's own outer border and (b) too "thick" (a filled blob, not a
    stroke) to be real curve ink -- e.g. a scan's solid black page
    background bleeding into a crop's edges.

    Left alone, that fill becomes the single biggest connected component
    once ``mask_dark_curve`` runs, and downstream keep-filters that measure
    "at least 5% the size of the biggest component" (meant to keep every
    fragment of the SAME curve while dropping noise) then discard the real
    curve outright -- it's nowhere near 5% the size of a page-margin fill.
    A real curve merely touching the border at one point survives this
    (it's thin, so it fails the thickness test); only genuinely filled
    regions are removed.
    """
    dark = (gray < dark_thresh).astype(np.uint8)
    h, w = dark.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    out = np.zeros((h, w), bool)
    for i in range(1, n):
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 25 or not (x0 <= 0 or y0 <= 0 or x0 + cw >= w or y0 + ch >= h):
            continue
        comp = labels == i
        skel_len = int(skeletonize_curve(comp).sum())
        if skel_len == 0 or area / skel_len > max_stroke_width:
            out |= comp
    return out


def _calib_anchor_points(calibration: dict) -> tuple:
    try:
        return (calibration["E1"]["px"], calibration["E2"]["px"],
                calibration["j1"]["px"], calibration["j2"]["px"])
    except (KeyError, TypeError):
        raise ValueError("calibration must have E1/E2/j1/j2 anchors with 'px'")


def exclusion_mask(shape: tuple, band: dict) -> np.ndarray:
    """Rebuild the boolean pixel mask a compact ``exclusion_band`` dict describes."""
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    xa = band.get("x_axis")
    if xa:
        y0 = max(0, int(round(xa["y"] - xa["half_width"])))
        y1 = min(h, int(round(xa["y"] + xa["half_width"])) + 1)
        x0, x1 = sorted(xa["x_range"]); x0 = max(0, x0); x1 = min(w, x1 + 1)
        m[y0:y1, x0:x1] = True
    ya = band.get("y_axis")
    if ya:
        x0 = max(0, int(round(ya["x"] - ya["half_width"])))
        x1 = min(w, int(round(ya["x"] + ya["half_width"])) + 1)
        y0, y1 = sorted(ya["y_range"]); y0 = max(0, y0); y1 = min(h, y1 + 1)
        m[y0:y1, x0:x1] = True
    return m


def _median_stroke_width(mask: np.ndarray, *, min_area: int = 25) -> float:
    """Median (ink-area / skeleton-length) over the mask's significant
    components — the same thinness measure :func:`_mask_to_curve` gates on,
    here read out as a number instead of used as a pass/fail threshold."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    widths = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        skel_len = int(skeletonize_curve(labels == i).sum())
        if skel_len > 0:
            widths.append(stats[i, cv2.CC_STAT_AREA] / skel_len)
    return float(np.median(widths)) if widths else 2.0


def measure_line_and_axis_width(crop_rgb: np.ndarray, calibration: dict, *,
                                axis_width_override: float | None = None,
                                line_width_override: float | None = None) -> dict:
    """Measure curve-line width vs. axis-line width, and the axis-exclusion band.

    ``axis_width_override``/``line_width_override`` let a human correct the
    automatic measurement (via a slider) when it's wrong -- e.g. thrown off
    by a curve that crosses the axis many times -- without re-deriving the
    rest of this function's math by hand.

    The axes are the loci THROUGH the calibration points (E1/E2 pin the
    x-axis row, j1/j2 pin the y-axis column) — precise because calibration
    already located them, unlike guessing from image structure. Extraction
    can then mask ``exclusion_band`` out of the ink before tracing so a solid
    axis line is never mistaken for curve data. ``line_width`` (measured on
    the non-axis ink) drives adaptive defaults: brush ~= 1.8x, on-ink
    tolerance ~= 1x (see STUDIO_PLAN.md §8).

    An axis is drawn as one continuous straight line and routinely extends
    past the outermost calibration tick (e.g. to the plot's frame or origin)
    -- calibration only pins the axis's pixel ROW/COLUMN, not where it stops.
    So while *thickness* is only sampled between the two calibration points
    (where the axis is known to exist, clear of any surrounding frame/legend),
    the exclusion band that masks it out of extraction spans the full crop.
    """
    from .guided import _ink_mask

    e1, e2, j1, j2 = _calib_anchor_points(calibration)
    h, w = crop_rgb.shape[:2]
    x_axis_row = int(round((e1[1] + e2[1]) / 2))
    y_axis_col = int(round((j1[0] + j2[0]) / 2))
    sample_x_span = (int(round(e1[0])), int(round(e2[0])))
    sample_y_span = (int(round(j1[1])), int(round(j2[1])))

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    dark = gray < 220

    if axis_width_override is not None:
        axis_width = float(axis_width_override)
    else:
        axis_w_x = _axis_thickness(dark, fixed_index=x_axis_row, span=sample_x_span, along_columns=True)
        axis_w_y = _axis_thickness(dark, fixed_index=y_axis_col, span=sample_y_span, along_columns=False)
        axis_width = float(np.median([axis_w_x, axis_w_y]))

    half = max(1.0, axis_width / 2.0 + 1.0)   # measured half-thickness + a small margin
    exclusion_band = {
        "x_axis": {"y": x_axis_row, "half_width": half, "x_range": [0, w]},
        "y_axis": {"x": y_axis_col, "half_width": half, "y_range": [0, h]},
    }

    if line_width_override is not None:
        line_width = round(float(line_width_override), 2)
    else:
        ink = _ink_mask(crop_rgb)
        non_axis_ink = ink & ~exclusion_mask(crop_rgb.shape, exclusion_band)
        line_width = round(_median_stroke_width(non_axis_ink), 2)

    return {
        "line_width": line_width,
        "axis_width": round(axis_width, 2),
        "suggested_brush": round(max(3.0, 1.8 * line_width), 2),
        "exclusion_band": exclusion_band,
    }


# --------------------------------------------------------------------------- #
# 2-3. Isolate the curve
# --------------------------------------------------------------------------- #
def mask_dark_curve(rgb: np.ndarray, *, value_thresh: float = 0.55) -> np.ndarray:
    """Boolean mask of dark (curve/text/ticks) pixels, by brightness only.

    Deliberately ignores saturation: anti-aliasing blends a black line into a
    coloured fill, which *raises* apparent HSV saturation even though the
    pixel is visually part of the black curve. Brightness (HSV "value") does
    not have this failure mode — fill and background are both much brighter
    than an anti-aliased black edge.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    return val < value_thresh


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component (drops ticks/arrows/text)."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(np.argmax(areas)) + 1
    return labels == biggest


def skeletonize_curve(mask: np.ndarray) -> np.ndarray:
    """Thin a filled curve blob to a 1-pixel-wide skeleton."""
    return skeletonize(mask)


# --------------------------------------------------------------------------- #
# 4. Skeleton -> ordered polyline
# --------------------------------------------------------------------------- #
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_graph(skel: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    ys, xs = np.where(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for p in pts:
        y, x = p
        graph[p] = [(y + dy, x + dx) for dy, dx in _NEIGHBORS if (y + dy, x + dx) in pts]
    return graph


def _prune_spurs(graph: dict, max_spur_len: int = 15) -> dict:
    """Iteratively remove short dead-end branches (tick-mark stubs, noise)."""
    graph = {k: list(v) for k, v in graph.items()}
    changed = True
    while changed:
        changed = False
        leaves = [p for p, nb in graph.items() if len(nb) == 1]
        for leaf in leaves:
            if leaf not in graph:
                continue
            # walk the spur until a junction (degree != 2) or dead end
            path = [leaf]
            prev, cur = None, leaf
            for _ in range(max_spur_len + 1):
                nbrs = [n for n in graph.get(cur, []) if n != prev]
                if len(nbrs) != 1:
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
            else:
                continue  # spur longer than max_spur_len: keep it (real branch)
            if len(graph.get(cur, [])) <= 2 or len(path) > max_spur_len:
                continue  # reached another leaf/long path: not a stub to prune
            # remove the spur pixels (but keep the junction pixel `cur`)
            for p in path[:-1]:
                for nb in graph.get(p, []):
                    if p in graph.get(nb, []):
                        graph[nb].remove(p)
                graph.pop(p, None)
            changed = True
    return graph


def skeleton_to_polyline(skel: np.ndarray, *, max_spur_len: int = 15) -> np.ndarray:
    """Prune spurs then walk the skeleton pixel graph into an ordered (N,2) array.

    Returns points as ``(x, y)`` pixel coordinates. Handles both an open path
    (two remaining endpoints) and a closed loop (no endpoints after pruning).
    """
    graph = _skeleton_graph(skel)
    if not graph:
        return np.empty((0, 2))
    graph = _prune_spurs(graph, max_spur_len)
    if not graph:
        return np.empty((0, 2))

    endpoints = [p for p, nb in graph.items() if len(nb) == 1]
    start = endpoints[0] if endpoints else next(iter(graph))

    visited = {start}
    order = [start]
    prev, cur = None, start
    while True:
        nbrs = [n for n in graph.get(cur, []) if n != prev]
        unvisited = [n for n in nbrs if n not in visited]
        if not unvisited:
            break
        nxt = unvisited[0] if len(unvisited) == 1 else min(
            unvisited, key=lambda n: abs(n[0] - cur[0]) + abs(n[1] - cur[1]))
        visited.add(nxt)
        order.append(nxt)
        prev, cur = cur, nxt

    ys = np.array([p[0] for p in order], dtype=float)
    xs = np.array([p[1] for p in order], dtype=float)
    return np.column_stack([xs, ys])


# --------------------------------------------------------------------------- #
# Multi-colour curve splitting (raster analogue of the vector colour grouping)
# --------------------------------------------------------------------------- #
_HUE_NAMES = [(20, "red"), (45, "orange"), (70, "yellow"), (160, "green"),
              (200, "cyan"), (260, "blue"), (300, "violet"), (340, "magenta"),
              (361, "red")]


def _hue_name(hue_deg: float) -> str:
    for hi, name in _HUE_NAMES:
        if hue_deg < hi:
            return name
    return "red"


def _mask_to_curve(mask: np.ndarray, *, max_spur_len: int,
                   max_stroke_width: float = 9.0,
                   min_x_span_frac: float = 0.4) -> np.ndarray | None:
    """Components -> skeletons -> stitched polyline, with plausibility gates.

    A curve that crosses another gets overpainted at the intersections and its
    mask breaks into several pieces, so ALL significant components are kept
    (not just the largest) and their skeleton segments are stitched back into
    one traversal by the same greedy endpoint matcher the vector branch uses.

    Gates — thinness: a stroked curve's pixel area is roughly (skeleton length
    x stroke width); a filled region (gradient fill, shaded area) is far
    fatter and gets rejected. Span: a CV sweeps the potential window, so the
    x-extent must cover a decent fraction of the plot — this kills the narrow
    vertical stripes a hue slice of a smooth gradient produces.
    """
    from .postprocess import order_curve

    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    if biggest < 50:
        return None
    keep = [i + 1 for i, a in enumerate(areas)
            if a >= max(40, 0.05 * biggest)]
    kept_mask = np.isin(labels, keep)

    area = int(kept_mask.sum())
    skel = skeletonize_curve(kept_mask)
    skel_len = int(skel.sum())
    if skel_len < 30 or area / max(skel_len, 1) > max_stroke_width:
        return None

    # walk each skeleton fragment separately, then stitch
    sn, slabels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    pieces = []
    for si in range(1, sn):
        piece = skeleton_to_polyline(slabels == si, max_spur_len=max_spur_len)
        if len(piece) >= 5:
            pieces.append(piece)
    if not pieces:
        return None
    poly = order_curve(pieces) if len(pieces) > 1 else pieces[0]
    if len(poly) < 30:
        return None
    if np.ptp(poly[:, 0]) < min_x_span_frac * mask.shape[1]:
        return None
    return poly


# Junction-aware strand decomposition (fixes solid/dashed overlays and curve x
# curve / curve x axis crossings, where the naive single-walk tracer dies).
USE_STRANDS = True


def _mask_to_curves(mask: np.ndarray, *, max_stroke_width: float = 9.0,
                    min_x_span_frac: float = 0.4, drop_axes: bool = False
                    ) -> list[np.ndarray]:
    """Like :func:`_mask_to_curve` but junction-aware and may return SEVERAL
    curves (e.g. a solid curve and its dashed sibling).

    Same component-keeping and thinness gate, then the skeleton is split into
    smooth strands that pass straight through crossings (:mod:`strands`), and
    the strands are assembled into curves. Each returned curve must still sweep
    a decent fraction of the width (rejects gradient stripes)."""
    from .strands import skeleton_to_strands, strands_to_curves

    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    if biggest < 50:
        return []
    keep = [i + 1 for i, a in enumerate(areas) if a >= max(40, 0.05 * biggest)]
    kept_mask = np.isin(labels, keep)
    area = int(kept_mask.sum())
    skel = skeletonize_curve(kept_mask)
    skel_len = int(skel.sum())
    if skel_len < 30 or area / max(skel_len, 1) > max_stroke_width:
        return []

    strands = skeleton_to_strands(skel)
    curves = strands_to_curves(strands, drop_axes=drop_axes, ink_mask=kept_mask)
    out = []
    for i, poly in enumerate(curves):
        if len(poly) >= 30 and np.ptp(poly[:, 0]) >= min_x_span_frac * mask.shape[1]:
            out.append((poly, "solid" if i == 0 else "dashed"))
    return out


def _curves_from_mask(mask: np.ndarray, *, max_spur_len: int, drop_axes: bool = False,
                      include_legacy: bool = False) -> list[tuple[np.ndarray, str]]:
    """Roled curves for one colour/dark mask: ``[(polyline, role), ...]``.

    Junction-aware strand curves ("solid"/"dashed") recover the solid-through-
    dashed and curve-through-crossing cases the single walk drops. With
    ``include_legacy`` the old single-walk curve is also returned, tagged
    "alt" — a second candidate the benchmark's best-match can prefer on a clean
    lone curve whose self-crossings the strand pairing traces slightly worse.
    The CLI leaves it off, so its output stays one curve per physical stroke."""
    out: list[tuple[np.ndarray, str]] = []
    if USE_STRANDS:
        out = _mask_to_curves(mask, drop_axes=drop_axes)
    if not out:
        poly = _mask_to_curve(mask, max_spur_len=max_spur_len)
        return [(poly, "solid")] if poly is not None else []
    if include_legacy:
        poly = _mask_to_curve(mask, max_spur_len=max_spur_len)
        if poly is not None:
            out.append((poly, "alt"))
    return out


def _role_suffix(role: str) -> str:
    return {"solid": "", "dashed": "_dash", "alt": "_alt"}.get(role, "")


def split_color_curves(interior: np.ndarray, *, sat_thresh: float = 0.35,
                       min_pixels: int = 200, hue_bins: int = 36,
                       value_thresh: float = 0.55, max_spur_len: int = 15,
                       include_legacy: bool = False) -> list[dict]:
    """Separate the curves inside a plot interior by colour, plus the dark one.

    Clusters the hues of saturated pixels (histogram peaks), builds one mask
    per hue cluster, and traces each through the same skeleton pipeline as the
    dark curve. Returns ``[{name, rgb, polyline_px}, ...]`` in interior pixel
    space, largest first. Gradient fills and stripes are rejected by
    :func:`_mask_to_curve`'s thinness/span gates, so a figure like a rainbow-
    filled single-curve CV still yields exactly one (dark) curve.
    """
    hsv = cv2.cvtColor(interior, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0          # 0..360
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    colorful = (sat >= sat_thresh) & (val >= 0.15)

    curves: list[dict] = []
    claimed = np.zeros(interior.shape[:2], dtype=bool)
    if int(colorful.sum()) >= min_pixels:
        hist, edges = np.histogram(hue[colorful], bins=hue_bins, range=(0.0, 360.0))
        binw = 360.0 / hue_bins
        for b in range(hue_bins):
            c = int(hist[b])
            if c < min_pixels:
                continue
            if c < hist[(b - 1) % hue_bins] or c < hist[(b + 1) % hue_bins]:
                continue  # not a local peak
            center = (edges[b] + edges[b + 1]) / 2
            dist = np.abs(((hue - center + 180.0) % 360.0) - 180.0)
            mask = colorful & (dist <= binw)
            polys = _curves_from_mask(mask, max_spur_len=max_spur_len,
                                      include_legacy=include_legacy)
            if not polys:
                continue
            claimed |= mask
            mean_rgb = tuple(round(float(v) / 255.0, 3)
                             for v in interior[mask].mean(axis=0))
            base = _hue_name(center)
            for poly, role in polys:
                curves.append({"name": base + _role_suffix(role),
                               "rgb": mean_rgb, "polyline_px": poly})

    # The dark (black/grey) curve. Exclude only pixels already claimed by an
    # ACCEPTED colour curve — not all saturated pixels: a black line running
    # through a coloured gradient fill acquires the fill's hue/saturation in
    # its anti-aliased blend while staying dark, and darkness (not lack of
    # colour) is what defines it. This keeps single-dark-curve figures with
    # decorative fills working exactly as before.
    dark = mask_dark_curve(interior, value_thresh=value_thresh) & ~claimed
    for poly, role in _curves_from_mask(dark, max_spur_len=max_spur_len,
                                        include_legacy=include_legacy):
        curves.append({"name": "dark" + _role_suffix(role),
                       "rgb": (0.1, 0.1, 0.1), "polyline_px": poly})

    # de-duplicate names (two peaks can share a base name: red vs red2)
    seen: dict[str, int] = {}
    for cdict in curves:
        n = seen.get(cdict["name"], 0)
        seen[cdict["name"]] = n + 1
        if n:
            cdict["name"] = f"{cdict['name']}{n + 1}"
    curves.sort(key=lambda d: len(d["polyline_px"]), reverse=True)
    return curves


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def _erase_straight_lines(mask: np.ndarray, *, min_len_frac: float = 0.4,
                          thickness: int = 3) -> np.ndarray:
    """Zero out long, perfectly straight horizontal/vertical runs (plot axes).

    Classic voltammograms are drawn with bare crossing axes and no box, so
    :func:`detect_all_frames` finds nothing. The axis lines are the only dark
    features that run dead-straight for a large fraction of the panel; a CV
    curve is locally curvy and never contributes a single-row/column run that
    long. We erase those runs (dilated a little to cover anti-aliased edges),
    leaving the curve — which reconnects into a few long arcs the postprocessor
    can stitch.
    """
    out = mask.copy()
    h, w = mask.shape
    hmin = max(20, int(min_len_frac * w))
    vmin = max(20, int(min_len_frac * h))
    t = thickness
    for r, s, e in _line_segments(mask, hmin, horizontal=True):
        out[max(0, r - t):r + t + 1, s:e] = False
    for c, s, e in _line_segments(mask, vmin, horizontal=False):
        out[s:e, max(0, c - t):c + t + 1] = False
    return out


def _components_as_polylines(mask: np.ndarray, *, min_area: int,
                             max_spur_len: int) -> list[np.ndarray]:
    """Skeletonise each sizeable connected component into an (x, y) polyline."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    polylines = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = labels == i
        skel = skeletonize_curve(comp)
        poly = skeleton_to_polyline(skel, max_spur_len=max_spur_len)
        if len(poly) >= 8:
            polylines.append(poly)
    return polylines


def extract_frameless_curve(rgb: np.ndarray, *, value_thresh: float = 0.55,
                            max_spur_len: int = 15) -> np.ndarray:
    """Extract the dominant dark curve from a panel with no detectable frame.

    For classic crossing-axis voltammograms (no bounding box): mask dark
    pixels, skeletonise, and decompose into smooth strands
    (:mod:`strands`) — a curve crossing an axis is exactly a junction, and the
    strand tracer follows straight through it rather than getting cut there.
    Axis-shaped strands (long, straight, horizontal/vertical) are dropped and
    the rest stitched into one curve. This avoids the failure mode of the
    older erase-the-axis-pixels approach: erasing cuts the curve at every
    crossing, and the plain nearest-endpoint restitch then shortcuts across the
    gap — visible as diagonal chords through the plot in low-contrast scans.

    Dense components (body text on a full-page scan, even where touching
    glyphs merge into big blobs) are dropped before strand decomposition by the
    same sparse-and-long test :func:`locate_plot_regions` uses — a thin curve
    stroke fills a low fraction of its own bounding box no matter how big that
    box is, while a paragraph of touching text is comparatively dense. This
    matters for correctness (drop text, keep the curve) but also for
    performance: the final stitch is O(strand count²), so even one big blob of
    merged text glyphs — dense, but not necessarily small in raw pixel area —
    can make this pathologically slow if let through. Returns an (x, y) pixel
    polyline (empty if nothing remains).
    """
    from .strands import skeleton_to_strands, strands_to_curves
    from .postprocess import dedupe

    mask = mask_dark_curve(rgb, value_thresh=value_thresh)
    h, w = mask.shape
    page_diag = float(np.hypot(h, w))
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.empty((0, 2))
    keep = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        diag = float(np.hypot(ww, hh))
        fill = area / (ww * hh + 1)
        if diag >= 0.12 * page_diag and fill < 0.30:
            keep.append(i)
    if not keep:
        return np.empty((0, 2))
    big_mask = np.isin(labels, keep)

    skel = skeletonize_curve(big_mask)
    curves = strands_to_curves(skeleton_to_strands(skel), drop_axes=True,
                               ink_mask=big_mask)
    if not curves:
        return np.empty((0, 2))
    return dedupe(curves[0])


def _merge_boxes(boxes: list[tuple[int, int, int, int]], *, gap: int = 0
                 ) -> list[tuple[int, int, int, int]]:
    """Union overlapping/adjacent (x0, y0, x1, y1) boxes (a curve split across
    a couple of components becomes one figure region)."""
    boxes = list(boxes)
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        out: list[tuple[int, int, int, int]] = []
        while boxes:
            a = boxes.pop()
            ax0, ay0, ax1, ay1 = a
            hit = None
            for b in out:
                bx0, by0, bx1, by1 = b
                if (ax0 <= bx1 + gap and bx0 <= ax1 + gap
                        and ay0 <= by1 + gap and by0 <= ay1 + gap):
                    hit = b
                    break
            if hit is not None:
                out.remove(hit)
                out.append((min(ax0, hit[0]), min(ay0, hit[1]),
                            max(ax1, hit[2]), max(ay1, hit[3])))
                merged = True
            else:
                out.append(a)
        boxes = out
    return boxes


def locate_plot_regions(gray: np.ndarray, *, dark_thresh: int = 160,
                        min_diag_frac: float = 0.18, max_fill: float = 0.30,
                        pad_frac: float = 0.03) -> list[tuple[int, int, int, int]]:
    """Find plot sub-regions inside a full-page scan (text + one or more figures).

    A voltammogram trace is a single large but *sparsely filled* connected
    component — a thin ink stroke sprawling across a big bounding box (fill
    ~2-10%) — whereas body-text characters are small and any filled block/rule
    is dense or spans the full page width. We keep components that are large and
    thin, pad and merge their boxes, and return them so the curve can be
    extracted from a tight crop instead of the whole text page. Returns [] when
    nothing plot-like stands out (caller then treats the region as a single
    plot)."""
    h, w = gray.shape
    page_diag = float(np.hypot(h, w))
    dark = (gray < dark_thresh).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    px, py = int(pad_frac * w), int(pad_frac * h)
    boxes = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        diag = float(np.hypot(ww, hh))
        fill = area / (ww * hh + 1)
        if (diag >= min_diag_frac * page_diag and fill < max_fill
                and ww < 0.95 * w and hh < 0.95 * h):
            boxes.append((max(0, x - px), max(0, y - py),
                          min(w, x + ww + px), min(h, y + hh + py)))
    return _merge_boxes(boxes)


def extract_scan_curves(rgb: np.ndarray, *, dark_thresh: int = 160,
                        min_diag_frac: float = 0.18, max_fill: float = 0.30,
                        max_spur_len: int = 15) -> list[np.ndarray]:
    """Extract voltammogram curves from a full-page scan (text + figures).

    Locates each curve as a large, sparsely-filled connected component (see
    :func:`locate_plot_regions`) and traces *only that component's pixels* — so
    body text and figure captions can never be stitched into the curve, which is
    the failure mode of extracting from a padded bounding box. Within a
    component the ink is decomposed into strands that pass straight through
    axis crossings (:mod:`strands`) instead of being cut and re-stitched there,
    and axis-shaped strands are dropped. Returns an (x, y) polyline per curve
    found (a component can yield more than one, e.g. a solid + dashed pair)."""
    from .strands import skeleton_to_strands, strands_to_curves
    from .postprocess import dedupe

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    page_diag = float(np.hypot(h, w))
    dark = (gray < dark_thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)

    curves = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        diag = float(np.hypot(ww, hh))
        fill = area / (ww * hh + 1)
        if not (diag >= min_diag_frac * page_diag and fill < max_fill
                and ww < 0.95 * w and hh < 0.95 * h):
            continue
        comp = labels == i                       # this figure's ink only
        skel = skeletonize_curve(comp)
        for poly in strands_to_curves(skeleton_to_strands(skel), drop_axes=True,
                                      ink_mask=comp):
            if len(poly) >= 20:
                curves.append(dedupe(poly))
    return curves


def _extract_from_frame(img: np.ndarray, frame: tuple[int, int, int, int], *,
                        value_thresh: float, frame_inset_px: int,
                        max_spur_len: int, include_legacy: bool = False) -> dict:
    """Curve isolation for one already-located frame, in ``img``'s own pixel space."""
    x0, y0, x1, y1 = frame
    h, w = img.shape[:2]
    iy0, iy1 = y0 + frame_inset_px, y1 - frame_inset_px
    ix0, ix1 = x0 + frame_inset_px, x1 - frame_inset_px
    if iy1 <= iy0 or ix1 <= ix0:
        # The frame is too small to inset safely (e.g. a spurious sliver from
        # frame detection) -- rather than slice to empty and crash downstream
        # (cv2.cvtColor on a (0, n) array), fall back to the frame's own
        # bounds uninset, clamped to the image; if even that's degenerate,
        # use the whole image.
        iy0, iy1 = (y0, y1) if y1 > y0 else (0, h)
        ix0, ix1 = (x0, x1) if x1 > x0 else (0, w)
    interior = img[iy0:iy1, ix0:ix1]

    mask = mask_dark_curve(interior, value_thresh=value_thresh)
    comp = largest_component(mask)
    skel = skeletonize_curve(comp)
    poly_px = skeleton_to_polyline(skel, max_spur_len=max_spur_len)  # (x, y) in `interior` px
    poly_px = poly_px + np.array([ix0, iy0])                         # -> `img` pixel space

    offset = np.array([ix0, iy0], dtype=float)
    curves = []
    for cdict in split_color_curves(interior, value_thresh=value_thresh,
                                    max_spur_len=max_spur_len,
                                    include_legacy=include_legacy):
        curves.append({**cdict, "polyline_px": cdict["polyline_px"] + offset})

    return {"polyline_px": poly_px, "frame_px": frame, "mask": comp,
            "skeleton": skel, "curves": curves}


def extract_from_cropped_image(img: np.ndarray, *, value_thresh: float = 0.55,
                               frame_inset_px: int = 4, max_spur_len: int = 15,
                               include_legacy: bool = False) -> dict:
    """Detect-frame + mask + skeletonise an image that ALREADY isolates one
    plot — e.g. a panel a human cropped by hand (bypassing PDF/page/zoom
    entirely, unlike :func:`extract_raster_curve`).

    Because the crop already excludes everything except one figure (no
    schematic, no neighbouring sub-plots), ``detect_all_frames``-style frame
    detection is reliable here even on composite figures where it explodes
    into spurious candidates over the WHOLE page (a crystal-structure
    schematic's sphere shapes read as dozens of fake plot frames) — cropping
    first is what makes plain frame detection trustworthy again, not a change
    to the detector itself. Falls back to a 5% margin if no frame is found
    (mirrors :func:`extract_raster_curve`)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frame = detect_frame_bbox(gray)
    if frame is None:
        h, w = gray.shape
        margin = 0.05
        frame = (int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin)))
    return _extract_from_frame(img, frame, value_thresh=value_thresh,
                               frame_inset_px=frame_inset_px, max_spur_len=max_spur_len,
                               include_legacy=include_legacy)


def extract_raster_curve(
    pdf_path: str,
    page_number: int,
    region_bbox: tuple[float, float, float, float],
    *,
    zoom: float = 4.0,
    value_thresh: float = 0.55,
    frame_inset_px: int = 4,
    max_spur_len: int = 15,
) -> dict:
    """Full single-panel raster pipeline: render -> detect frame -> mask ->
    largest component -> skeletonize -> ordered polyline.

    Use when ``region_bbox`` already isolates one plot (a single-figure PDF
    page, or a pre-cropped panel). For a composite image holding several
    plot panels, use ``extract_all_panel_curves`` instead.

    Returns a dict with the ``polyline`` in **PDF-point coordinates** (so it is
    directly comparable/combinable with the vector branch and any
    ``Calibration``), plus intermediate arrays useful for debugging/plots.
    """
    img = render_region(pdf_path, page_number, region_bbox, zoom=zoom)
    result = extract_from_cropped_image(img, value_thresh=value_thresh,
                                        frame_inset_px=frame_inset_px, max_spur_len=max_spur_len)
    poly_pdf = result["polyline_px"] / zoom + np.array([region_bbox[0], region_bbox[1]])
    return {**result, "polyline": poly_pdf, "image": img}


def extract_all_panel_curves(
    pdf_path: str,
    page_number: int,
    region_bbox: tuple[float, float, float, float],
    *,
    zoom: float = 4.0,
    value_thresh: float = 0.55,
    frame_inset_px: int = 4,
    max_spur_len: int = 15,
    include_legacy: bool = False,
) -> list[dict]:
    """Full raster pipeline for a **composite, multi-panel** figure region.

    Renders the region once, auto-detects every plot panel in it
    (``detect_all_frames``), and runs curve extraction independently in each.
    Returns one result dict per panel (same keys as ``extract_raster_curve``,
    plus ``ticks``), in reading order (top-to-bottom, left-to-right) — index
    them the same way the vector branch indexes panel letters.

    ``include_legacy`` adds the pre-strand single-walk trace of each colour as an
    extra "*_alt" candidate (used by the accuracy benchmark, off for the CLI).
    """
    img = render_region(pdf_path, page_number, region_bbox, zoom=zoom)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frames = detect_all_frames(gray)

    results = []
    for frame in frames:
        result = _extract_from_frame(img, frame, value_thresh=value_thresh,
                                     frame_inset_px=frame_inset_px, max_spur_len=max_spur_len,
                                     include_legacy=include_legacy)
        poly_pdf = result["polyline_px"] / zoom + np.array([region_bbox[0], region_bbox[1]])
        ticks = detect_axis_ticks(gray, frame)
        results.append({**result, "polyline": poly_pdf, "image": img, "ticks": ticks})
    return results
