"""Find a plot's axes box and tick marks in a rendered page image.

Split out of the old ``raster_extract.py`` when the raster/scanned-figure
pipeline was removed (vector-only refactor). Even a vector figure's *curve
geometry* comes straight from the PDF's path data, but knowing which strokes
are axes vs. curve, and where the tick marks sit, still means looking at the
rendered picture in pixels — a PDF has no "this rectangle is the axes box"
tag. So this module stays even though nothing else about scanned figures does.

Used by:
  * ``vector_extract.detect_panels`` — find each plot's axes box, to assign
    curves to the panel that contains them.
  * ``autocalib.detect_ticks_for_bbox`` — tick positions for the assisted/OCR
    calibration path in ``vector-calibrate``.
  * ``veccal.scan`` — the same, reused across a page's panels instead of
    re-run per panel.
"""
from __future__ import annotations

import numpy as np


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
