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


def find_image_regions(pdf_path: str, page_number: int,
                       *, min_area_frac: float = 0.05) -> list[ImageRegion]:
    """Embedded images on a page, largest first, dropping tiny icons/logos."""
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    regions = []
    for img in page.get_images(full=True):
        xref = img[0]
        for r in page.get_image_rects(xref):
            area = abs(r.width * r.height)
            if area / page_area >= min_area_frac:
                regions.append(ImageRegion(xref, (r.x0, r.y0, r.x1, r.y1)))
    doc.close()
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
                      min_len_frac: float = 0.35) -> tuple[int, int, int, int] | None:
    """Find the plot's rectangular axes box via long straight Hough lines.

    Works even when the curve, ticks and frame share the same dark colour: the
    frame's four border lines are by far the longest straight dark segments in
    the image, so filtering by minimum length isolates them from the (locally
    curvy, shorter) data curve. Returns ``(x0, y0, x1, y1)`` in pixel
    coordinates, or ``None`` if no confident frame was found.
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


def detect_axis_ticks(gray: np.ndarray, frame_px: tuple[int, int, int, int], *,
                      dark_thresh: int = 220, tick_band: int = 6,
                      min_count: int = 3) -> dict[str, list[int]]:
    """Find tick-mark pixel positions along the bottom and left axes of a frame.

    Looks for short dark tick marks just outside the frame border (below the
    bottom edge for x-ticks, left of the left edge for y-ticks) and returns
    their positions in the **same pixel space as** ``frame_px`` (i.e. absolute
    in ``gray``, matching ``polyline_px`` elsewhere in this module) — not
    frame-relative. This turns manual calibration from "find two exact pixel
    coordinates" into "read off two tick labels", since the pixel positions
    are already known once ticks are detected; the user only supplies the two
    corresponding data values.
    """
    left, top, right, bottom = frame_px
    dark = gray < dark_thresh
    edge_margin = tick_band + 2  # drop positions this close to a corner (border bleed)

    xband = dark[bottom + 2:bottom + 2 + tick_band, left:right]
    x_positions = _cluster_peaks(np.where(xband.sum(axis=0) >= min_count)[0])
    x_positions = [p + left for p in x_positions
                  if edge_margin <= p <= (right - left) - edge_margin]

    y0 = max(0, left - 2 - tick_band)
    yband = dark[top:bottom, y0:left - 2]
    y_positions = _cluster_peaks(np.where(yband.sum(axis=1) >= min_count)[0])
    y_positions = [p + top for p in y_positions
                  if edge_margin <= p <= (bottom - top) - edge_margin]

    return {"x_ticks": x_positions, "y_ticks": y_positions}


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
# End-to-end
# --------------------------------------------------------------------------- #
def _extract_from_frame(img: np.ndarray, frame: tuple[int, int, int, int], *,
                        value_thresh: float, frame_inset_px: int,
                        max_spur_len: int) -> dict:
    """Curve isolation for one already-located frame, in ``img``'s own pixel space."""
    x0, y0, x1, y1 = frame
    inset = frame_inset_px
    interior = img[y0 + inset:y1 - inset, x0 + inset:x1 - inset]

    mask = mask_dark_curve(interior, value_thresh=value_thresh)
    comp = largest_component(mask)
    skel = skeletonize_curve(comp)
    poly_px = skeleton_to_polyline(skel, max_spur_len=max_spur_len)  # (x, y) in `interior` px
    poly_px = poly_px + np.array([x0 + inset, y0 + inset])           # -> `img` pixel space

    return {"polyline_px": poly_px, "frame_px": frame, "mask": comp, "skeleton": skel}


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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    frame = detect_frame_bbox(gray)
    if frame is None:
        h, w = gray.shape
        margin = 0.05
        frame = (int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin)))

    result = _extract_from_frame(img, frame, value_thresh=value_thresh,
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
) -> list[dict]:
    """Full raster pipeline for a **composite, multi-panel** figure region.

    Renders the region once, auto-detects every plot panel in it
    (``detect_all_frames``), and runs curve extraction independently in each.
    Returns one result dict per panel (same keys as ``extract_raster_curve``,
    plus ``ticks``), in reading order (top-to-bottom, left-to-right) — index
    them the same way the vector branch indexes panel letters.
    """
    img = render_region(pdf_path, page_number, region_bbox, zoom=zoom)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frames = detect_all_frames(gray)

    results = []
    for frame in frames:
        result = _extract_from_frame(img, frame, value_thresh=value_thresh,
                                     frame_inset_px=frame_inset_px, max_spur_len=max_spur_len)
        poly_pdf = result["polyline_px"] / zoom + np.array([region_bbox[0], region_bbox[1]])
        ticks = detect_axis_ticks(gray, frame)
        results.append({**result, "polyline": poly_pdf, "image": img, "ticks": ticks})
    return results
