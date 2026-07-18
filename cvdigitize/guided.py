"""Guided extraction: turn a rough human trace into a pixel-accurate curve.

When automatic extraction can't resolve an ambiguity — which stroke of an
8-curve bundle is which, solid vs dashed, a curve lost in a noisy scan — a
person scribbles a rough guide roughly along the curve they mean (a few seconds,
paint-style, no precision needed). This module then does the precise part:

  1. keep only the curve ink lying in a corridor around the guide (this resolves
     *which* ink belongs to the intended curve),
  2. skeletonise that ink and split it into smooth strands
     (:mod:`strands`) so overlapping neighbours that dip into the corridor are
     followed straight through, not merged in,
  3. order the strand points by their position **along the guide** (the guide
     also gives the sweep direction), and
  4. resample to an even trace.

The output snaps to the real ink, so it is as accurate as an automatic trace —
the human only supplies intent, not coordinates. Rough in, precise out.
"""
from __future__ import annotations

import numpy as np

try:                                    # cv2 is a hard dep of the raster branch
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None


def _ink_mask(rgb: np.ndarray, value_thresh: float = 0.6, sat_thresh: float = 0.30
              ) -> np.ndarray:
    """Dark OR saturated-colour pixels — any curve ink, regardless of colour."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    return (val < value_thresh) | ((sat >= sat_thresh) & (val >= 0.15))


def _corridor_mask(shape, guide_xy: np.ndarray, radius: int) -> np.ndarray:
    """Boolean mask of pixels within ``radius`` of the guide polyline."""
    m = np.zeros(shape[:2], np.uint8)
    pts = np.round(guide_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(m, [pts], isClosed=False, color=1, thickness=max(1, 2 * radius))
    return m > 0


def _densify(guide: np.ndarray, step: float = 2.0) -> np.ndarray:
    """Resample a polyline to ~``step``-pixel spacing (a dense ordered path)."""
    from .postprocess import dedupe
    guide = dedupe(np.asarray(guide, float))
    if len(guide) < 2:
        return guide
    seg = np.hypot(*(np.diff(guide, axis=0).T))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(cum[-1] / step))
    t = np.linspace(0, cum[-1], n)
    x = np.interp(t, cum, guide[:, 0])
    y = np.interp(t, cum, guide[:, 1])
    return np.column_stack([x, y])


def extract_near_guide(rgb: np.ndarray, guide_xy, *, radius: int | None = None,
                       value_thresh: float = 0.6, resample_n: int = 600
                       ) -> np.ndarray:
    """Extract the curve the guide points at, as an (x, y) pixel polyline.

    ``guide_xy`` is a rough (M, 2) polyline in image-pixel coordinates (the human
    scribble). We densify the guide and, at each position along it, snap to the
    nearest curve-ink pixel within ``radius`` — so the trace follows the guide's
    order (hence the sweep, and *both* branches of a loop, which the guide visits
    in turn) while sitting exactly on the real ink. ``radius`` defaults to ~2 %
    of the image diagonal, covering a hand-drawn guide's wobble. Returns an empty
    array if no ink is found along the guide.
    """
    if cv2 is None:
        raise RuntimeError("guided extraction needs OpenCV (opencv-python-headless)")
    from scipy.spatial import cKDTree
    from .postprocess import dedupe, resample_arclength

    guide = np.asarray(guide_xy, dtype=float)
    if len(guide) < 2:
        return np.empty((0, 2))
    h, w = rgb.shape[:2]
    if radius is None:
        radius = max(6, int(0.02 * np.hypot(h, w)))

    ink = _ink_mask(rgb, value_thresh=value_thresh)
    corridor = _corridor_mask(rgb.shape, guide, radius)
    ys, xs = np.where(ink & corridor)
    if len(xs) < 10:
        return np.empty((0, 2))
    ink_pts = np.column_stack([xs, ys]).astype(float)
    tree = cKDTree(ink_pts)

    # Walk the (densified) guide and, at each position, snap to the nearest ink
    # pixel within the corridor. The guide gives the order (and visits both
    # branches of a loop in turn); the snap puts every point on the real ink.
    dense = _densify(guide, step=2.0)
    snapped = []
    for g in dense:
        near = tree.query_ball_point(g, radius)
        if not near:
            continue
        cand = ink_pts[near]
        snapped.append(cand[np.argmin(np.hypot(cand[:, 0] - g[0], cand[:, 1] - g[1]))])
    if len(snapped) < 10:
        return np.empty((0, 2))
    ordered = dedupe(np.asarray(snapped))
    if len(ordered) < 10:
        return np.empty((0, 2))
    return resample_arclength(ordered, n=resample_n)


def extract_guides(rgb: np.ndarray, guides: list[dict], **kw) -> list[dict]:
    """Run :func:`extract_near_guide` for each named guide.

    ``guides`` is ``[{"name": str, "points": [[x, y], ...]}, ...]`` (as exported
    by the trace-assist tool). Returns ``[{"name", "polyline_px"}, ...]``,
    skipping any guide that yields no ink.
    """
    out = []
    for g in guides:
        poly = extract_near_guide(rgb, g["points"], **kw)
        if len(poly):
            out.append({"name": g.get("name", ""), "polyline_px": poly})
    return out
