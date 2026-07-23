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


def split_strokes(points, jump_factor: float = 5.0, min_abs_jump: float = 15.0
                  ) -> list[np.ndarray]:
    """Split a flat point list back into separate strokes.

    A human trace is usually drawn as several mouse-drags (pen lifted between
    them). If the exporter only kept a flat point list with no stroke boundary
    markers, the gap between the end of one drag and the start of the next
    still shows up as an outlier-sized step — far bigger than the steady
    per-sample spacing within a drag — so we split there. This makes the
    extractor robust to flattened exports; the trace-assist tool now exports
    real stroke boundaries too, so this is a safety net, not the primary path.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return [pts] if len(pts) else []
    step = np.hypot(*(np.diff(pts, axis=0).T))
    med = np.median(step) or 1.0
    thresh = max(min_abs_jump, jump_factor * med)
    cuts = np.where(step > thresh)[0] + 1
    return [s for s in np.split(pts, cuts) if len(s) >= 2]


def _snap_stroke(stroke: np.ndarray, tree, ink_pts: np.ndarray, radius: float
                 ) -> np.ndarray:
    """Densify one stroke and snap each position to the nearest ink pixel."""
    dense = _densify(stroke, step=2.0)
    snapped = []
    for g in dense:
        near = tree.query_ball_point(g, radius)
        if not near:
            continue
        cand = ink_pts[near]
        snapped.append(cand[np.argmin(np.hypot(cand[:, 0] - g[0], cand[:, 1] - g[1]))])
    return np.asarray(snapped) if snapped else np.empty((0, 2))


def extract_near_guide(rgb: np.ndarray, guide_xy, *, radius: int | None = None,
                       value_thresh: float = 0.6, resample_n: int = 600,
                       strokes: list | None = None, return_gaps: bool = False):
    """Extract the curve the guide points at, as an (x, y) pixel polyline.

    ``guide_xy`` is a rough (M, 2) polyline in image-pixel coordinates (the human
    scribble); pass ``strokes`` (a list of separate strokes) instead when the
    guide was drawn as several mouse-drags — each is snapped to ink
    independently, so a pen-lift between drags is never mistaken for curve ink.
    If only ``guide_xy`` is given, likely stroke boundaries are recovered from
    anomalous point-to-point jumps (:func:`split_strokes`).

    Each stroke may be a plain ``[[x,y], ...]`` point list (uses ``radius``) or
    a ``{"radius": r, "pts": [[x,y], ...]}`` dict carrying its OWN brush size —
    the tracer records the brush width at the moment each stroke was drawn, so
    changing the brush mid-curve only affects strokes drawn after the change,
    never ones already on the canvas.

    Each stroke is densified and, at every position along it, snapped to the
    nearest curve-ink pixel within its radius (default: ~2% of the image
    diagonal — enough to cover a hand-drawn wobble). The per-stroke snapped
    pieces are then stitched into one ordered curve the same way the rest of
    the pipeline stitches fragmented sub-paths (nearest-endpoint), so strokes
    drawn in any order or direction still assemble correctly. Returns an empty
    array if no ink is found near the guide.

    If the strokes together don't cover the whole curve, the join between two
    pieces is necessarily a straight line (there is no ink to snap to in an
    untraced span) — this is honest, not wrong, but worth surfacing. With
    ``return_gaps=True`` the return becomes ``(polyline, gaps)`` where ``gaps``
    is a list of ``((x0,y0),(x1,y1))`` straight-line spans the caller can mark
    distinctly (dashed, greyed out) rather than presenting as traced ink.
    """
    if cv2 is None:
        raise RuntimeError("guided extraction needs OpenCV (opencv-python-headless)")
    from scipy.spatial import cKDTree
    from .postprocess import dedupe, resample_arclength, order_curve

    def _ret(poly, gaps):
        return (poly, gaps) if return_gaps else poly

    h, w = rgb.shape[:2]
    default_radius = radius if radius is not None else max(6, int(0.02 * np.hypot(h, w)))

    if strokes is not None:
        stroke_list = []
        for s in strokes:
            if isinstance(s, dict):
                pts = np.asarray(s.get("pts", s.get("points", [])), float)
                r = float(s.get("radius") or default_radius)
            else:
                pts = np.asarray(s, float)
                r = float(default_radius)
            if len(pts) >= 2:
                stroke_list.append((pts, r))
    else:
        stroke_list = [(s, float(default_radius)) for s in split_strokes(guide_xy)]
    if not stroke_list:
        return _ret(np.empty((0, 2)), [])

    ink = _ink_mask(rgb, value_thresh=value_thresh)
    corridor = np.zeros(rgb.shape[:2], bool)
    for pts, r in stroke_list:
        corridor |= _corridor_mask(rgb.shape, pts, int(round(r)))
    ys, xs = np.where(ink & corridor)
    if len(xs) < 10:
        return _ret(np.empty((0, 2)), [])
    ink_pts = np.column_stack([xs, ys]).astype(float)
    tree = cKDTree(ink_pts)

    pieces = [p for p in (_snap_stroke(pts, tree, ink_pts, r) for pts, r in stroke_list)
             if len(p) >= 3]
    if not pieces:
        return _ret(np.empty((0, 2)), [])
    ordered = order_curve(pieces) if len(pieces) > 1 else pieces[0]
    ordered = dedupe(ordered)
    if len(ordered) < 10:
        return _ret(np.empty((0, 2)), [])

    # A real snapped run advances in ~_densify-step (2px) hops; a join between
    # two unconnected pieces (or a stretch with no ink at all) is a far bigger
    # jump — flag those as untraced gaps rather than silently presenting them
    # as if they were guided.
    gaps = []
    if len(ordered) > 1 and return_gaps:
        mean_r = float(np.mean([r for _, r in stroke_list]))
        step = np.hypot(*(np.diff(ordered, axis=0).T))
        thresh = max(3.0 * mean_r, 4.0 * (np.median(step) or 1.0))
        for i in np.where(step > thresh)[0]:
            gaps.append((tuple(ordered[i]), tuple(ordered[i + 1])))

    return _ret(resample_arclength(ordered, n=resample_n), gaps)


def extract_guides(rgb: np.ndarray, guides: list[dict], *, return_gaps: bool = False,
                   **kw) -> list[dict]:
    """Run :func:`extract_near_guide` for each named guide.

    ``guides`` is ``[{"name": str, "points": [...]}, ...]`` or
    ``[{"name": str, "strokes": [[...], [...]]}, ...]`` (as exported by the
    trace-assist tool). A guide may carry its own ``"radius"`` (the brush size
    used to draw it), which overrides the ``radius`` in ``kw`` for that curve.
    Returns ``[{"name", "polyline_px"[, "gaps"]}, ...]``, skipping any guide that
    yields no ink. With ``return_gaps=True`` each result also carries the
    untraced-span list from :func:`extract_near_guide`.
    """
    out = []
    for g in guides:
        src = {"strokes": g["strokes"]} if g.get("strokes") else {}
        guide_arg = None if src else g["points"]
        gkw = dict(kw)
        if g.get("radius"):
            gkw["radius"] = int(g["radius"])
        res = extract_near_guide(rgb, guide_arg, return_gaps=return_gaps, **src, **gkw)
        poly, gaps = res if return_gaps else (res, None)
        if len(poly):
            item = {"name": g.get("name", ""), "polyline_px": poly}
            if return_gaps:
                item["gaps"] = gaps
            out.append(item)
    return out
