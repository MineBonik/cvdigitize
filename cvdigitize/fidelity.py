"""Reference-free trace quality: how well an extracted curve lies on real ink.

The honest test of a digitization is to lay the trace back over the figure and
see whether the line sits on the ink or drifts off it — no ground-truth curve
needed. This module measures exactly that: distance-transform the ink the curve
was traced from, walk along the extracted polyline, and read how far each step
is from the nearest ink. A good trace hugs the ink (distance ~0 everywhere); a
diagonal chord shot across the empty plot spends a long run far from any ink.

**Dash awareness (critical).** A correctly-traced *dashed* curve bridges the gap
between each dash with a short straight segment, so it is legitimately off-ink
for those short spans. We must not punish that. So off-ink runs are split by
length: SHORT runs (dash-sized) are tolerated; only LONG runs — chords leaping
across the plot to unrelated ink — count as defects and are returned for
highlighting / triage. The same short-vs-long rule is what lets the ink-aware
stitcher (:func:`cvdigitize.postprocess.order_curve`) bridge dashes but refuse
chords.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None


def _distance_to_ink(ink_mask: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance to the nearest ink (True) pixel; 0 on ink."""
    non_ink = (~ink_mask.astype(bool)).astype(np.uint8)
    if not non_ink.any():               # ink everywhere -> distance 0 everywhere
        return np.zeros(ink_mask.shape, np.float32)
    if not ink_mask.any():              # no ink at all -> "infinitely" far
        return np.full(ink_mask.shape, float(np.hypot(*ink_mask.shape)), np.float32)
    return cv2.distanceTransform(non_ink, cv2.DIST_L2, 3)


def _densify(xy: np.ndarray, step: float) -> np.ndarray:
    from .postprocess import resample_arclength
    return resample_arclength(np.asarray(xy, float), spacing=step)


def ink_fidelity(polyline_px, ink_mask: np.ndarray, *, stroke_width: float | None = None,
                 sample_step: float = 2.0, dash_gap_frac: float = 0.04,
                 tol: float | None = None) -> dict:
    """Score how faithfully ``polyline_px`` lies on ``ink_mask`` (0..100, higher=better).

    ``polyline_px`` is an (N, 2) trace in the mask's pixel coordinates; ``ink_mask``
    is the boolean ink the curve was traced from (e.g. the output of
    ``mask_dark_curve`` or a per-hue mask from ``split_color_curves``).

    A sample is *on ink* when it is within ``tol`` px of ink (``tol`` defaults to a
    small fraction of the panel diagonal, or ``stroke_width`` if given). Consecutive
    off-ink samples form a span; spans shorter than ``dash_gap_frac`` of the panel
    diagonal are tolerated as dash gaps, longer ones are chord DEFECTS.

    Returns ``{score, frac_off_ink, coverage, n_defects, off_ink_spans}`` where
    ``off_ink_spans`` is ``[{"start": (x,y), "end": (x,y), "len": px}, ...]`` for the
    long defect spans only — the chords to highlight red and route to trace_assist.
    """
    if cv2 is None:
        raise RuntimeError("ink_fidelity needs OpenCV (opencv-python-headless)")
    xy = np.asarray(polyline_px, float)
    h, w = ink_mask.shape[:2]
    diag = float(np.hypot(h, w)) or 1.0
    empty = {"score": 0.0, "frac_off_ink": 1.0, "coverage": 0.0,
             "n_defects": 0, "off_ink_spans": []}
    if len(xy) < 2:
        return empty

    if tol is None:
        tol = max(3.0, stroke_width or 0.0, 0.005 * diag)
    dash_gap_len = dash_gap_frac * diag

    dt = _distance_to_ink(ink_mask.astype(bool))
    samples = _densify(xy, sample_step)
    if len(samples) < 2:
        return empty
    xi = np.clip(np.round(samples[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(samples[:, 1]).astype(int), 0, h - 1)
    dist = dt[yi, xi]
    off = dist > tol                     # per-sample: off the ink?

    # group consecutive off-ink samples into spans
    spans = []
    i = 0
    n = len(off)
    while i < n:
        if off[i]:
            j = i
            while j < n and off[j]:
                j += 1
            spans.append((i, j - 1))     # inclusive index range
            i = j
        else:
            i += 1

    total_len = (len(samples) - 1) * sample_step or 1.0
    defect_spans, defect_len = [], 0.0
    for a, b in spans:
        span_len = (b - a) * sample_step
        if span_len > dash_gap_len:      # long -> chord defect (not a dash gap)
            defect_len += span_len
            defect_spans.append({"start": tuple(np.round(samples[a], 1)),
                                 "end": tuple(np.round(samples[b], 1)),
                                 "len": round(span_len, 1)})

    score = max(0.0, min(100.0, 100.0 * (1.0 - defect_len / total_len)))
    coverage = _ink_coverage(samples, ink_mask.astype(bool), tol)
    return {"score": round(score, 1),
            "frac_off_ink": round(float(off.mean()), 3),
            "coverage": round(coverage, 3),
            "n_defects": len(defect_spans),
            "off_ink_spans": defect_spans}


def _ink_coverage(samples: np.ndarray, ink_mask: np.ndarray, tol: float) -> float:
    """Fraction of ink that lies within ``tol`` px of the trace (catches a trace
    that only covers part of the curve — missing branches, truncated sweeps)."""
    if not ink_mask.any():
        return 0.0
    h, w = ink_mask.shape
    band = np.zeros((h, w), np.uint8)
    pts = np.round(samples).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(band, [pts], False, 1, thickness=max(1, int(round(2 * tol))))
    covered = int((ink_mask & (band > 0)).sum())
    return covered / int(ink_mask.sum())


def grade(score: float) -> str:
    """Bucket a fidelity score: 'good' (>=85), 'fair' (>=65), else 'poor'."""
    return "good" if score >= 85 else "fair" if score >= 65 else "poor"
