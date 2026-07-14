"""Post-processing of an extracted CV curve (M1).

A CV curve is a closed hysteresis loop swept once (anodic scan E_low->E_high,
then cathodic scan back). In a vector PDF it is stored as many disconnected
sub-paths in arbitrary order and direction, and the sampled points are unevenly
spaced with duplicate potentials (two currents per E). Albert flagged exactly
this: the digitized CSV "is not raw".

This module reconstructs a physically sensible curve:

1. ``order_curve``     greedily stitch sub-paths into one ordered traversal.
2. ``split_branches``  cut the loop into anodic / cathodic scans at the E turns.
3. ``resample_*``      re-sample on an even grid (uniform arc-length for a clean
                       smooth trace, or uniform-E per branch to mimic the
                       constant-scan-rate sampling of a real potentiostat).

All functions work in whatever coordinates you pass (PDF pixels or calibrated
E/j); calibration is a separate, linear step (see ``calibrate.py``).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Stitch arbitrary-order sub-paths into a single ordered polyline
# ---------------------------------------------------------------------------
def order_curve(polylines: list[np.ndarray]) -> np.ndarray:
    """Greedily connect sub-paths end-to-end into one ordered (N, 2) array.

    Starts at the left-most endpoint (near the lower potential limit) and
    repeatedly appends the nearest remaining sub-path endpoint, flipping that
    sub-path when it is joined by its tail. This recovers the sweep order even
    though the PDF stores the segments shuffled.
    """
    segs = [np.asarray(p, dtype=float) for p in polylines if len(p) >= 2]
    if not segs:
        return np.empty((0, 2))
    if len(segs) == 1:
        return segs[0].copy()

    # Choose the starting segment + orientation from the global left-most endpoint.
    best_i, best_flip, best_x = 0, False, np.inf
    for i, s in enumerate(segs):
        if s[0, 0] < best_x:
            best_x, best_i, best_flip = s[0, 0], i, False
        if s[-1, 0] < best_x:
            best_x, best_i, best_flip = s[-1, 0], i, True

    used = [False] * len(segs)
    first = segs[best_i][::-1] if best_flip else segs[best_i]
    chain = [first]
    used[best_i] = True
    tail = first[-1]

    for _ in range(len(segs) - 1):
        best_j, best_flip, best_d = -1, False, np.inf
        for j, s in enumerate(segs):
            if used[j]:
                continue
            ds = float(np.hypot(*(tail - s[0])))
            de = float(np.hypot(*(tail - s[-1])))
            if ds < best_d:
                best_d, best_j, best_flip = ds, j, False
            if de < best_d:
                best_d, best_j, best_flip = de, j, True
        seg = segs[best_j][::-1] if best_flip else segs[best_j]
        chain.append(seg)
        used[best_j] = True
        tail = seg[-1]

    return np.vstack(chain)


def dedupe(xy: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop consecutive duplicate points (within ``tol``)."""
    if len(xy) < 2:
        return xy
    keep = np.ones(len(xy), dtype=bool)
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    keep[1:] = d > tol
    return xy[keep]


# ---------------------------------------------------------------------------
# 2. Split the ordered loop into anodic / cathodic scans
# ---------------------------------------------------------------------------
def split_branches(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an ordered loop at its potential turning point.

    Returns ``(forward, reverse)`` where ``forward`` runs from the low-E start
    up to the high-E vertex and ``reverse`` returns. Assumes ``order_curve``
    started the traversal at the low-E end (so E rises then falls).
    """
    if len(xy) < 3:
        return xy, xy[:0]
    i_hi = int(np.argmax(xy[:, 0]))
    forward = xy[: i_hi + 1]
    reverse = xy[i_hi:]
    return forward, reverse


# ---------------------------------------------------------------------------
# 3. Resampling
# ---------------------------------------------------------------------------
def _cumulative_arclength(xy: np.ndarray) -> np.ndarray:
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_arclength(xy: np.ndarray, n: int | None = None,
                       spacing: float | None = None) -> np.ndarray:
    """Resample a polyline to points evenly spaced along its arc length.

    Give either ``n`` (number of points) or ``spacing`` (distance between
    points, in the curve's own units). Produces an even, smooth trace and
    removes uneven spacing / duplicate points while preserving shape.
    """
    xy = dedupe(xy)
    if len(xy) < 2:
        return xy
    s = _cumulative_arclength(xy)
    total = s[-1]
    if spacing is not None:
        n = max(2, int(round(total / spacing)) + 1)
    if n is None:
        n = len(xy)
    su = np.linspace(0.0, total, n)
    x = np.interp(su, s, xy[:, 0])
    y = np.interp(su, s, xy[:, 1])
    return np.column_stack([x, y])


def resample_uniform_x(branch: np.ndarray, n: int) -> np.ndarray:
    """Resample one monotonic-in-x branch onto a uniform x (potential) grid.

    Mimics a potentiostat sampling at a constant scan rate (uniform steps in
    E). Non-monotonic wobbles are handled by sorting on x and averaging the
    current of duplicate x values.
    """
    if len(branch) < 2:
        return branch
    order = np.argsort(branch[:, 0], kind="stable")
    xs, ys = branch[order, 0], branch[order, 1]
    # Aggregate duplicate x (near-vertical peaks) by mean y.
    ux, inv = np.unique(xs, return_inverse=True)
    uy = np.zeros_like(ux)
    counts = np.zeros_like(ux)
    np.add.at(uy, inv, ys)
    np.add.at(counts, inv, 1.0)
    uy /= counts
    grid = np.linspace(ux[0], ux[-1], n)
    yg = np.interp(grid, ux, uy)
    return np.column_stack([grid, yg])


def clean_cv(polylines: list[np.ndarray], *, n_arclength: int = 1000
             ) -> dict[str, np.ndarray]:
    """Full loop clean-up from raw sub-paths.

    Returns a dict with the ordered ``loop`` and an evenly arc-length-sampled
    ``smooth`` trace, plus the ``forward``/``reverse`` scan branches.
    """
    loop = dedupe(order_curve(polylines))
    forward, reverse = split_branches(loop)
    smooth = resample_arclength(loop, n=n_arclength)
    return {"loop": loop, "smooth": smooth, "forward": forward, "reverse": reverse}
