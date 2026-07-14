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

from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Loop-likeness: a CV is a closed loop that encloses area; a schematic line,
# a Nyquist arc or an axis fragment is not. Used to gauge how "CV-like" an
# extracted trace is (reporting/triage only — never drops data).
# ---------------------------------------------------------------------------
def loop_metrics(xy: np.ndarray) -> dict[str, float]:
    """Return ``{'area_frac', 'closure', 'loopiness'}`` for an ordered curve.

    ``area_frac``  polygon area enclosed by the trace / its bounding-box area
                   (a fat CV loop is ~0.2-0.6; a thin open line is ~0).
    ``closure``    1 - gap(start,end)/bbox_diagonal (1 = perfectly closed).
    ``loopiness``  area_frac * closure — a single 0..1 CV-likeness score.
    """
    if len(xy) < 4:
        return {"area_frac": 0.0, "closure": 0.0, "loopiness": 0.0}
    x, y = xy[:, 0], xy[:, 1]
    w = float(np.ptp(x)); h = float(np.ptp(y))
    bbox_area = w * h
    if bbox_area <= 0:
        return {"area_frac": 0.0, "closure": 0.0, "loopiness": 0.0}
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    area_frac = min(1.0, area / bbox_area)
    diag = float(np.hypot(w, h)) or 1.0
    gap = float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
    closure = max(0.0, 1.0 - gap / diag)
    return {"area_frac": area_frac, "closure": closure,
            "loopiness": area_frac * closure}


# ---------------------------------------------------------------------------
# 0. Drop isolated stray sub-paths (e.g. legend colour swatches)
# ---------------------------------------------------------------------------
def _arclen(pl: np.ndarray) -> float:
    return float(np.hypot(np.diff(pl[:, 0]), np.diff(pl[:, 1])).sum())


def _endpoint_components(polylines: list[np.ndarray], tol: float) -> list[list[int]]:
    """Union-find sub-paths whose endpoints lie within ``tol`` of each other."""
    n = len(polylines)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    ends = [(pl[0], pl[-1]) for pl in polylines]
    for i in range(n):
        for j in range(i + 1, n):
            (si, ei), (sj, ej) = ends[i], ends[j]
            d = min(np.hypot(*(si - sj)), np.hypot(*(si - ej)),
                    np.hypot(*(ei - sj)), np.hypot(*(ei - ej)))
            if d <= tol:
                union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def keep_main_components(polylines: list[np.ndarray], *, tol_frac: float = 0.05,
                         keep_frac: float = 0.12) -> list[np.ndarray]:
    """Return sub-paths belonging to the significant connected components.

    A CV curve is one or two large components (the anodic/cathodic branches);
    a legend colour swatch is a tiny isolated component drawn in the same
    colour. We cluster sub-paths by endpoint proximity (tolerance = a fraction
    of the group's bbox diagonal) and keep every component whose total arc
    length is at least ``keep_frac`` of the largest — dropping stray swatches
    while preserving both real branches.
    """
    pls = [np.asarray(p, float) for p in polylines if len(p) >= 2]
    if len(pls) <= 1:
        return pls
    pts = np.vstack(pls)
    diag = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) or 1.0
    comps = _endpoint_components(pls, tol_frac * diag)
    sizes = [sum(_arclen(pls[i]) for i in c) for c in comps]
    mx = max(sizes) or 1.0
    kept: list[np.ndarray] = []
    for c, size in zip(comps, sizes):
        if size >= keep_frac * mx:
            kept.extend(pls[i] for i in c)
    return kept or pls


# ---------------------------------------------------------------------------
# 1. Stitch arbitrary-order sub-paths into a single ordered polyline
# ---------------------------------------------------------------------------
def order_curve(polylines: list[np.ndarray]) -> np.ndarray:
    """Greedily connect sub-paths end-to-end into one ordered (N, 2) array.

    Starts at the left-most endpoint (near the lower potential limit) and
    repeatedly appends the nearest remaining sub-path endpoint, flipping that
    sub-path when it is joined by its tail. This recovers the sweep order even
    though the PDF stores the segments shuffled.

    Note: this always incorporates every input sub-path — it trusts its
    caller (``vector_extract``'s colour/shape filters, ``keep_main_components``)
    to have already excluded anything that isn't really part of the curve.
    An earlier version tried to additionally reject "implausibly large"
    gaps here as a second line of defence, but a curve's own legitimate
    internal gaps and a truly unrelated shape's distance turned out not to be
    reliably separable by any single threshold — it silently truncated good
    curves more often than it caught bad ones. Fix contamination at the
    source (shape/colour filtering) instead of guessing distances here.
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


def clean_cv(polylines: list[np.ndarray], *, n_arclength: int = 1000,
             drop_stray: bool = True) -> dict[str, np.ndarray]:
    """Full loop clean-up from raw sub-paths.

    Returns a dict with the ordered ``loop`` and an evenly arc-length-sampled
    ``smooth`` trace, plus the ``forward``/``reverse`` scan branches.
    """
    if drop_stray:
        polylines = keep_main_components(polylines)
    loop = dedupe(order_curve(polylines))
    forward, reverse = split_branches(loop)
    smooth = resample_arclength(loop, n=n_arclength)
    return {"loop": loop, "smooth": smooth, "forward": forward, "reverse": reverse}
