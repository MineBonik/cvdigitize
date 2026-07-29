"""Vector curve extraction from a PDF figure (M0).

The core insight from the rizo_2025 reference dataset: each CV curve in a
vector PDF is drawn with a distinct stroke colour. Vladislav separated those
curves by hand (merging out-of-order path segments, deleting the other
curves). This module automates that separation directly from the PDF's vector
geometry using PyMuPDF's ``page.get_drawings()``.

A curve in the PDF is stored as many disconnected sub-paths (line and Bezier
segments) in arbitrary drawing order. Here we (1) group every sub-path by its
stroke colour and (2) flatten each sub-path into a polyline of (x, y) points.
Ordering the sub-paths into one continuous loop is a later step (M1); for M0 we
keep the sub-paths so we can prove the colour separation is correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import fitz  # PyMuPDF
import numpy as np

# ---------------------------------------------------------------------------
# Known curve colours (RGB 0..1) -> human name, taken from the rizo reference.
# Used only for friendly labelling/filenames; extraction never relies on it.
# ---------------------------------------------------------------------------
KNOWN_COLORS: dict[tuple[float, float, float], str] = {
    (0.0, 0.0, 0.0): "black",
    (0.0, 0.0, 1.0): "blue",
    (0.502, 0.502, 0.0): "khaki",
    (1.0, 0.502, 0.0): "orange",
    (1.0, 0.0, 1.0): "pink",
    (1.0, 0.0, 0.0): "red",
    (0.502, 0.0, 0.502): "violet",
}


def color_name(rgb: tuple[float, float, float]) -> str:
    """Nearest known colour name, else a hex-ish fallback like ``c_804000``."""
    best, best_d = None, 1e9
    for known, name in KNOWN_COLORS.items():
        d = sum((a - b) ** 2 for a, b in zip(rgb, known))
        if d < best_d:
            best, best_d = name, d
    if best_d <= 0.02:  # within tolerance of a known curve colour
        return best
    r, g, b = (int(round(c * 255)) for c in rgb)
    return f"c_{r:02x}{g:02x}{b:02x}"


def _bezier_points(p0, p1, p2, p3, n: int = 12) -> list[tuple[float, float]]:
    """Sample a cubic Bezier into ``n`` points (endpoints included)."""
    ts = np.linspace(0.0, 1.0, n)
    mt = 1.0 - ts
    x = (mt**3 * p0.x + 3 * mt**2 * ts * p1.x + 3 * mt * ts**2 * p2.x + ts**3 * p3.x)
    y = (mt**3 * p0.y + 3 * mt**2 * ts * p1.y + 3 * mt * ts**2 * p2.y + ts**3 * p3.y)
    return list(zip(x.tolist(), y.tolist()))


def _is_thin_line_dims(w: float, h: float, *, thin_thresh: float = 2.0,
                       long_thresh: float = 15.0) -> bool:
    """True for dimensions shaped like an axis/frame line, not a data marker.

    Some PDF exporters draw axis borders, tick marks and gridlines as thin
    *filled* rectangles rather than stroked lines (e.g. a "line" 0.8pt thick
    and 210pt long) — sometimes via the ``re`` opcode, sometimes as four
    ``l`` (lineto) segments tracing the same shape. Either way, walking all 4
    corners of such a rectangle necessarily includes an edge equal to its long
    dimension, which — if wrongly treated as curve data — shows up as a huge,
    spurious jump in the digitized trace. A real data-point marker is small in
    *both* dimensions; a frame/tick line is thin in exactly one. That
    asymmetry is what this check keys on.
    """
    return min(w, h) <= thin_thresh and max(w, h) >= long_thresh


def _is_axis_aligned_rect(poly: np.ndarray, *, tol: float = 0.5) -> bool:
    """True if every point sits on one of exactly 2 x-levels and 2 y-levels.

    A frame/tick rectangle traced as 4-5 points has *exactly* two distinct x
    positions and two distinct y positions (perfectly axis-aligned corners). A
    real curve segment does not, even where it happens to be short, thin and
    steep (e.g. near a sharp peak) — its points vary continuously rather than
    snapping to two discrete levels. Requiring this shape, not just thinness,
    avoids mistaking a genuine steep/thin curve segment for a frame line.
    """
    ux = np.unique(np.round(poly[:, 0] / tol))
    uy = np.unique(np.round(poly[:, 1] / tol))
    return len(ux) <= 2 and len(uy) <= 2


def _is_thin_line_polyline(poly: np.ndarray, *, max_corners: int = 5, **kwargs) -> bool:
    """True if a small closed polyline is shaped like an axis/frame line.

    Catches a thin frame/tick rectangle whether it was drawn as a single
    ``re`` op or as a sequence of ``l`` (lineto) ops tracing the same shape —
    both end up here as a small (<= ``max_corners``-point), axis-aligned,
    thin-and-long polyline.
    """
    if len(poly) > max_corners or not _is_axis_aligned_rect(poly):
        return False
    w = float(poly[:, 0].max() - poly[:, 0].min())
    h = float(poly[:, 1].max() - poly[:, 1].min())
    return _is_thin_line_dims(w, h, **kwargs)


def _is_narrow_tall_column(poly: np.ndarray, *, max_width: float = 15.0,
                          min_height: float = 100.0) -> bool:
    """True for a shape far narrower than it is tall — a text column or
    page-divider rule, not a CV curve.

    Some PDFs render body text (or decorative column rules) with *stroke*
    colour rather than fill, so it can land in the same colour group as a
    real stroked curve and, if several glyphs merge into one long drawing, get
    force-stitched in as a spurious near-vertical "curve". A genuine CV
    branch always has comparable extent in both directions — even the
    steepest real peak in this project's reference data never exceeds ~40pt
    of height while under 15pt wide (see rizo Pt(111)); this heuristic uses
    100pt as a wide safety margin above that.
    """
    w = float(poly[:, 0].max() - poly[:, 0].min())
    h = float(poly[:, 1].max() - poly[:, 1].min())
    return w <= max_width and h >= min_height


def _is_tick_mark(poly: np.ndarray, *, max_points: int = 3, max_diag: float = 12.0) -> bool:
    """True for a tiny, isolated 2-3 point segment — an axis tick, not curve data.

    Axis tick marks are typically drawn as their own short, standalone
    straight segment (one PDF "l" op each) in whatever colour the axis uses
    (often black, regardless of which curve is nearby) — so they can land in
    a real curve's colour group and skew its bounding box, distorting shape
    comparisons even though they never affect the stitched trace itself
    (their length is negligible next to the real curve). A real digitized
    curve segment, even a coarse one, spans far more than a tick's ~1-10pt
    diagonal; this project's reference data shows every genuine ≤3-point
    fragment already under this size is a tick (never legitimate curve data).
    """
    if len(poly) > max_points:
        return False
    diag = float(np.hypot(poly[:, 0].max() - poly[:, 0].min(),
                          poly[:, 1].max() - poly[:, 1].min()))
    return diag <= max_diag


def _items_to_polylines(items, bezier_samples: int = 12) -> list[np.ndarray]:
    """Flatten one drawing's item list into a list of polylines (Nx2 arrays).

    Each ``item`` is a tuple whose first element is the op code: ``"l"`` (line),
    ``"c"`` (cubic Bezier), ``"re"`` (rectangle), ``"qu"`` (quad). We split into
    a new polyline whenever the geometry is not contiguous with the previous
    point (a pen-up), so a single drawing may yield several polylines.
    """
    polylines: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    def _flush():
        nonlocal current
        if len(current) >= 2:
            polylines.append(current)
        current = []

    def _cont(pt) -> bool:
        # Is pt (approximately) the last point of the current polyline? An
        # empty `current` is never "continuous" -- it forces pt to be
        # appended as the new starting point instead of being silently
        # dropped (there is nothing yet for it to be a continuation of).
        if not current:
            return False
        lx, ly = current[-1]
        return abs(lx - pt.x) < 1e-3 and abs(ly - pt.y) < 1e-3

    for it in items:
        op = it[0]
        if op == "l":
            _, p1, p2 = it
            if not _cont(p1):
                _flush()
                current.append((p1.x, p1.y))
            current.append((p2.x, p2.y))
        elif op == "c":
            _, p1, p2, p3, p4 = it
            if not _cont(p1):
                _flush()
                current.append((p1.x, p1.y))
            pts = _bezier_points(p1, p2, p3, p4, bezier_samples)
            current.extend(pts[1:])  # p1 already present
        elif op == "re":  # rectangle -> its own closed polyline
            _flush()
            rect = it[1]
            current = [
                (rect.x0, rect.y0), (rect.x1, rect.y0),
                (rect.x1, rect.y1), (rect.x0, rect.y1), (rect.x0, rect.y0),
            ]
            _flush()
        elif op == "qu":  # quad -> polyline through its 4 corners
            _flush()
            q = it[1]
            current = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y),
                       (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
            _flush()
    _flush()
    arrays = [np.asarray(p, dtype=float) for p in polylines]
    return [p for p in arrays
           if not _is_thin_line_polyline(p) and not _is_narrow_tall_column(p)
           and not _is_tick_mark(p)]


@dataclass
class CurveGroup:
    """All sub-paths drawn in a single stroke colour on a page."""

    rgb: tuple[float, float, float]
    name: str
    polylines: list[np.ndarray] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return int(sum(len(p) for p in self.polylines))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        pts = np.vstack(self.polylines)
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))


def _round_rgb(rgb, ndigits: int = 3) -> tuple[float, float, float]:
    return tuple(round(float(c), ndigits) for c in rgb)


def is_light_gray(rgb: tuple[float, float, float]) -> bool:
    """True for pale/grey strokes (grid lines, frames) — not curve colours.

    Keeps black (0,0,0) and any saturated colour; drops light near-neutral
    greys where every channel is high and close together.
    """
    lo, hi = min(rgb), max(rgb)
    return lo > 0.6 and (hi - lo) < 0.15


def extract_color_groups(
    pdf_path: str,
    page_number: int,
    *,
    min_points: int = 50,
    clip: tuple[float, float, float, float] | None = None,
    bezier_samples: int = 12,
    drop_light: bool = True,
    use_fill_fallback: bool = True,
) -> list[CurveGroup]:
    """Extract curves from ``page_number`` grouped by colour.

    Most PDFs draw curves as *stroked* paths (``color``). Some plotting/export
    tools instead convert each stroke to its own thin filled outline polygon
    (``color`` is ``None``, only ``fill`` is set) — a "ribbon" tracing both
    sides of the line width rather than a single centreline. When
    ``use_fill_fallback`` is set, such shapes are grouped by fill colour
    instead of being silently dropped; the ribbon is thin enough that the
    downstream loop-ordering/resampling still produces a usable curve.

    Fill-colour grouping only kicks in for a colour that has **no** stroked
    drawings at all on the page — fill-only shapes are also how PDF text is
    rendered (each glyph is a small filled outline), and merging glyph shapes
    into an already-good stroke-based curve group (e.g. black axis-label text
    into a black stroked curve) would corrupt it. A colour with real stroke
    data never needs the fallback anyway.

    Parameters
    ----------
    min_points : drop colour groups thinner than this (kills stray marks).
    clip : optional (x0, y0, x1, y1) in PDF points; keep only polylines whose
        centroid falls inside it (used to drop legends/axis labels).
    drop_light : skip pale grey strokes (grid lines / frames).
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    drawings = page.get_drawings()
    stroked_rgbs = {_round_rgb(d["color"]) for d in drawings if d.get("color") is not None}
    groups: dict[tuple[float, float, float], CurveGroup] = {}

    for d in drawings:
        color = d.get("color")
        if color is None:
            if not use_fill_fallback:
                continue
            color = d.get("fill")
            if color is None:
                continue
            if _round_rgb(color) in stroked_rgbs:
                continue  # this colour already has real stroke data; don't add glyph noise
        rgb = _round_rgb(color)
        if drop_light and is_light_gray(rgb):
            continue
        polylines = _items_to_polylines(d["items"], bezier_samples)
        if clip is not None:
            x0, y0, x1, y1 = clip
            polylines = [
                p for p in polylines
                if x0 <= p[:, 0].mean() <= x1 and y0 <= p[:, 1].mean() <= y1
            ]
        if not polylines:
            continue
        grp = groups.get(rgb)
        if grp is None:
            grp = groups[rgb] = CurveGroup(rgb=rgb, name=color_name(rgb))
        grp.polylines.extend(polylines)

    doc.close()
    result = [g for g in groups.values() if g.n_points >= min_points]
    result.sort(key=lambda g: g.n_points, reverse=True)
    return result


def union_bbox(groups: list["CurveGroup"]) -> tuple[float, float, float, float]:
    """Bounding box covering every polyline across ``groups`` (PDF points)."""
    pts = np.vstack([np.vstack(g.polylines) for g in groups])
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _cell_of(pl: np.ndarray, grid: tuple[float, float, float, float],
             nx: int, ny: int) -> tuple[int, int]:
    x0, y0, x1, y1 = grid
    dx = (x1 - x0) / nx or 1.0
    dy = (y1 - y0) / ny or 1.0
    col = min(nx - 1, max(0, int((pl[:, 0].mean() - x0) / dx)))
    row = min(ny - 1, max(0, int((pl[:, 1].mean() - y0) / dy)))
    return col, row


def split_into_panels(group: "CurveGroup", nx: int, ny: int, *,
                      grid_bbox: tuple[float, float, float, float] | None = None,
                      ) -> dict[tuple[int, int], "CurveGroup"]:
    """Partition a colour group's sub-paths into an ``nx`` by ``ny`` panel grid.

    Assign each sub-path to a grid cell by its centroid. Pass ``grid_bbox`` to
    use a shared grid (recommended for multi-panel figures — a per-colour bbox
    is skewed when a colour is absent from some panels). Returns
    ``{(col, row): CurveGroup}`` with row 0 at the top (smallest PDF y); panel
    (a) is ``(0, 0)``.
    """
    grid = grid_bbox if grid_bbox is not None else group.bbox
    cells: dict[tuple[int, int], CurveGroup] = {}
    for pl in group.polylines:
        key = _cell_of(pl, grid, nx, ny)
        cell = cells.get(key)
        if cell is None:
            cell = cells[key] = CurveGroup(rgb=group.rgb, name=group.name)
        cell.polylines.append(pl)
    return cells


def extract_panels(
    pdf_path: str,
    page_number: int,
    nx: int,
    ny: int,
    *,
    min_points: int = 20,
    min_curve_points: int = 200,
    bezier_samples: int = 12,
) -> dict[tuple[int, int], list["CurveGroup"]]:
    """Panel-first extraction: return ``{(col, row): [CurveGroup, ...]}``.

    Curves are localised to a shared panel grid FIRST, then grouped by stroke
    colour WITHIN each panel. This is robust to a curve being drawn with a
    slightly different RGB in different panels (a real quirk in the rizo
    figure, where panel (a)'s violet is a bluer purple than the others).
    """
    all_groups = extract_color_groups(pdf_path, page_number, min_points=min_points,
                                       bezier_samples=bezier_samples)
    if not all_groups:
        return {}
    grid = union_bbox(all_groups)

    # (col,row) -> rgb -> CurveGroup
    panels: dict[tuple[int, int], dict[tuple, CurveGroup]] = {}
    for g in all_groups:
        for pl in g.polylines:
            key = _cell_of(pl, grid, nx, ny)
            bucket = panels.setdefault(key, {})
            cg = bucket.get(g.rgb)
            if cg is None:
                cg = bucket[g.rgb] = CurveGroup(rgb=g.rgb, name=g.name)
            cg.polylines.append(pl)

    result: dict[tuple[int, int], list[CurveGroup]] = {}
    for key, bucket in panels.items():
        curves = [c for c in bucket.values() if c.n_points >= min_curve_points]
        curves.sort(key=lambda c: c.n_points, reverse=True)
        if curves:
            result[key] = curves
    return result


def panel_label(col: int, row: int, nx: int) -> str:
    """Grid cell -> figure letter, row-major from top-left: (0,0)->'a'.

    Past the 26th panel it continues 'aa', 'ab', ... rather than running off the
    end of the alphabet. A bare ``chr(ord("a") + n)`` used to emit control
    characters there, which then became unusable ``panel_\\x85`` directory names
    (a real crash on pages where frame detection over-segments into 20+ panels).
    """
    n = row * nx + col
    label = ""
    while True:
        n, rem = divmod(n, 26)
        label = chr(ord("a") + rem) + label
        if n == 0:
            break
        n -= 1
    return label


@dataclass
class Panel:
    """One detected plot within a page: its axes frame and the curves in it."""

    label: str
    frame_pdf: tuple[float, float, float, float]   # (x0, y0, x1, y1) PDF points
    curves: list[CurveGroup]


def detect_panels(
    pdf_path: str,
    page_number: int,
    *,
    min_points: int = 60,
    zoom: float = 3.0,
    min_curve_points: int = 150,
    image=None,
    frames_px=None,
) -> list[Panel]:
    """Detect each plot's axes frame and assign curves to the frame that holds
    them — automatic panel splitting for ANY layout (grids, or several
    separate figures stacked on one page), replacing a hand-specified grid.

    A colour group's sub-paths are distributed to whichever detected frame
    contains their centroid; sub-paths outside every frame (legend swatches,
    stray marks) are dropped for free. Panels are lettered in reading order
    (top-to-bottom, then left-to-right). Frames that end up with too few curve
    points (SEM insets, empty axes) are discarded.

    Falls back to a single whole-page panel when no frames are found, so the
    caller always gets at least one panel to work with.

    ``image`` and ``frames_px`` let a caller that already rendered this page at
    ``zoom``, or already ran frame detection on it, hand those in instead of
    paying for them twice. Both were the two most expensive steps of a folder
    scan; passing them is purely an optimisation and changes no results.
    """
    import cv2

    from .ingest import render_page
    from .raster_extract import detect_all_frames

    groups = extract_color_groups(pdf_path, page_number, min_points=min_points)
    if not groups:
        return []

    if frames_px is None:
        img = render_page(pdf_path, page_number, zoom=zoom) if image is None else image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        frames_px = detect_all_frames(gray)
    # to PDF points, with a small outward margin (curves can touch the border)
    margin = 3.0
    frames = [(f[0] / zoom - margin, f[1] / zoom - margin,
               f[2] / zoom + margin, f[3] / zoom + margin) for f in frames_px]

    if not frames:
        whole = [g for g in groups if g.n_points >= min_curve_points]
        return [Panel("", union_bbox(whole) if whole else (0, 0, 0, 0), whole)] if whole else []

    # reading order for labels
    order = sorted(range(len(frames)),
                   key=lambda i: (round(frames[i][1] / 20), frames[i][0]))
    # panel_label continues "aa", "ab", ... past the 26th frame. A bare
    # chr(ord("a") + k) emitted control characters there, and the label goes
    # straight into a directory name — pages where frame detection finds 27+
    # frames really do occur (climent_2017 p1 finds 20).
    label_of = {fi: panel_label(k, 0, len(order) or 1) for k, fi in enumerate(order)}

    panels: dict[int, dict[tuple, CurveGroup]] = {i: {} for i in range(len(frames))}
    for g in groups:
        for pl in g.polylines:
            cx, cy = pl[:, 0].mean(), pl[:, 1].mean()
            for fi, (fx0, fy0, fx1, fy1) in enumerate(frames):
                if fx0 <= cx <= fx1 and fy0 <= cy <= fy1:
                    bucket = panels[fi]
                    cg = bucket.get(g.rgb)
                    if cg is None:
                        cg = bucket[g.rgb] = CurveGroup(rgb=g.rgb, name=g.name)
                    cg.polylines.append(pl)
                    break  # a point belongs to at most one (first) frame

    result: list[Panel] = []
    for fi in order:
        curves = [c for c in panels[fi].values() if c.n_points >= min_curve_points]
        curves.sort(key=lambda c: c.n_points, reverse=True)
        if curves:
            result.append(Panel(label_of[fi], frames[fi], curves))
    return result


def find_figure_pages(pdf_path: str, *, min_colors: int = 2,
                      min_points: int = 60) -> list[int]:
    """Heuristic: pages that carry several multi-colour vector curves."""
    doc = fitz.open(pdf_path)
    pages: list[int] = []
    for pno in range(doc.page_count):
        groups = extract_color_groups(pdf_path, pno, min_points=min_points)
        colored = [g for g in groups if g.name != "black"]
        if len(colored) >= min_colors and groups:
            pages.append(pno)
    doc.close()
    return pages
