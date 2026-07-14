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
        # Is pt (approximately) the last point of the current polyline?
        if not current:
            return True
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
    return [np.asarray(p, dtype=float) for p in polylines]


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
) -> list[CurveGroup]:
    """Extract stroked curves from ``page_number`` grouped by stroke colour.

    Parameters
    ----------
    min_points : drop colour groups thinner than this (kills stray marks).
    clip : optional (x0, y0, x1, y1) in PDF points; keep only polylines whose
        centroid falls inside it (used to drop legends/axis labels).
    drop_light : skip pale grey strokes (grid lines / frames).
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    groups: dict[tuple[float, float, float], CurveGroup] = {}

    for d in page.get_drawings():
        color = d.get("color")
        if color is None:  # unstroked (fill-only) element -> skip
            continue
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
    """Grid cell -> figure letter, row-major from top-left: (0,0)->'a'."""
    return chr(ord("a") + row * nx + col)


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
