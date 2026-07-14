"""PDF ingest & figure classification (M1/M2 boundary).

Decides, per page, whether CV curves can be recovered from vector geometry
(the high-fidelity M0/M1 path) or whether the figure is a raster image that
would need image-based tracing (M2, not yet implemented). Also renders pages
to PNG for overlays and future raster work.
"""
from __future__ import annotations

from dataclasses import dataclass

import fitz
import numpy as np


@dataclass
class PageInfo:
    number: int
    n_drawings: int
    n_curve_items: int
    n_stroke_colors: int
    n_images: int
    image_area_frac: float
    kind: str            # "vector-curves" | "raster" | "sparse"


def _page_curve_stats(page) -> tuple[int, int, set]:
    drawings = page.get_drawings()
    n_items = 0
    colors = set()
    for d in drawings:
        if d.get("color") is not None:
            colors.add(tuple(round(c, 3) for c in d["color"]))
        for it in d["items"]:
            if it[0] in ("l", "c"):
                n_items += 1
    return len(drawings), n_items, colors


def _image_area_fraction(page) -> float:
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    covered = 0.0
    for img in page.get_images(full=True):
        try:
            for r in page.get_image_rects(img[0]):
                covered += abs(r.width * r.height)
        except Exception:
            pass
    return min(1.0, covered / page_area)


def classify_page(page) -> PageInfo:
    n_draw, n_items, colors = _page_curve_stats(page)
    n_images = len(page.get_images(full=True))
    img_frac = _image_area_fraction(page)

    # Heuristic: many vector line/curve items with >=2 stroke colours -> real
    # vector plot. A page dominated by a large raster with few vectors -> raster.
    if n_items >= 500 and len(colors) >= 2:
        kind = "vector-curves"
    elif img_frac > 0.25 and n_items < 500:
        kind = "raster"
    else:
        kind = "sparse"
    return PageInfo(page.number, n_draw, n_items, len(colors), n_images, img_frac, kind)


def classify_pdf(pdf_path: str) -> list[PageInfo]:
    doc = fitz.open(pdf_path)
    infos = [classify_page(doc[p]) for p in range(doc.page_count)]
    doc.close()
    return infos


def render_page(pdf_path: str, page_number: int, zoom: float = 3.0) -> np.ndarray:
    """Render a page to an RGB numpy array (H, W, 3)."""
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    arr = img[:, :, :3].copy()
    doc.close()
    return arr
