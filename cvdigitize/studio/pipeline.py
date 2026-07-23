"""Thin adapters: existing ``cvdigitize`` functions -> Studio API shapes.

Keeps ``server.py`` request-handling-only — every adapter here just calls
into the already-tested extraction code (see STUDIO_PLAN.md §10 reuse map).
"""
from __future__ import annotations

import os

from ..ingest import classify_pdf, render_page
from ..raster_extract import find_image_regions, render_region
from . import workspace as ws


def analyze_paper(workspace_dir: str, pdf_path: str) -> dict:
    """/api/open_paper: classify every page, extract sources, write analysis.json.

    Embedded images are extracted (higher-res, no page furniture) when a page
    has any; otherwise a full-page render is the fallback source to crop from.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    stem = ws.paper_stem(pdf_path)
    ws.ensure_paper_dir(workspace_dir, stem)

    infos = classify_pdf(pdf_path)
    pages = []
    for pi in infos:
        sources = []
        regions = find_image_regions(pdf_path, pi.number) if pi.n_images else []
        if regions:
            for i, region in enumerate(regions):
                img = render_region(pdf_path, pi.number, region.bbox, zoom=4.0)
                name = f"p{pi.number}_img{i}.png"
                ws.save_source_png(workspace_dir, stem, name, img)
                sources.append(name)
        else:
            img = render_page(pdf_path, pi.number, zoom=3.0)
            name = f"p{pi.number}_full.png"
            ws.save_source_png(workspace_dir, stem, name, img)
            sources.append(name)
        pages.append({
            "n": pi.number, "kind": pi.kind, "n_embedded": len(regions),
            "vector": pi.kind == "vector-curves", "sources": sources,
        })

    analysis = {"pdf_path": pdf_path, "paper": stem, "pages": pages}
    ws.write_analysis(workspace_dir, stem, analysis)
    return analysis
