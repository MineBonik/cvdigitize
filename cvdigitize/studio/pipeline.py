"""Thin adapters: existing ``cvdigitize`` functions -> Studio API shapes.

Keeps ``server.py`` request-handling-only — every adapter here just calls
into the already-tested extraction code (see STUDIO_PLAN.md §10 reuse map).
"""
from __future__ import annotations

import base64
import os

import cv2

from ..ingest import classify_pdf, render_page
from ..raster_extract import (crop_tick_labels, detect_axis_ticks,
                              detect_frame_bbox, find_image_regions, render_region)
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


def _png_data_url(rgb) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover - imencode failure is not realistically reachable here
        raise RuntimeError("failed to encode label crop")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# the tick-label-crop keys are named by "which end of the axis"; the Studio
# calibration anchors are named by axis-role (matching the E1/E2/j1/j2 the
# rest of the codebase already uses for a 4-point calibration)
_LABEL_KEY_TO_ANCHOR = {"x_lo": "E1", "x_hi": "E2", "y_lo": "j1", "y_hi": "j2"}


def autocalibrate_crop(workspace_dir: str, paper: str, crop: str) -> dict:
    """/api/autocalibrate: detect the axis frame + tick *positions* only (no
    OCR, no guessed values — STUDIO_PLAN.md §2's locked decision). The human
    reads the returned zoomed label crops and types the 2 values per axis.
    """
    img = ws.load_crop_image(workspace_dir, paper, crop)
    if img is None:
        raise FileNotFoundError(f"{paper}/{crop}")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frame = detect_frame_bbox(gray)
    if frame is None:
        h, w = gray.shape
        margin = 0.05
        frame = (int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin)))
    left, top, right, bottom = frame

    ticks = detect_axis_ticks(gray, frame)
    xt, yt = ticks["x_ticks"], ticks["y_ticks"]

    points = {}
    if len(xt) >= 2:
        points["E1"] = [xt[0], bottom]
        points["E2"] = [xt[-1], bottom]
    if len(yt) >= 2:
        points["j1"] = [left, yt[0]]
        points["j2"] = [left, yt[-1]]

    label_crops = crop_tick_labels(img, frame, ticks)
    label_urls = {_LABEL_KEY_TO_ANCHOR[k]: _png_data_url(im)
                 for k, im in label_crops.items() if k in _LABEL_KEY_TO_ANCHOR}

    return {"points": points, "labelCrops": label_urls, "frame": list(frame)}
