"""Thin adapters: existing ``cvdigitize`` functions -> Studio API shapes.

Keeps ``server.py`` request-handling-only — every adapter here just calls
into the already-tested extraction code (see STUDIO_PLAN.md §10 reuse map).
"""
from __future__ import annotations

import base64
import math
import os

import cv2
import numpy as np

from ..calibrate import calibration_from_anchors
from ..ingest import classify_pdf, render_page
from ..raster_extract import (_extract_from_frame, crop_tick_labels,
                              detect_axis_ticks, detect_frame_bbox,
                              exclusion_mask, find_image_regions,
                              frame_border_mask, frame_border_thickness,
                              measure_line_and_axis_width, render_region,
                              strip_border_fill_mask)
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
    state = ws.read_state(workspace_dir, stem)
    analysis["last_crop"] = state.get("lastCrop")
    analysis["last_step"] = state.get("lastStep", 1)
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


def _require_crop_image(workspace_dir: str, paper: str, crop: str):
    """Load a crop's image, rejecting both "file missing" (None) and a
    decoded-but-empty array (a degenerate 0-size crop) at this boundary --
    letting either through crashes deep inside OpenCV with a cryptic native
    assertion (``!_src.empty()``) instead of a clear, catchable error."""
    img = ws.load_crop_image(workspace_dir, paper, crop)
    if img is None or img.size == 0:
        raise FileNotFoundError(f"{paper}/{crop}")
    return img


def autocalibrate_crop(workspace_dir: str, paper: str, crop: str) -> dict:
    """/api/autocalibrate: detect the axis frame + tick *positions* only (no
    OCR, no guessed values — STUDIO_PLAN.md §2's locked decision). The human
    reads the returned zoomed label crops and types the 2 values per axis.
    """
    img = _require_crop_image(workspace_dir, paper, crop)

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


def _clean_border_fill(img: np.ndarray) -> np.ndarray:
    """Paint out any solid dark margin/background bleeding in from the
    crop's own outer border (e.g. a scan's black page background). Left in,
    it becomes the single biggest connected dark component, and the
    downstream "keep components >= 5% of the biggest" filter (meant to keep
    every fragment of the SAME curve, drop noise) then discards the real
    curve outright -- a real bug, reported live as "no curve found" on a
    curve plainly visible in the crop.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fill = strip_border_fill_mask(gray)
    if not fill.any():
        return img
    cleaned = img.copy()
    cleaned[fill] = 255
    return cleaned


def _load_crop_and_calibration(workspace_dir: str, paper: str, crop: str):
    img = _require_crop_image(workspace_dir, paper, crop)
    img = _clean_border_fill(img)
    meta = ws.load_crop_meta(workspace_dir, paper, crop)
    calibration = meta.get("calibration") if meta else None
    if not calibration:
        raise ValueError("crop has no saved calibration yet - finish Step 2 first")
    return img, calibration


def measure_crop(workspace_dir: str, paper: str, crop: str) -> dict:
    """/api/measure: line width vs. axis width + the axis-exclusion band (Step 3)."""
    img, calibration = _load_crop_and_calibration(workspace_dir, paper, crop)
    result = measure_line_and_axis_width(img, calibration)
    ws.set_crop_measurement(workspace_dir, paper, crop,
                            line_width=result["line_width"], axis_width=result["axis_width"])
    return result


def _calibration_object(calibration: dict):
    return calibration_from_anchors(
        x_anchor1=(calibration["E1"]["px"][0], calibration["E1"]["value"]),
        x_anchor2=(calibration["E2"]["px"][0], calibration["E2"]["value"]),
        y_anchor1=(calibration["j1"]["px"][1], calibration["j1"]["value"]),
        y_anchor2=(calibration["j2"]["px"][1], calibration["j2"]["value"]),
        x_unit=calibration.get("E_unit", "V"), y_unit=calibration.get("j_unit", ""),
    )


def _score_and_shape_curves(img, calibration: dict, raw_curves: list) -> list:
    """Calibrate + reference-free-fidelity-score a list of {name, rgb, polyline_px}."""
    from ..qc import _fidelity_for_curves

    cal = _calibration_object(calibration)
    fid_input = [{"name": c["name"], "xy": c["polyline_px"], "rgb": c.get("rgb")} for c in raw_curves]
    fids = _fidelity_for_curves(img, fid_input)

    out = []
    for c, fid in zip(raw_curves, fids):
        xy_px = np.asarray(c["polyline_px"], float)
        xy_real = cal.apply(xy_px)
        out.append({
            "name": c["name"], "rgb": list(c.get("rgb") or (0.1, 0.1, 0.1)),
            "xy_px": np.round(xy_px, 2).tolist(), "xy_real": np.round(xy_real, 6).tolist(),
            "fidelity": fid,
        })
    return out


def _mask_axis_and_frame(img: np.ndarray, measurement: dict):
    """Paint the calibrated axis band white -- for BLIND auto-extract, which
    has no guide and could otherwise mistake the plotted axis line itself
    for a whole separate "curve" -- and, when a frame is actually detected,
    its own border too. Returns ``(masked_image, frame_or_None)``.
    """
    excl = exclusion_mask(img.shape, measurement["exclusion_band"])
    masked = img.copy()
    masked[excl] = 255
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frame = detect_frame_bbox(gray)
    if frame is not None:
        border_thickness = frame_border_thickness(gray < 220, frame)
        masked[frame_border_mask(masked.shape, frame, border_thickness + 2)] = 255
    return masked, frame


def _mask_frame_only(img: np.ndarray) -> np.ndarray:
    """Paint out just the detected frame's own border -- for hand-trace and
    re-center, where a human's guide is already strong evidence of where the
    real curve is, including where it legitimately runs close to or crosses
    the plotted axis line (a CV routinely does, near zero current). Blanking
    the axis band there too -- needed for blind auto-extract, see
    :func:`_mask_axis_and_frame` -- would erase genuine curve ink at exactly
    that point instead: a real bug, a hand-traced curve's tail got cut off
    right where it neared the axis, with no ink left there to snap to.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frame = detect_frame_bbox(gray)
    if frame is None:
        return img
    border_thickness = frame_border_thickness(gray < 220, frame)
    masked = img.copy()
    masked[frame_border_mask(masked.shape, frame, border_thickness + 2)] = 255
    return masked


def autoextract_crop(workspace_dir: str, paper: str, crop: str, *,
                     axis_width_override: float | None = None,
                     line_width_override: float | None = None) -> dict:
    """/api/autoextract: mask the axis out, then detect-frame + trace (Step 4).

    Masking ``exclusion_band`` to white before extraction is the same
    technique the crop tool already uses to blank a legend box (paint over,
    don't special-case the extractor) -- cleaner than the strand-level
    ``_is_axis_strand`` heuristic because calibration pins the axis exactly.
    Frame detection runs on the UNMASKED image first: blanking the axis can
    erase most of the frame's own border, and re-running Hough-line frame
    detection on that gutted image risks latching onto a leftover corner
    fragment instead of returning "no frame found" -- detecting once, before
    masking, avoids that trap entirely.

    ``axis_width_override``/``line_width_override`` (from Step 3's sliders)
    let a human correct a bad automatic measurement before extracting.
    """
    img, calibration = _load_crop_and_calibration(workspace_dir, paper, crop)
    measurement = measure_line_and_axis_width(
        img, calibration, axis_width_override=axis_width_override,
        line_width_override=line_width_override)
    masked, frame = _mask_axis_and_frame(img, measurement)
    if frame is None:
        h, w = img.shape[:2]
        margin = 0.05
        frame = (int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin)))

    # A fixed 4px inset can leave a sliver of the frame's own border inside
    # `interior` if that border happens to be drawn thicker (its thickness
    # is NOT assumed equal to the calibrated axis lines' -- they can differ,
    # e.g. a thin 2px axis inside a thicker 6px outer frame). That sliver
    # then gets picked up as "curve ink", and once a real curve fades out
    # near the axis, the stitcher bridges the gap onto it and follows it
    # all the way around the frame (a real bug, seen live: a straight chord
    # shot from the curve's faded end up to a corner and back along two
    # edges). Measuring the frame's own border directly and insetting by at
    # least that (+ margin) keeps the whole border out regardless.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    border_thickness = frame_border_thickness(gray < 220, frame)
    frame_inset_px = max(4, int(math.ceil(border_thickness)) + 3)

    result = _extract_from_frame(masked, frame, value_thresh=0.55,
                                 frame_inset_px=frame_inset_px, max_spur_len=15)
    curves = _score_and_shape_curves(img, calibration, result.get("curves", []))
    return {"curves": curves, "measurement": measurement}


def trace_crop(workspace_dir: str, paper: str, crop: str, guides: list, *,
               axis_width_override: float | None = None,
               line_width_override: float | None = None) -> dict:
    """/api/trace: the hand-trace fallback, always reachable when the
    auto-extract eye check fails (STUDIO_PLAN.md §2) -- snaps human guide
    strokes to ink with the per-stroke-radius extraction G2 fixed.

    Masks only the detected frame's own border out of the ink -- NOT the
    calibrated axis band (that's for autoextract_crop's blind, unguided
    case; see _mask_frame_only). The frame border is never legitimately
    curve data, so masking it is always safe and still prevents a guide
    from escaping onto it and "following" it around the plot (a real bug
    seen live). The axis band, though, a real curve routinely runs close to
    or crosses -- masking it here erased genuine curve ink exactly there,
    a real bug (a hand-traced curve's tail got cut off right where it
    neared the axis).
    """
    from ..guided import extract_guides

    img, calibration = _load_crop_and_calibration(workspace_dir, paper, crop)
    measurement = measure_line_and_axis_width(
        img, calibration, axis_width_override=axis_width_override,
        line_width_override=line_width_override)
    masked = _mask_frame_only(img)

    results = extract_guides(masked, guides, return_gaps=True)
    raw_curves = [{"name": r["name"] or "curve", "rgb": None, "polyline_px": r["polyline_px"]}
                 for r in results]
    # fidelity is scored against the REAL (unmasked) ink -- honest about how
    # well the trace sits on the actual figure, not the whited-out version
    curves = _score_and_shape_curves(img, calibration, raw_curves)
    for c, r in zip(curves, results):
        c["gaps"] = [[list(p0), list(p1)] for p0, p1 in (r.get("gaps") or [])]
    return {"curves": curves, "measurement": measurement}


def accept_curves(workspace_dir: str, paper: str, crop: str, curves: list) -> dict:
    """/api/accept_curves: persist the chosen curves onto the crop (Step 4 -> 5)."""
    return ws.set_crop_curves(workspace_dir, paper, crop, curves)


def _smooth_polyline(xy: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    """Light Savitzky-Golay smoothing along the polyline's own point order.

    Re-projecting each point of an ALREADY dense, precise curve onto its
    local ink center (see recenter_curves) has each point make its own,
    independent choice from noisy pixel-level ink boundaries -- neighbouring
    points don't agree, and the result is jagged/staircased even though each
    individual point is a reasonable center estimate (confirmed live: this
    is exactly what made a decent trace visibly worse after "straightening").
    A local-polynomial smooth removes that point-to-point disagreement while
    still tracking a real sharp peak (unlike a plain moving average, which
    would round it off).
    """
    n = len(xy)
    w = min(window, n - 1 if n % 2 == 0 else n)
    if w < poly + 2 or w < 3:
        return xy
    if w % 2 == 0:
        w -= 1
    from scipy.signal import savgol_filter
    sx = savgol_filter(xy[:, 0], w, poly)
    sy = savgol_filter(xy[:, 1], w, poly)
    return np.column_stack([sx, sy])


def recenter_curves(workspace_dir: str, paper: str, crop: str, curves: list, *,
                    axis_width_override: float | None = None,
                    line_width_override: float | None = None) -> dict:
    """/api/recenter: a one-click "straighten" pass -- re-centers each
    curve's EXISTING polyline on the middle of its local ink width, then
    smooths the point-to-point jaggedness that independent re-projection
    introduces (see _smooth_polyline).

    Snapping during the original trace already centers each point (see
    guided._snap_stroke), but that estimate comes from just the two
    neighbouring guide samples' direction -- not always enough in one pass
    near a sharp peak or a busy junction, where a human can see by eye that
    the trace still hugs one side of the ink. Re-running the SAME centering
    logic with the already-extracted curve as its own guide is a cheap
    second pass that only needs the curve's current points, not the
    original hand-drawn strokes. Masks only the frame border, not the
    calibrated axis band -- see _mask_frame_only.
    """
    from ..guided import extract_near_guide

    img, calibration = _load_crop_and_calibration(workspace_dir, paper, crop)
    measurement = measure_line_and_axis_width(
        img, calibration, axis_width_override=axis_width_override,
        line_width_override=line_width_override)
    masked = _mask_frame_only(img)
    radius = max(3.0, measurement["line_width"] * 1.5)

    raw_curves = []
    for c in curves:
        xy = np.asarray(c["xy_px"], float)
        if len(xy) >= 2:
            recentered = extract_near_guide(masked, xy, radius=radius)
            if len(recentered) >= 5:
                recentered = _smooth_polyline(recentered)
        else:
            recentered = xy
        raw_curves.append({"name": c.get("name", "curve"), "rgb": c.get("rgb"),
                          "polyline_px": recentered if len(recentered) else xy})
    curves_out = _score_and_shape_curves(img, calibration, raw_curves)
    return {"curves": curves_out, "measurement": measurement}


_INDEX_TMPL = """<!doctype html><meta charset="utf-8"><title>CV Studio — {paper}</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem;max-width:900px;color:#1a1a1a}}
h1{{margin-bottom:.2rem}} h2{{margin-top:2rem}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:.4rem .7rem;
  text-align:left;font-size:.9rem}}
code{{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}}
</style>
<h1>CV Studio &mdash; {paper}</h1>
<h2>Crops ({n_crops})</h2>
<table><tr><th>crop</th><th>type</th><th>calibrated</th><th>line/axis width</th><th>curves</th></tr>
{crop_rows}
</table>
<h2>Saved curves ({n_curves})</h2>
<table><tr><th>file</th></tr>
{curve_rows}
</table>
"""


def _write_paper_index(workspace_dir: str, paper: str) -> str:
    """Regenerate the per-paper dashboard (§4 Step 5: "Per-paper index.html
    refreshes")."""
    crops = ws.list_crops(workspace_dir, paper)
    crop_rows = "".join(
        f"<tr><td>{c['name']}</td><td>{c['type']}</td>"
        f"<td>{'yes' if c.get('calibration') else 'no'}</td>"
        f"<td>{c.get('lineWidth') or '—'} / {c.get('axisWidth') or '—'}</td>"
        f"<td>{len(c.get('curves') or [])}</td></tr>"
        for c in crops
    ) or "<tr><td colspan=5><i>none yet</i></td></tr>"

    curve_files = ws.list_curve_files(workspace_dir, paper)
    curve_rows = "".join(f"<tr><td><code>curves/{f}</code></td></tr>" for f in curve_files) \
        or "<tr><td><i>none yet</i></td></tr>"

    html = _INDEX_TMPL.format(paper=paper, n_crops=len(crops), crop_rows=crop_rows,
                              n_curves=len(curve_files), curve_rows=curve_rows)
    path = os.path.join(ws.paper_dir(workspace_dir, paper), "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def save_curves(workspace_dir: str, paper: str, crop: str, curves: list) -> dict:
    """/api/save_curves: write each curve as an echemdb datapackage (Step 5).

    Naming (§9): ``{paper_stem}_{figtag}_{label}`` -- figtag defaults to the
    crop name, editable to echemdb ``f2a`` style; label is the sanitized
    curve name. Carries the calibration and source crop in the metadata
    comment so provenance is recoverable.
    """
    from ..package import CurveMeta, write_datapackage

    crop_meta = ws.load_crop_meta(workspace_dir, paper, crop)
    if crop_meta is None:
        raise FileNotFoundError(f"{paper}/{crop}")
    calibration = crop_meta.get("calibration") or {}
    analysis = ws.read_analysis(workspace_dir, paper) or {}
    out_dir = ws.curves_dir(workspace_dir, paper)
    os.makedirs(out_dir, exist_ok=True)

    # crop names are always "{paper}_crop{n}"; the default figtag is just the
    # "crop{n}" suffix (§9) -- using the crop name as-is would double the
    # paper prefix into the filename ("{paper}_{paper}_crop1_...")
    default_figtag = crop[len(paper) + 1:] if crop.startswith(paper + "_") else crop

    written = []
    for c in curves:
        figtag = ws.safe_name(c.get("figtag") or default_figtag)
        label = ws.safe_name(c.get("label") or c.get("name") or "curve")
        name = f"{paper}_{figtag}_{label}"
        data = np.asarray(c["xy_real"], float)
        meta = CurveMeta(
            name=name, figure=figtag, curve=c.get("label") or c.get("name") or "",
            x_label="E", x_unit=calibration.get("E_unit", "V"),
            y_label="j", y_unit=calibration.get("j_unit", ""),
            source_pdf=analysis.get("pdf_path", ""), method="digitized",
            comment=f"CV Studio: crop={crop}, type={crop_meta.get('type', '')}",
        )
        paths = write_datapackage(out_dir, data, meta, yaml=True)
        written.append({k: os.path.relpath(v, ws.paper_dir(workspace_dir, paper)).replace("\\", "/")
                        for k, v in paths.items()})

    _write_paper_index(workspace_dir, paper)
    ws.touch_state(workspace_dir, paper, crop=crop, step=5,
                   decision=f"saved {len(curves)} curve(s) from {crop} to curves/")
    return {"written": written}
