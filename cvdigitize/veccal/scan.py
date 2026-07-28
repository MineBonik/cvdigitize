"""Find every vector CV panel in a folder of PDFs and prepare it for calibration.

This is the "collect the work" half of ``cvdigitize vector-calibrate``. It walks
a folder, keeps only pages that hold native vector graphics, extracts the curves
mathematically from the PDF's own path geometry, drops panels that are not
cyclic voltammograms, and writes one **work unit per panel** — deliberately not
per curve, because every curve inside a panel shares that panel's axes and
therefore its calibration.

Geometry lives in two spaces:

* **PDF points** — the canonical space. Curve loops, the axes frame and the
  calibration anchors are all stored here, which is the space
  :class:`cvdigitize.calibrate.Calibration` already works in, so a calibration
  built from this data applies to the curves with no conversion.
* **Crop pixels** — display only. The panel image is the rendered page cropped
  around the axes frame with enough margin to keep the tick labels and axis
  titles readable, since a human has to read those numbers off the picture.
  ``crop_px = pdf_pt * zoom - origin``; the browser converts back when it
  reports a clicked anchor.

Each unit ships with a pre-filled calibration guess whenever the page's tick
labels are real text (``autocalib.match_calibration``), plus every tick position
we could detect as click-to-snap targets for when the guess is wrong or absent.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from ..autocalib import (find_axis_label_sets, find_axis_units,
                         match_calibration)
from ..ingest import render_page
from ..metadata import (detect_plot_legend, extract_figure_metadata,
                        legend_for_panel, parse_curve_legend)
from ..paper_meta import extract_paper_metadata
from ..postprocess import dedupe, keep_main_components, loop_metrics, order_curve
from ..vector_extract import (color_name, detect_panels, find_figure_pages,
                              union_bbox)

ZOOM = 3.0
#: PDF points of margin kept around a panel's frame so its tick labels, axis
#: titles and legend stay visible in the cropped image the user calibrates from.
MARGIN_PT = 64.0

_FIG_NO = re.compile(r"\bfig(?:ure)?\.?\s*([0-9]+)", re.I)


def _fig_number(caption: str) -> str:
    m = _FIG_NO.search(caption or "")
    return m.group(1) if m else ""


#: Hue centres (degrees) for the descriptive colour namer below.
_HUES = [(0, "red"), (20, "orange"), (45, "amber"), (58, "yellow"),
         (72, "khaki"), (100, "green"), (160, "teal"), (185, "cyan"),
         (210, "azure"), (240, "blue"), (270, "violet"), (292, "purple"),
         (320, "magenta"), (345, "red")]


def _describe_color(rgb) -> str:
    """A readable colour word for any RGB, e.g. "dark-gray", "red", "violet".

    :func:`cvdigitize.vector_extract.color_name` is deliberately conservative —
    it only names the seven reference curve colours and otherwise falls back to
    a hex string, a contract other code and its tests rely on. Filenames want a
    word every time, so this defers to that function whenever it commits to a
    name (keeping veccal's filenames consistent with the rest of the tool) and
    only falls back to a hue/lightness description for everything else.
    """
    known = color_name(rgb)
    if not known.startswith("c_"):
        return known

    r, g, b = (max(0.0, min(1.0, float(c))) for c in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    val, chroma = mx, mx - mn
    sat = chroma / mx if mx > 0 else 0.0

    if sat < 0.16:                       # neutral: name by lightness
        if val < 0.16:
            return "black"
        if val < 0.42:
            return "dark-gray"
        if val < 0.72:
            return "gray"
        if val < 0.94:
            return "light-gray"
        return "white"

    if mx == r:
        hue = 60.0 * (((g - b) / chroma) % 6.0)
    elif mx == g:
        hue = 60.0 * (((b - r) / chroma) + 2.0)
    else:
        hue = 60.0 * (((r - g) / chroma) + 4.0)

    name = min(_HUES, key=lambda h: min(abs(hue - h[0]), 360 - abs(hue - h[0])))[1]
    if val < 0.42:
        return f"dark-{name}"
    if sat < 0.42 and val > 0.86:
        return f"light-{name}"
    return name


def _slug(text: str, *, maxlen: int = 40) -> str:
    """Filename-safe fragment of a legend label ("0.02 M HClO4" -> "0.02M-HClO4")."""
    s = re.sub(r"\s*\+\s*", "+", (text or "").strip())
    s = re.sub(r"(\d)\s+([A-Za-z])", r"\1\2", s)      # "0.02 M" -> "0.02M"
    s = re.sub(r"[^\w.+()-]+", "-", s, flags=re.UNICODE)
    return s.strip("-_.")[:maxlen]


@dataclass
class CurveUnit:
    """One digitized curve inside a panel, before calibration."""

    color: str                      # machine colour key, e.g. "c_f14040"
    color_label: str                # human colour, e.g. "red"
    rgb: list                       # 0..1 floats, for drawing
    legend: str                     # in-figure legend text, "" if none found
    name: str                       # suggested filename stem (user-editable)
    sample: str                     # suggested sample label for the metadata
    n_points: int
    loopiness: float
    loop_pdf: list = field(default_factory=list)     # [[x, y], ...] PDF points


@dataclass
class PanelUnit:
    """One plot panel: the calibration unit. Every curve here shares its axes."""

    uid: str
    pdf: str
    page: int
    stem: str
    panel: str                      # "a", "b", ... or "" for a lone plot
    figure: str                     # "1a", "3", ... best effort from the caption
    image: str                      # panel PNG, relative to the work dir
    zoom: float
    origin: list                    # [ox, oy] render px of the crop's top-left
    size: list                      # [w, h] px of the crop
    frame_pdf: list                 # panel axes frame in PDF points
    curves: list = field(default_factory=list)
    calib_guess: dict | None = None      # pre-filled Calibration, PDF-pt anchors
    calib_source: str = "none"           # "auto-text" | "none"
    x_ticks: list = field(default_factory=list)   # snap targets, PDF pt + value
    y_ticks: list = field(default_factory=list)
    axis_titles: list = field(default_factory=list)   # [x_title, y_title]
    #: Units read off the axis titles, "" where the figure did not say. Never
    #: guessed: a CV in mA cm-2 saved as uA/cm2 is wrong by 1000x and looks
    #: entirely plausible, so the user fills a blank rather than correcting a
    #: default they have no reason to distrust.
    axis_units: list = field(default_factory=list)     # [x_unit, y_unit]
    figure_meta: dict = field(default_factory=dict)   # scan rate, electrolyte...
    paper_meta: dict = field(default_factory=dict)    # doi, title, journal...
    status: str = "pending"              # "pending" | "saved" | "skipped"


def _ticks_from_labels(label_sets, orientation: str,
                       frame_pdf: tuple[float, float, float, float]) -> list[dict]:
    """Tick labels for the axis of ``frame_pdf``, as {pos, value} in PDF points.

    Picks the label row/column closest to the relevant frame edge that also
    spans it, then returns its individual labels so the browser can offer each
    printed number as a snap target.
    """
    x0, y0, x1, y1 = frame_pdf
    best, best_gap = None, 1e9
    for s in label_sets:
        if s.orientation != orientation:
            continue
        lo, hi = s.pixel_span
        if orientation == "x":
            # the row of numbers sits just below the frame and spans its width
            gap = s.line_coord - y1
            spans = lo <= x1 and hi >= x0
        else:
            gap = x0 - s.line_coord
            spans = lo <= y1 and hi >= y0
        if spans and -8 <= gap < best_gap:
            best, best_gap = s, gap
    if best is None:
        return []
    out = []
    for lab in best.labels:
        pos = lab.cx if orientation == "x" else lab.cy
        out.append({"pos": round(float(pos), 3), "value": float(lab.value)})
    return sorted(out, key=lambda t: t["pos"])


def _axis_units(pdf: str, page: int, frame_pdf) -> tuple[str, str]:
    try:
        return find_axis_units(pdf, page, frame_pdf)
    except Exception:
        return ("", "")


def _tick_detection(pdf: str, page: int, bbox, img):
    """Axes frame + tick-mark positions for a panel, reusing the page render."""
    from ..autocalib import detect_ticks_for_bbox
    try:
        return detect_ticks_for_bbox(pdf, page, bbox, zoom=ZOOM, image=img)
    except Exception:
        return None


def _ocr_calibration(det: dict):
    """Calibration from detected ticks + OCR'd outer labels, or None.

    Only fires when both axes read confidently (``ocr.read_axis_values``
    enforces that), because a half-read axis would silently mis-scale the data.
    The result is still shown to the user for confirmation.
    """
    from .. import ocr
    from ..autocalib import assisted_tick_calibration
    from ..raster_extract import crop_tick_labels

    if not ocr.available():
        return None
    try:
        crops = crop_tick_labels(det["image"], det["frame_px"], det["ticks"],
                                 zoom=det["zoom"])
        x_pair, y_pair = ocr.read_axis_values(crops)
        if not x_pair or not y_pair:
            return None
        return assisted_tick_calibration(det, x_pair, y_pair,
                                         x_unit="", y_unit="")
    except Exception:
        return None


def _snap_to_ticks(cal, x_ticks: list[dict], y_ticks: list[dict]):
    """Move a calibration's anchors onto real tick marks, same mapping.

    ``autocalib.match_calibration`` anchors at the *curve's* bounding box, so it
    is exact but its numbers are wherever the ink happens to stop — 0.0731 V
    rather than 0.2 V. A human is being asked to confirm this against the
    printed axis, which is far easier when the anchor sits on a tick and shows
    that tick's own number.

    Moving the anchors is only safe if the ticks lie on the calibration's own
    line: this function and ``match_calibration`` choose their label sets by
    different criteria (frame edge vs curve bbox) and *can* disagree, in which
    case re-anchoring would change the mapping rather than just relabel it. So
    each axis is re-anchored only after checking that the calibration already
    maps those tick positions to those tick values; otherwise that axis is left
    exactly as it was.
    """
    from dataclasses import replace

    def _outer(ticks):
        labelled = [t for t in ticks if t.get("value") is not None]
        if len(labelled) < 2 or labelled[0]["pos"] == labelled[-1]["pos"]:
            return None
        return labelled[0], labelled[-1]

    def _agrees(lo, hi, p1, v1, p2, v2) -> bool:
        """True if the map through (p1,v1)-(p2,v2) predicts both tick values."""
        if p2 == p1:
            return False
        slope = (v2 - v1) / (p2 - p1)
        span = abs(hi["value"] - lo["value"]) or 1.0
        for t in (lo, hi):
            predicted = v1 + (t["pos"] - p1) * slope
            if abs(predicted - t["value"]) > 0.02 * span:
                return False
        return True

    out = cal
    xs = _outer(x_ticks)
    if xs and _agrees(*xs, cal.x1, cal.ex1, cal.x2, cal.ex2):
        lo, hi = xs
        out = replace(out, x1=lo["pos"], ex1=lo["value"],
                      x2=hi["pos"], ex2=hi["value"])
    ys = _outer(y_ticks)
    if ys and _agrees(*ys, cal.y1, cal.jy1, cal.y2, cal.jy2):
        lo, hi = ys
        out = replace(out, y1=lo["pos"], jy1=lo["value"],
                      y2=hi["pos"], jy2=hi["value"])
    return out


def _stamp_tick_values(ticks: list[dict], p1: float, v1: float,
                       p2: float, v2: float) -> None:
    """Label every detected tick by interpolating the two OCR'd anchors.

    Turns bare tick positions into readable numbers in the UI, so the user can
    see at a glance whether the OCR'd scale makes sense across the whole axis
    instead of only at its two ends.
    """
    if p2 == p1:
        return
    slope = (v2 - v1) / (p2 - p1)
    for t in ticks:
        if t.get("value") is None:
            t["value"] = round(v1 + (t["pos"] - p1) * slope, 6)


def _crop_panel(img: np.ndarray, frame_pdf, zoom: float):
    """Crop the rendered page around a panel, keeping its axis furniture."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = frame_pdf
    cx0 = max(0, int((x0 - MARGIN_PT) * zoom))
    cy0 = max(0, int((y0 - MARGIN_PT * 0.55) * zoom))
    cx1 = min(w, int((x1 + MARGIN_PT * 0.45) * zoom))
    cy1 = min(h, int((y1 + MARGIN_PT) * zoom))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:      # degenerate frame: use the page
        cx0, cy0, cx1, cy1 = 0, 0, w, h
    return img[cy0:cy1, cx0:cx1], (cx0, cy0)


def _panel_curves(curves, panel_label: str, stem: str, figure: str,
                  legend_map: dict) -> list[CurveUnit]:
    """Build this panel's curves with unique, readable suggested filenames.

    Naming, in the order the descriptiveness runs out: the figure's own legend
    text for that colour, else a colour word. Two curves in one panel can be
    different shades that describe to the same word ("red" for #ff4040 and
    #cc0000), so any repeated tail gets its hex appended — filenames inside a
    panel must not collide, and the user can still rename either one.
    """
    built: list[tuple] = []
    for cg in curves:
        loop = dedupe(order_curve(keep_main_components(cg.polylines)))
        if len(loop) < 2:
            continue
        cname = _describe_color(cg.rgb)
        legend = (legend_map.get(cg.name) or "").strip()
        built.append((cg, loop, cname, legend, _slug(legend) or cname))

    seen: dict[str, int] = {}
    for _, _, _, _, tail in built:
        seen[tail] = seen.get(tail, 0) + 1

    # "fig1a" when the caption gave a figure number; a bare panel letter when it
    # did not, since "figa" reads like a typo rather than a figure reference.
    if figure and any(ch.isdigit() for ch in figure):
        fig_part = f"_fig{figure}"
    elif figure or panel_label:
        fig_part = f"_{figure or panel_label}"
    else:
        fig_part = ""
    out = []
    for cg, loop, cname, legend, tail in built:
        if seen[tail] > 1:
            r, g, b = (int(round(max(0.0, min(1.0, float(v))) * 255)) for v in cg.rgb)
            tail = f"{tail}-{r:02x}{g:02x}{b:02x}"
        out.append(CurveUnit(
            color=cg.name, color_label=cname, rgb=[round(float(v), 4) for v in cg.rgb],
            legend=legend, name=f"{stem}{fig_part}_{tail}", sample=legend,
            n_points=int(len(loop)),
            loopiness=round(float(loop_metrics(loop)["loopiness"]), 3),
            loop_pdf=[[round(float(a), 4), round(float(b), 4)] for a, b in loop],
        ))
    return out


#: A page needs at least this many vector path items to be worth opening as a
#: figure. Well below what a real plot draws (hundreds of sub-paths), but far
#: above a page of body text with a rule or two.
MIN_CURVE_ITEMS = 120

#: Smallest panel worth showing, in PDF points. The whole workflow asks a human
#: to read this panel's tick labels off a picture of it, so a plot too small to
#: carry legible labels is not a work unit no matter how CV-shaped its ink is.
#: Frame detection over-segments some pages into grids of tiny repeated drawings
#: (climent_2017 p1: twenty 40x51pt thumbnails), and those would otherwise
#: become twenty panels to click through. Real plots clear this comfortably —
#: luo_2022's four-panel figure is 155x126pt each.
MIN_PANEL_W_PT = 70.0
MIN_PANEL_H_PT = 55.0


def candidate_pages(pdf: str) -> list[int]:
    """Pages of ``pdf`` that plausibly hold a vector figure.

    ``vector_extract.find_figure_pages`` requires two or more *non-black*
    stroke colours, which is a good filter for multi-curve colour figures but
    silently drops the classic all-black voltammogram this corpus is full of.
    Colour count says nothing about whether something is a CV, so here the
    candidate test is colour-blind — just "enough vector geometry to be a
    figure" — and the loop-score gate downstream decides what is actually a CV.
    """
    from ..ingest import classify_pdf

    pages: set[int] = set()
    try:
        pages.update(find_figure_pages(pdf))
    except Exception:
        pass
    try:
        for info in classify_pdf(pdf):
            if info.n_curve_items >= MIN_CURVE_ITEMS:
                pages.add(info.number)
    except Exception:
        pass
    return sorted(pages)


def scan_pdf(pdf: str, work_dir: str, *, cv_threshold: float = 0.08,
             min_points: int = 60) -> list[PanelUnit]:
    """Every CV-like vector panel in one PDF, as calibration-ready work units."""
    stem = os.path.splitext(os.path.basename(pdf))[0]
    pages = candidate_pages(pdf)
    if not pages:
        return []

    paper = extract_paper_metadata(pdf).as_dict()
    units: list[PanelUnit] = []

    for page in pages:
        try:
            panels = detect_panels(pdf, page, min_points=min_points,
                                   min_curve_points=max(60, min_points))
        except Exception:
            continue
        if not panels:
            continue

        # Two gates, both before any rendering or file I/O. Size first because
        # it is free; then CV-likeness, whose loop score is calibration-
        # invariant, to drop schematics, Tafel plots and micrographs.
        keep = []
        for p in panels:
            x0, y0, x1, y1 = p.frame_pdf
            if (x1 - x0) < MIN_PANEL_W_PT or (y1 - y0) < MIN_PANEL_H_PT:
                continue
            score = max((loop_metrics(dedupe(order_curve(
                keep_main_components(c.polylines))))["loopiness"]
                for c in p.curves), default=0.0)
            if score >= cv_threshold:
                keep.append(p)
        if not keep:
            continue

        try:
            label_sets = find_axis_label_sets(pdf, page)
        except Exception:
            label_sets = []
        titles = ("", "")
        if label_sets:
            titles = (next((s.title for s in label_sets
                            if s.orientation == "x" and s.title), ""),
                      next((s.title for s in label_sets
                            if s.orientation == "y" and s.title), ""))
        try:
            fmeta = extract_figure_metadata(pdf, page, axis_titles=titles or None)
        except Exception:
            fmeta = None
        caption = fmeta.caption if fmeta else ""
        fig_no = _fig_number(caption)
        cap_legend = parse_curve_legend(caption) if fmeta else {}

        img = render_page(pdf, page, zoom=ZOOM)
        # Drop the panel letter only for a figure that really is a single plot.
        # If the page had several panels and the CV filter left one, that letter
        # still identifies which sub-plot of the published figure this came from.
        single = len(panels) == 1

        for p in keep:
            plabel = "" if single else p.label
            try:
                plot_legend = detect_plot_legend(pdf, page, p.curves)
            except Exception:
                plot_legend = {}
            legend_map = {**plot_legend,
                          **legend_for_panel(cap_legend, plabel or p.label or "")}

            crop, (ox, oy) = _crop_panel(img, p.frame_pdf, ZOOM)
            uid = f"{stem}_p{page}" + (f"_{plabel}" if plabel else "")
            rel_img = os.path.join("panels", uid + ".png")
            abs_img = os.path.join(work_dir, rel_img)
            os.makedirs(os.path.dirname(abs_img), exist_ok=True)
            _write_png(abs_img, crop)

            figure = f"{fig_no}{plabel}" if fig_no else plabel
            cunits = _panel_curves(p.curves, plabel, stem, figure, legend_map)
            if not cunits:
                continue

            guess, source, x_ticks, y_ticks = None, "none", [], []
            bbox = union_bbox(p.curves)

            # Pre-fill, best source first:
            #  1. real text tick labels -> position AND value are known.
            #  2. detected tick marks + OCR of the outer labels -> the path for
            #     figures whose text is outlined (most of this corpus), where
            #     the text layer holds nothing to fit.
            # Either way every detected tick becomes a click-to-snap target so
            # a wrong guess is a click plus a number, not a hunt for a pixel.
            if label_sets and bbox:
                x_ticks = _ticks_from_labels(label_sets, "x", p.frame_pdf)
                y_ticks = _ticks_from_labels(label_sets, "y", p.frame_pdf)
                cal = match_calibration(label_sets, bbox)
                if cal is not None:
                    guess = asdict(_snap_to_ticks(cal, x_ticks, y_ticks))
                    source = "auto-text"

            if bbox and (guess is None or not x_ticks or not y_ticks):
                det = _tick_detection(pdf, page, bbox, img)
                if det is not None:
                    z = det["zoom"]
                    if not x_ticks:
                        x_ticks = [{"pos": round(t / z, 3), "value": None}
                                   for t in det["ticks"]["x_ticks"]]
                    if not y_ticks:
                        y_ticks = [{"pos": round(t / z, 3), "value": None}
                                   for t in det["ticks"]["y_ticks"]]
                    if guess is None:
                        ocr_cal = _ocr_calibration(det)
                        if ocr_cal is not None:
                            guess, source = asdict(ocr_cal), "auto-ocr"
                            _stamp_tick_values(x_ticks, ocr_cal.x1, ocr_cal.ex1,
                                               ocr_cal.x2, ocr_cal.ex2)
                            _stamp_tick_values(y_ticks, ocr_cal.y1, ocr_cal.jy1,
                                               ocr_cal.y2, ocr_cal.jy2)

            units.append(PanelUnit(
                uid=uid, pdf=pdf, page=page, stem=stem, panel=plabel,
                figure=figure, image=rel_img.replace("\\", "/"), zoom=ZOOM,
                origin=[int(ox), int(oy)],
                size=[int(crop.shape[1]), int(crop.shape[0])],
                frame_pdf=[round(float(v), 3) for v in p.frame_pdf],
                curves=[asdict(c) for c in cunits],
                calib_guess=guess, calib_source=source,
                x_ticks=x_ticks, y_ticks=y_ticks,
                axis_titles=list(titles),
                axis_units=list(_axis_units(pdf, page, p.frame_pdf)),
                figure_meta=(fmeta.as_dict() if fmeta and not fmeta.is_empty else {}),
                paper_meta=paper,
            ))
    return units


def _write_png(path: str, rgb: np.ndarray) -> None:
    import cv2
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def index_path(work_dir: str) -> str:
    return os.path.join(work_dir, "index.json")


def geometry_path(work_dir: str, uid: str) -> str:
    return os.path.join(work_dir, "geometry", uid + ".json")


def _split_geometry(work_dir: str, unit: dict) -> dict:
    """Move a unit's point arrays into ``geometry/<uid>.json``; return the rest.

    Curve geometry dwarfs everything else — 243 curves of up to 3000 points each
    made a single index.json of 46 MB. The server re-reads the index on every
    request and the browser downloaded all of it up front, so the size was paid
    per keystroke rather than once. Keeping the index to the fields needed to
    *list* panels, and loading points only for the panel on screen, is the
    difference between a usable tool and an unusable one.
    """
    os.makedirs(os.path.join(work_dir, "geometry"), exist_ok=True)
    geometry = {c["color"]: c.pop("loop_pdf", []) for c in unit.get("curves", [])}
    with open(geometry_path(work_dir, unit["uid"]), "w", encoding="utf-8") as f:
        json.dump(geometry, f, separators=(",", ":"))
    return unit


def load_geometry(work_dir: str, uid: str) -> dict:
    """``{colour_key: [[x, y], ...]}`` in PDF points for one panel."""
    path = geometry_path(work_dir, uid)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def unit_with_geometry(work_dir: str, unit: dict) -> dict:
    """A copy of ``unit`` with each curve's ``loop_pdf`` filled back in."""
    geometry = load_geometry(work_dir, unit["uid"])
    out = {**unit, "curves": [dict(c) for c in unit.get("curves", [])]}
    for curve in out["curves"]:
        curve["loop_pdf"] = geometry.get(curve["color"], [])
    return out


def needs_migration(index: dict) -> bool:
    """True for an index written before geometry was split into its own files."""
    return any("loop_pdf" in c
               for u in index.get("units", []) for c in u.get("curves", []))


def migrate_index(work_dir: str) -> dict:
    """Split an old fat index in place, preserving status and edited names."""
    index = load_index(work_dir)
    if not needs_migration(index):
        return index
    for unit in index["units"]:
        _split_geometry(work_dir, unit)
    save_index(work_dir, index)
    return index


def scan_folder(folder: str, work_dir: str, *, cv_threshold: float = 0.08,
                progress=None) -> dict:
    """Scan every PDF in ``folder``; write ``work_dir/index.json``.

    ``progress`` is an optional ``callable(done, total, label)`` for CLI output.
    Returns the index dict. Re-scanning is safe: it rebuilds from the PDFs, but
    any per-unit ``status`` already recorded is carried over so a part-finished
    calibration session is not reset.
    """
    pdfs = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    os.makedirs(work_dir, exist_ok=True)

    previous: dict[str, dict] = {}
    if os.path.exists(index_path(work_dir)):
        try:
            with open(index_path(work_dir), encoding="utf-8") as f:
                for u in json.load(f).get("units", []):
                    previous[u["uid"]] = u
        except Exception:
            previous = {}

    units: list[dict] = []
    for i, pdf in enumerate(pdfs, 1):
        if progress:
            progress(i, len(pdfs), os.path.basename(pdf))
        for unit in scan_pdf(pdf, work_dir, cv_threshold=cv_threshold):
            d = _split_geometry(work_dir, asdict(unit))
            old = previous.get(d["uid"])
            if old:
                d["status"] = old.get("status", "pending")
                # keep the names/samples the user already edited
                for new_c, old_c in zip(d["curves"], old.get("curves", [])):
                    if old_c.get("color") == new_c.get("color"):
                        new_c["name"] = old_c.get("name", new_c["name"])
                        new_c["sample"] = old_c.get("sample", new_c["sample"])
                if old.get("saved_calibration"):
                    d["saved_calibration"] = old["saved_calibration"]
            units.append(d)

    index = {"source_folder": folder, "n_pdfs": len(pdfs), "units": units}
    with open(index_path(work_dir), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index


def load_index(work_dir: str) -> dict:
    with open(index_path(work_dir), encoding="utf-8") as f:
        return json.load(f)


def save_index(work_dir: str, index: dict) -> None:
    with open(index_path(work_dir), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
