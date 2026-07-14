"""Command-line interface for cvdigitize.

Quickest start — just point it at a PDF, no subcommand needed:
    cvdigitize paper.pdf

That auto-picks a page, extracts whatever curves it can, and writes a
normalised CSV + an HTML report you can open and look at immediately. Add
calibration once you know what you're looking at (see below).

Examples
--------
    # inspect a PDF: which pages hold vector CV figures / rasterized figures?
    cvdigitize info paper.pdf

    # extract curves from the auto-detected figure page (whole figure)
    cvdigitize extract paper.pdf
    cvdigitize paper.pdf                     # same thing, shorthand

    # [vector figure] multi-panel: split 2x2, process panel (a), with calibration
    cvdigitize extract paper.pdf --panels 2x2 --panel a \\
        --calibration configs/paper_fig1a.calib.json --scan-rate "50 mV/s"

    # [raster/scanned figure] panels + tick values are auto-detected; you just
    # supply the two outermost axis labels (note the =, needed for negative
    # numbers so the shell doesn't mistake "-0.8" for another flag):
    cvdigitize extract paper.pdf --raster-panel 0 \\
        --x-ticks="-0.8,0.2" --y-ticks="25,-50" --x-unit "V" --y-unit "uA"

    # render a page with a pixel grid to read off axis-anchor pixels
    # (only needed for vector-figure calibration; raster ticks are automatic)
    cvdigitize grid paper.pdf --page 1

If you installed via the venv directly instead of the `cvdigitize` launcher,
replace `cvdigitize` above with `.venv\\Scripts\\python.exe -m cvdigitize`
(Windows) or `.venv/bin/python -m cvdigitize` (macOS/Linux).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ingest import classify_pdf, render_page
from .vector_extract import (extract_color_groups, extract_panels,
                             find_figure_pages, panel_label)
from .postprocess import (order_curve, dedupe, resample_arclength,
                          keep_main_components, loop_metrics)
from .calibrate import Calibration, calibration_from_anchors
from .package import write_datapackage, write_csv, CurveMeta
from .raster_extract import (find_image_regions, extract_all_panel_curves,
                             crop_tick_labels)


# --------------------------------------------------------------------------- #
def _parse_grid(s: str) -> tuple[int, int]:
    s = s.lower().replace("×", "x")
    nx, ny = (int(v) for v in s.split("x"))
    return nx, ny


def _plot_color(rgb):
    return (max(0, min(1, rgb[0])), max(0, min(1, rgb[1])), max(0, min(1, rgb[2])))


def _parse_two_floats(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def _auto_page(pdf: str) -> tuple[int, str]:
    """Pick a page to work on and report its kind, when ``--page`` is omitted.

    Prefers a vector-curves page (higher fidelity, no calibration guesswork
    needed for the split step); falls back to the first raster page with a
    sizeable embedded image; else page 0. Among several vector candidates the
    one with the MOST line/curve items wins — a dense CV figure has orders of
    magnitude more than decorative cover art or a small inset, so ranking by
    content (not page order) avoids landing on a banner page.
    """
    infos = classify_pdf(pdf)
    figs = find_figure_pages(pdf)
    if figs:
        items = {pi.number: pi.n_curve_items for pi in infos}
        best = max(figs, key=lambda p: items.get(p, 0))
        return best, "vector-curves"
    for pi in infos:
        if pi.kind == "raster":
            return pi.number, "raster"
    return 0, (infos[0].kind if infos else "sparse")


def cmd_info(args):
    infos = classify_pdf(args.pdf)
    figs = find_figure_pages(args.pdf)
    raster = [pi.number for pi in infos if pi.kind == "raster"]
    print(f"{args.pdf}\n{'page':>4} {'kind':<15} {'draw':>6} {'items':>7} "
          f"{'colors':>7} {'imgs':>5} {'img%':>6} {'big%':>6}")
    for pi in infos:
        mark = " *" if pi.number in figs else (" R" if pi.number in raster else "")
        print(f"{pi.number:>4} {pi.kind:<15} {pi.n_drawings:>6} {pi.n_curve_items:>7} "
              f"{pi.n_stroke_colors:>7} {pi.n_images:>5} {pi.image_area_frac*100:>5.0f}% "
              f"{pi.largest_image_frac*100:>5.0f}%{mark}")
    print(f"\nVector figure candidates (*): {figs or 'none'}")
    print(f"Raster figure candidates (R): {raster or 'none'}")
    if figs:
        print(f"\nNext: cvdigitize extract \"{args.pdf}\"")
    elif raster:
        print(f"\nNext: cvdigitize extract \"{args.pdf}\" --page {raster[0]}   "
              f"(rasterized figure — panels & ticks auto-detected)")
    return 0


def cmd_grid(args):
    """Render a page with a pixel-coordinate grid to help read axis anchors."""
    img = render_page(args.pdf, args.page, zoom=args.zoom)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 150, h / 150))
    ax.imshow(img, extent=[0, w / args.zoom, h / args.zoom, 0])  # PDF-point coords
    step = args.step
    for x in np.arange(0, w / args.zoom, step):
        ax.axvline(x, color="red", lw=0.3, alpha=0.4)
    for y in np.arange(0, h / args.zoom, step):
        ax.axhline(y, color="red", lw=0.3, alpha=0.4)
    ax.set_title(f"{os.path.basename(args.pdf)} p{args.page} — grid step {step} PDF-pt\n"
                 "read axis-tick pixel positions here to build a calibration JSON")
    out = args.out or os.path.join("data", "out",
                                   os.path.splitext(os.path.basename(args.pdf))[0],
                                   f"grid_p{args.page}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {out}")
    return 0


def _process_curves(curves, calib, resample_n):
    """order -> (calibrate) -> resample each curve; return list of dicts."""
    out = []
    for cg in curves:
        loop = dedupe(order_curve(keep_main_components(cg.polylines)))
        if calib is not None:
            data = calib.apply(loop)
            xlab, ylab = calib.x_label, calib.y_label
            xunit, yunit = calib.x_unit, calib.y_unit
        else:
            # normalise to [0,1], y flipped so 'up' is positive
            x, y = loop[:, 0], loop[:, 1]
            xr, yr = x.max() - x.min() or 1, y.max() - y.min() or 1
            data = np.column_stack([(x - x.min()) / xr, 1 - (y - y.min()) / yr])
            xlab, ylab, xunit, yunit = "x_norm", "y_norm", "0..1", "0..1 (up=+)"
        if resample_n and len(data) > 2:
            data = resample_arclength(data, n=resample_n)
        out.append({"group": cg, "data": data, "xlab": xlab, "ylab": ylab,
                    "xunit": xunit, "yunit": yunit,
                    "loopiness": round(loop_metrics(data)["loopiness"], 3)})
    return out


def _write_html_report(out_dir, stem, report):
    """A self-contained HTML summary (embedded PNGs) for quick sharing."""
    import base64

    def embed(path):
        try:
            with open(path, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode()
        except OSError:
            return ""

    panels = sorted({c["panel"] for c in report["curves"]})
    blocks = []
    for pl in panels:
        pdir = os.path.join(out_dir, f"panel_{pl}") if pl else out_dir
        rows = "".join(
            f"<tr><td>{c['name']}</td><td>{c['color']}</td>"
            f"<td>{c['n_points']}</td><td>{c['units'][0]} / {c['units'][1]}</td></tr>"
            for c in report["curves"] if c["panel"] == pl
        )
        title = f"Panel {pl}" if pl else "Figure"
        blocks.append(f"""
        <section>
          <h2>{title}</h2>
          <div class="imgs">
            <figure><img src="{embed(os.path.join(pdir,'curves.png'))}"><figcaption>digitized output</figcaption></figure>
            <figure><img src="{embed(os.path.join(pdir,'overlay.png'))}"><figcaption>overlay on original</figcaption></figure>
          </div>
          <table><thead><tr><th>curve</th><th>colour</th><th>points</th><th>units</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </section>""")

    cal = "calibrated (real units)" if report["calibrated"] else "UNCALIBRATED (normalised)"
    html = f"""<!doctype html><meta charset="utf-8"><title>cvdigitize — {stem}</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem;max-width:1100px;color:#1a1a1a}}
h1{{margin-bottom:.2rem}} .sub{{color:#666;margin-top:0}}
section{{border-top:1px solid #ddd;padding-top:1rem;margin-top:1.5rem}}
.imgs{{display:flex;gap:1rem;flex-wrap:wrap}} figure{{margin:0}} img{{max-width:520px;border:1px solid #eee}}
figcaption{{color:#666;font-size:.85rem}} table{{border-collapse:collapse;margin-top:.8rem}}
td,th{{border:1px solid #ddd;padding:.3rem .6rem;font-size:.9rem;text-align:left}}
code{{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}}
</style>
<h1>cvdigitize — {stem}</h1>
<p class="sub">page {report['page']} · grid {report['panels_grid'][0]}×{report['panels_grid'][1]} ·
{len(report['curves'])} curves · {cal}</p>
{''.join(blocks)}
<p class="sub">Generated by cvdigitize. Data files (CSV/JSON/YAML) are alongside this report.</p>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def cmd_extract(args):
    pdf = args.pdf
    stem = os.path.splitext(os.path.basename(pdf))[0]
    if args.page is not None:
        page, kind = args.page, None
    else:
        page, kind = _auto_page(pdf)
        print(f"No --page given, auto-picked page {page} (detected as {kind}).")
    out_dir = args.out or os.path.join("data", "out", stem)
    os.makedirs(out_dir, exist_ok=True)

    if kind is None:  # explicit --page: figure out what's actually there
        infos = classify_pdf(pdf)
        kind = infos[page].kind if page < len(infos) else "sparse"
    if kind == "raster":
        return _extract_raster(args, pdf, page, stem, out_dir)
    return _extract_vector(args, pdf, page, stem, out_dir)


def _extract_raster(args, pdf, page, stem, out_dir):
    regions = find_image_regions(pdf, page)
    if not regions:
        print(f"Page {page} has no sizeable embedded image to trace. "
              f"Try a different --page (see `cvdigitize info`).")
        return 1

    region = regions[0]
    results = extract_all_panel_curves(pdf, page, region.bbox)
    results = [r for r in results if len(r["polyline"]) >= 20]
    if not results:
        print(f"No traceable curve found in the image on page {page}.")
        return 1

    rdir = out_dir
    os.makedirs(rdir, exist_ok=True)

    if args.raster_panel is None and len(results) > 1:
        # Discovery mode: show what was found, let the user pick + calibrate.
        fig, ax = plt.subplots(figsize=(11, 13))
        ax.imshow(results[0]["image"]); ax.axis("off")
        for i, r in enumerate(results):
            x0, y0, x1, y1 = r["frame_px"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="lime", linewidth=2))
            ax.text(x0 + 4, y0 + 20, f"[{i}]", color="lime", fontsize=16, weight="bold")
        ax.set_title(f"{stem} p{page} — {len(results)} plot panels auto-detected")
        fig.tight_layout()
        preview = os.path.join(rdir, f"raster_panels_p{page}.png")
        fig.savefig(preview, dpi=100); plt.close(fig)

        # calibration helper: zoomed crops of the outermost tick labels so the
        # user can read the two numbers per axis without hunting the figure.
        for i, r in enumerate(results):
            crops = crop_tick_labels(r["image"], r["frame_px"], r["ticks"])
            if not crops:
                continue
            # Label by position in the --x-ticks/--y-ticks pair (not geometry):
            # ticks come in detection order, so the crop at index 0 is the 1st
            # value the user types, index -1 the 2nd — unambiguous regardless of
            # which way the axis runs.
            order = [("x_lo", "--x-ticks 1st value"), ("x_hi", "--x-ticks 2nd value"),
                     ("y_lo", "--y-ticks 1st value"), ("y_hi", "--y-ticks 2nd value")]
            avail = [(k, lbl) for k, lbl in order if k in crops]
            fig, axs = plt.subplots(1, len(avail), figsize=(2.6 * len(avail), 2.2))
            axs = np.atleast_1d(axs)
            for ax, (k, lbl) in zip(axs, avail):
                ax.imshow(crops[k]); ax.set_title(lbl, fontsize=9); ax.axis("off")
            fig.suptitle(f"panel [{i}] — read these into --x-ticks/--y-ticks", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(rdir, f"calib_helper_panel{i}.png"), dpi=120)
            plt.close(fig)

        print(f"Found {len(results)} plot panels on page {page} (composite raster figure).")
        print(f"Preview with panel numbers: {preview}\n")
        for i, r in enumerate(results):
            xt, yt = r["ticks"]["x_ticks"], r["ticks"]["y_ticks"]
            print(f"  [{i}] {len(r['polyline'])} curve points, "
                  f"{len(xt)} x-ticks / {len(yt)} y-ticks auto-detected"
                  + (f"  (see calib_helper_panel{i}.png for the tick numbers)"
                     if (xt and yt) else ""))
        print(f'\nPick one and (optionally) calibrate, e.g.:\n'
              f'  python -m cvdigitize extract "{pdf}" --page {page} --raster-panel 0 '
              f'--x-ticks -0.8,0.2 --y-ticks 25,-50 --x-unit "V" --y-unit "uA"\n'
              f"(tick values = the two outermost axis labels printed on that panel; "
              f"omit them to get a normalised curve instead)")
        return 0

    idx = args.raster_panel or 0
    if idx >= len(results):
        print(f"--raster-panel {idx} out of range (found {len(results)} panel(s), 0-{len(results)-1}).")
        return 1
    r = results[idx]
    xt, yt = r["ticks"]["x_ticks"], r["ticks"]["y_ticks"]

    calib = None
    if args.x_ticks or args.y_ticks:
        if not (args.x_ticks and args.y_ticks):
            print("Pass both --x-ticks and --y-ticks to calibrate (or neither for a normalised curve).")
            return 1
        if len(xt) < 2 or len(yt) < 2:
            print(f"Only {len(xt)} x-tick(s)/{len(yt)} y-tick(s) auto-detected - need at least 2 "
                  f"per axis to calibrate. Try a cleaner/higher-res source, or a different panel.")
            return 1
        x_lo, x_hi = _parse_two_floats(args.x_ticks)
        y_lo, y_hi = _parse_two_floats(args.y_ticks)
        calib = calibration_from_anchors(
            x_anchor1=(xt[0], x_lo), x_anchor2=(xt[-1], x_hi),
            y_anchor1=(yt[0], y_lo), y_anchor2=(yt[-1], y_hi),
            x_unit=args.x_unit or "", y_unit=args.y_unit or "",
        )

    # Ticks are detected in the same pixel space as `image`/`polyline_px` (not
    # the PDF-point `polyline`) — calibrate and overlay against that directly.
    poly = r["polyline_px"]
    if calib is not None:
        data = calib.apply(poly)
        xlab, ylab, xunit, yunit = calib.x_label, calib.y_label, calib.x_unit, calib.y_unit
    else:
        x, y = poly[:, 0], poly[:, 1]
        xr, yr = (x.max() - x.min()) or 1, (y.max() - y.min()) or 1
        data = np.column_stack([(x - x.min()) / xr, 1 - (y - y.min()) / yr])
        xlab, ylab, xunit, yunit = "x_norm", "y_norm", "0..1", "0..1 (up=+)"
    if args.resample and len(data) > 2:
        data = resample_arclength(data, n=args.resample)

    plabel = f"r{idx}"
    pdir = os.path.join(rdir, f"panel_{plabel}")
    os.makedirs(pdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 13)); ax.imshow(r["image"]); ax.axis("off")
    x0, y0, x1, y1 = r["frame_px"]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="lime", linewidth=2))
    ax.plot(poly[:, 0], poly[:, 1], color="red", lw=1.2, label="extracted")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{stem} p{page} panel {plabel} (raster)")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "overlay.png"), dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(data[:, 0], data[:, 1], color="black", lw=1.0)
    ax.set_xlabel(f"{xlab} / {xunit}"); ax.set_ylabel(f"{ylab} / {yunit}")
    ax.set_title(("Digitized curve" if calib else "Digitized curve (normalised)")
                + f" — panel {plabel}")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "curves.png"), dpi=110); plt.close(fig)

    name = f"{stem}_{plabel}"
    meta = CurveMeta(name=name, figure=(args.figure or f"panel {plabel}"), curve=plabel,
                     scan_rate=args.scan_rate or "", x_label=xlab, x_unit=xunit,
                     y_label=ylab, y_unit=yunit, source_pdf=pdf, method="digitized",
                     comment="auto-extracted from a rasterized/scanned figure (colour-mask trace)"
                             + ("" if calib else "; UNCALIBRATED (normalised coords)"))
    if calib is not None:
        paths = write_datapackage(pdir, data, meta, yaml=not args.no_yaml)
    else:
        p = os.path.join(pdir, name + ".csv")
        write_csv(p, data, meta)
        paths = {"csv": p}

    report = {"pdf": pdf, "page": page, "panels_grid": [1, 1], "calibrated": calib is not None,
             "curves": [{"panel": plabel, "name": name, "color": plabel, "rgb": [0, 0, 0],
                        "n_points": len(data), "units": [xunit, yunit],
                        "loopiness": round(loop_metrics(data)["loopiness"], 3),
                        "files": {k: os.path.relpath(v, rdir) for k, v in paths.items()}}]}
    with open(os.path.join(rdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_html_report(rdir, stem, report)

    print(f"Extracted 1 curve from {stem} p{page} panel {plabel} "
          f"({'calibrated' if calib else 'UNCALIBRATED - pass --x-ticks/--y-ticks for real units'}).")
    print(f"Output: {rdir}")
    return 0


def _extract_vector(args, pdf, page, stem, out_dir):
    from .vector_extract import union_bbox
    from .autocalib import (find_axis_label_sets, match_calibration,
                            detect_ticks_for_bbox, assisted_tick_calibration)

    manual = Calibration.load(args.calibration) if args.calibration else None
    nx, ny = _parse_grid(args.panels)

    # Text-layer auto-calibration: many vector figures keep their axis tick
    # labels as real text; a linear fit through (label position, label value)
    # gives the full calibration with zero user input. Figures with outlined
    # text (no words) simply yield no label sets and fall through.
    label_sets = []
    if manual is None and not getattr(args, "no_autocalib", False):
        try:
            label_sets = find_axis_label_sets(pdf, page)
        except Exception:
            label_sets = []

    # collect curves grouped per panel (1x1 = whole figure)
    if (nx, ny) == (1, 1):
        panels = {(0, 0): extract_color_groups(pdf, page, min_points=args.min_points)}
    else:
        panels = extract_panels(pdf, page, nx, ny, min_curve_points=args.min_points)

    wanted = None
    if args.panel:
        col = "abcdefgh".index(args.panel.lower()) % nx
        row = "abcdefgh".index(args.panel.lower()) // nx
        wanted = (col, row)

    img = render_page(pdf, page, zoom=3.0)
    report = {"pdf": pdf, "page": page, "panels_grid": [nx, ny],
              "calibrated": manual is not None, "curves": []}
    total = 0
    modes_seen = []

    for key in sorted(panels):
        if wanted is not None and key != wanted:
            continue
        curves = panels[key]
        plabel = panel_label(key[0], key[1], nx) if (nx, ny) != (1, 1) else ""
        pdir = os.path.join(out_dir, f"panel_{plabel}") if plabel else out_dir
        os.makedirs(pdir, exist_ok=True)

        # ---- resolve this panel's calibration: manual > auto-text > assisted
        bbox = union_bbox(curves) if curves else None
        calib, mode, detection = manual, ("manual" if manual else "none"), None
        if calib is None and label_sets and bbox:
            calib = match_calibration(label_sets, bbox)
            if calib is not None:
                mode = "auto-text"
                print(f"Panel {plabel or '(whole figure)'}: auto-calibrated from axis "
                      f"text labels — x: [{calib.ex1:.3g}, {calib.ex2:.3g}] "
                      f"\"{calib.x_unit}\", y: [{calib.jy1:.3g}, {calib.jy2:.3g}] "
                      f"\"{calib.y_unit}\" (verify units in the YAML)")
        if calib is None and args.x_ticks and args.y_ticks and bbox:
            detection = detect_ticks_for_bbox(pdf, page, bbox)
            if detection is not None:
                calib = assisted_tick_calibration(
                    detection, _parse_two_floats(args.x_ticks),
                    _parse_two_floats(args.y_ticks),
                    x_unit=args.x_unit or "", y_unit=args.y_unit or "")
                mode = "assisted-ticks"
                print(f"Panel {plabel or '(whole figure)'}: calibrated from detected "
                      f"tick marks + your --x-ticks/--y-ticks values.")
            else:
                print(f"Panel {plabel or '(whole figure)'}: could not detect an axes "
                      f"frame with ticks — falling back to normalised output.")
        if calib is None and bbox is not None:
            # emit tick-label crops so the user can rerun with --x-ticks/--y-ticks
            detection = detection or detect_ticks_for_bbox(pdf, page, bbox)
            if detection is not None:
                from .raster_extract import crop_tick_labels
                crops = crop_tick_labels(detection["image"], detection["frame_px"],
                                         detection["ticks"], zoom=detection["zoom"])
                if crops:
                    order = [("x_lo", "--x-ticks 1st value"), ("x_hi", "--x-ticks 2nd value"),
                             ("y_lo", "--y-ticks 1st value"), ("y_hi", "--y-ticks 2nd value")]
                    avail = [(k, lbl) for k, lbl in order if k in crops]
                    fig, axs = plt.subplots(1, len(avail), figsize=(2.6 * len(avail), 2.2))
                    axs = np.atleast_1d(axs)
                    for ax, (k, lbl) in zip(axs, avail):
                        ax.imshow(crops[k]); ax.set_title(lbl, fontsize=9); ax.axis("off")
                    fig.suptitle(f"panel {plabel or 'main'} — read these into "
                                 f"--x-ticks/--y-ticks", fontsize=10)
                    fig.tight_layout()
                    fig.savefig(os.path.join(pdir, "calib_helper.png"), dpi=120)
                    plt.close(fig)
        modes_seen.append(mode)

        processed = _process_curves(curves, calib, args.resample)

        # overlay this panel's curves on the original render
        fig, ax = plt.subplots(figsize=(11, 13)); ax.imshow(img); ax.axis("off")
        for pc in processed:
            cg = pc["group"]
            for i, pl in enumerate(cg.polylines):
                ax.plot(pl[:, 0] * 3, pl[:, 1] * 3, color=_plot_color(cg.rgb),
                        lw=1.0, label=cg.name if i == 0 else None)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{stem} p{page}" + (f" panel {plabel}" if plabel else ""))
        fig.tight_layout(); fig.savefig(os.path.join(pdir, "overlay.png"), dpi=100)
        plt.close(fig)

        # clean plot of the digitized output itself (the deliverable curves)
        fig, ax = plt.subplots(figsize=(7, 5))
        for pc in processed:
            cg = pc["group"]
            ax.plot(pc["data"][:, 0], pc["data"][:, 1],
                    color=_plot_color(cg.rgb), lw=1.0, label=cg.name)
        ax.set_xlabel(f"{processed[0]['xlab']} / {processed[0]['xunit']}" if processed else "x")
        ax.set_ylabel(f"{processed[0]['ylab']} / {processed[0]['yunit']}" if processed else "y")
        ax.set_title(("Digitized curves" if calib else "Digitized curves (normalised)")
                     + (f" — panel {plabel}" if plabel else ""))
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(os.path.join(pdir, "curves.png"), dpi=110)
        plt.close(fig)

        cal_note = {
            "manual": "",
            "auto-text": "; calibration auto-detected from axis text labels (verify units)",
            "assisted-ticks": "; calibrated from detected tick marks + user-supplied values",
            "none": "; UNCALIBRATED (normalised coords)",
        }[mode]
        for pc in processed:
            cg = pc["group"]
            name = f"{stem}_" + (f"{plabel}_" if plabel else "") + cg.name
            meta = CurveMeta(
                name=name, figure=(args.figure or (f"panel {plabel}" if plabel else "")),
                curve=cg.name, scan_rate=args.scan_rate or "",
                x_label=pc["xlab"], x_unit=pc["xunit"],
                y_label=pc["ylab"], y_unit=pc["yunit"],
                source_pdf=pdf, method="digitized",
                comment="auto-extracted from vector PDF" + cal_note,
            )
            if calib is not None:
                paths = write_datapackage(pdir, pc["data"], meta, yaml=not args.no_yaml)
            else:
                # only CSV when uncalibrated
                p = os.path.join(pdir, name + ".csv")
                write_csv(p, pc["data"], meta)
                paths = {"csv": p}
            report["curves"].append({
                "panel": plabel, "name": name, "color": cg.name, "rgb": list(cg.rgb),
                "n_points": len(pc["data"]), "units": [pc["xunit"], pc["yunit"]],
                "loopiness": pc["loopiness"], "calibration": mode,
                "files": {k: os.path.relpath(v, out_dir) for k, v in paths.items()},
            })
            total += 1

    calibrated = any(m != "none" for m in modes_seen)
    report["calibrated"] = calibrated
    report["calibration_modes"] = modes_seen
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_html_report(out_dir, stem, report)

    mode_desc = ", ".join(sorted(set(modes_seen))) or "none"
    print(f"Extracted {total} curve(s) from {stem} p{page} "
          f"(calibration: {mode_desc}).")
    print(f"Output: {out_dir}")
    if not calibrated:
        print("For real units: check calib_helper.png (if present) and rerun with "
              "--x-ticks=LO,HI --y-ticks=LO,HI, or pass a --calibration JSON.")
    return 0


def cmd_batch(args):
    """Run over every PDF in a folder and build a gallery index.html.

    A survey tool: for each PDF it classifies the pages, auto-picks a figure,
    attempts an uncalibrated extraction, and records how many curves came out.
    Great for testing the tool across many papers at once and seeing at a
    glance which figures are vector vs raster and what was recovered.
    """
    import base64
    import glob

    pdfs = sorted(glob.glob(os.path.join(args.dir, "*.pdf")))
    if not pdfs:
        print(f"No PDFs found in {args.dir}")
        return 1
    out_dir = args.out or os.path.join("data", "out", "_batch")
    os.makedirs(out_dir, exist_ok=True)

    def embed(path):
        try:
            with open(path, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode()
        except OSError:
            return ""

    rows = []
    summary = []
    for pdf in pdfs:
        stem = os.path.splitext(os.path.basename(pdf))[0]
        try:
            infos = classify_pdf(pdf)
            vec = find_figure_pages(pdf)
            ras = [pi.number for pi in infos if pi.kind == "raster"]
            page, kind = _auto_page(pdf)
            paper_out = os.path.join(out_dir, stem)
            n_curves, overlay, note, best_loop = 0, "", "", 0.0
            try:
                a = argparse.Namespace(
                    pdf=pdf, page=page, panels="1x1", panel=None, calibration=None,
                    resample=800, min_points=60, figure=None, scan_rate=None,
                    no_yaml=True, no_autocalib=False, out=paper_out,
                    raster_panel=(0 if kind == "raster" else None),
                    x_ticks=None, y_ticks=None, x_unit=None, y_unit=None)
                _run_extract_quiet(a)
                rep_path = os.path.join(paper_out, "report.json")
                if os.path.exists(rep_path):
                    with open(rep_path) as f:
                        rep = json.load(f)
                    n_curves = len(rep["curves"])
                    loops = [c.get("loopiness", 0.0) for c in rep["curves"]]
                    best_loop = max(loops) if loops else 0.0
                for cand in glob.glob(os.path.join(paper_out, "**", "overlay.png"), recursive=True):
                    overlay = cand
                    break
            except Exception as e:  # keep the batch going
                note = f"extract error: {type(e).__name__}"
            # Loopiness gauges CV-likeness: closed loops enclosing area score high;
            # schematic lines / Nyquist arcs / axis fragments score low.
            conf = "likely CV" if best_loop >= 0.15 else ("maybe" if best_loop >= 0.05 else "unlikely CV")
            conf_color = {"likely CV": "#137333", "maybe": "#b26a00", "unlikely CV": "#a50e0e"}[conf]
            summary.append((stem, kind, len(vec), len(ras), n_curves, best_loop, conf, note))
            rows.append((best_loop, f"""
            <div class="card">
              <h3>{stem}</h3>
              <p class="meta">{len(infos)} pages · vector figs: {len(vec)} · raster figs: {len(ras)}
                 · page {page} ({kind}) · <b>{n_curves}</b> candidate curves
                 · <b style="color:{conf_color}">{conf}</b> (loop score {best_loop:.2f}){(' · '+note) if note else ''}</p>
              {'<img src="'+embed(overlay)+'">' if overlay else '<p class="none">no overlay</p>'}
            </div>"""))
            print(f"  {stem:48s} {kind:14s} vec={len(vec)} ras={len(ras)} "
                  f"curves={n_curves} loop={best_loop:.2f} [{conf}] {note}")
        except Exception as e:
            summary.append((stem, "ERROR", 0, 0, 0, str(e)))
            print(f"  {stem:52s} ERROR {e}")

    total_curves = sum(s[4] for s in summary)
    html = f"""<!doctype html><meta charset="utf-8"><title>cvdigitize batch — {os.path.basename(args.dir)}</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem;color:#1a1a1a;background:#fafafa}}
h1{{margin-bottom:.2rem}} .sub{{color:#666}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1rem;margin-top:1.5rem}}
.card{{background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:1rem;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.card h3{{margin:.1rem 0;font-size:.95rem;word-break:break-all}} .meta{{color:#666;font-size:.8rem}}
img{{max-width:100%;border:1px solid #eee;margin-top:.5rem}} .none{{color:#bbb;font-style:italic}}
</style>
<h1>cvdigitize — batch survey</h1>
<p class="sub">{len(pdfs)} PDFs from <code>{args.dir}</code> · {total_curves} candidate curves ·
sorted by CV-likeness (loop score). "candidate curves" are unverified — the tool
extracts from whatever figure it auto-picks; a low loop score usually means the
page is a schematic/Nyquist/other plot, not a CV.</p>
<div class="grid">{''.join(r for _, r in sorted(rows, key=lambda t: t[0], reverse=True))}</div>"""
    idx = os.path.join(out_dir, "index.html")
    with open(idx, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nProcessed {len(pdfs)} PDFs · {total_curves} curves total.")
    print(f"Gallery: {idx}")
    return 0


def _run_extract_quiet(args):
    """Invoke the right extract path, swallowing its stdout (used by batch)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        pdf, stem = args.pdf, os.path.splitext(os.path.basename(args.pdf))[0]
        os.makedirs(args.out, exist_ok=True)
        infos = classify_pdf(pdf)
        kind = infos[args.page].kind if args.page < len(infos) else "sparse"
        if kind == "raster":
            _extract_raster(args, pdf, args.page, stem, args.out)
        else:
            _extract_vector(args, pdf, args.page, stem, args.out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cvdigitize",
        description="Digitize Cyclic Voltammetry curves out of PDFs - vector or scanned.",
        epilog='Quick start: cvdigitize mypaper.pdf   (no subcommand needed)',
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="classify pages; list figure candidates")
    pi.add_argument("pdf")
    pi.set_defaults(func=cmd_info)

    pe = sub.add_parser("extract", help="extract & digitize curves")
    pe.add_argument("pdf")
    pe.add_argument("--page", type=int, default=None, help="0-indexed page (auto if omitted)")
    pe.add_argument("--panels", default="1x1", help="panel grid, e.g. 2x2 (default 1x1)")
    pe.add_argument("--panel", default=None, help="single panel letter a,b,c,... (default all)")
    pe.add_argument("--calibration", default=None, help="calibration JSON for real units")
    pe.add_argument("--resample", type=int, default=1000, help="arc-length points (0=off)")
    pe.add_argument("--min-points", type=int, default=60, help="min points per curve")
    pe.add_argument("--figure", default=None, help="figure label for metadata")
    pe.add_argument("--scan-rate", default=None, help="e.g. '50 mV/s'")
    pe.add_argument("--no-yaml", action="store_true", help="skip YAML metadata")
    pe.add_argument("--no-autocalib", action="store_true",
                    help="disable automatic calibration from the PDF text layer")
    pe.add_argument("--out", default=None, help="output dir (default data/out/<pdf>)")
    pe.add_argument("--raster-panel", type=int, default=None,
                    help="[raster pages] which auto-detected plot panel to use (0-indexed)")
    pe.add_argument("--x-ticks", default=None, metavar="LO,HI",
                    help="[raster pages] data values of the first/last auto-detected x-tick, "
                         "e.g. -0.8,0.2 (both --x-ticks and --y-ticks needed to calibrate)")
    pe.add_argument("--y-ticks", default=None, metavar="LO,HI",
                    help="[raster pages] data values of the first/last auto-detected y-tick")
    pe.add_argument("--x-unit", default=None, help="[raster pages] x-axis unit, e.g. 'V'")
    pe.add_argument("--y-unit", default=None, help="[raster pages] y-axis unit, e.g. 'uA'")
    pe.set_defaults(func=cmd_extract)

    pg = sub.add_parser("grid", help="render a page with a pixel grid for calibration")
    pg.add_argument("pdf")
    pg.add_argument("--page", type=int, default=0)
    pg.add_argument("--step", type=float, default=20.0, help="grid step in PDF points")
    pg.add_argument("--zoom", type=float, default=3.0)
    pg.add_argument("--out", default=None)
    pg.set_defaults(func=cmd_grid)

    pb = sub.add_parser("batch", help="run over a folder of PDFs; build a gallery")
    pb.add_argument("dir", help="folder containing PDFs")
    pb.add_argument("--out", default=None, help="output dir (default data/out/_batch)")
    pb.set_defaults(func=cmd_batch)
    return p


_KNOWN_COMMANDS = {"info", "extract", "grid", "batch", "-h", "--help"}


def _with_implicit_extract(argv: list[str]) -> list[str]:
    """Let ``cvdigitize mypaper.pdf`` work without typing ``extract`` first.

    If the first argument isn't a known subcommand, treat it (and everything
    after it) as arguments to ``extract`` — this is the single biggest
    friction point for a first-time user, so remove it.
    """
    if argv and argv[0] not in _KNOWN_COMMANDS:
        return ["extract", *argv]
    return argv


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(_with_implicit_extract(argv))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
