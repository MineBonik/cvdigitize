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
                          resample_uniform_potential, keep_main_components,
                          loop_metrics)
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


def _page_cv_score(pdf: str, page: int, kind: str) -> float:
    """Best CV-likeness (loop score) among a page's curves — no file I/O.

    Used to pick which figure page a paper's batch run should target, instead
    of blindly taking the first candidate.
    """
    try:
        if kind == "raster":
            from .raster_extract import find_image_regions, extract_all_panel_curves
            regions = find_image_regions(pdf, page)
            if not regions:
                return 0.0
            best = 0.0
            for r in extract_all_panel_curves(pdf, page, regions[0].bbox):
                for c in r.get("curves", []) or [{"polyline_px": r["polyline_px"]}]:
                    p = c["polyline_px"]
                    if len(p) >= 20:
                        best = max(best, loop_metrics(p)["loopiness"])
            return best
        from .vector_extract import detect_panels
        best = 0.0
        for panel in detect_panels(pdf, page):
            for c in panel.curves:
                loop = dedupe(order_curve(keep_main_components(c.polylines)))
                best = max(best, loop_metrics(loop)["loopiness"])
        return best
    except Exception:
        return 0.0


def _best_figure_page(pdf: str, vec: list[int], ras: list[int],
                      max_candidates: int = 5) -> tuple[int, str]:
    """Pick the most CV-like figure page across vector + raster candidates."""
    cands = [(p, "vector-curves") for p in vec] + [(p, "raster") for p in ras]
    if not cands:
        return _auto_page(pdf)
    cands = cands[:max_candidates]
    scored = [(_page_cv_score(pdf, p, k), p, k) for p, k in cands]
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_page, best_kind = scored[0]
    if best_score <= 0.0:  # nothing loop-like; fall back to the default heuristic
        return _auto_page(pdf)
    return best_page, best_kind


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


def _resample(data, n, mode):
    """Resample a digitized loop by arc length (smooth) or uniform potential
    (potentiostat-like, per scan branch)."""
    if not n or len(data) <= 2:
        return data
    if mode == "uniform-E":
        return resample_uniform_potential(data, n_per_branch=max(2, n // 2))
    return resample_arclength(data, n=n)


def _process_curves(curves, calib, resample_n, resample_mode="arclength"):
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
        data = _resample(data, resample_n, resample_mode)
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
            f"<tr><td>{c['color']}</td><td>{c.get('sample','') or '—'}</td>"
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
          <table><thead><tr><th>colour</th><th>sample (from caption)</th><th>points</th><th>units</th></tr></thead>
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


def _print_fidelity_triage(curves, pdir):
    """Print the reference-free ink-fidelity summary + which curves to eyeball.

    Fidelity is measured with no ground truth (how well each trace sits on its own
    ink), so it works on any paper. Amber/red curves are the ones to open in
    check.html — red dashes there mark chords across empty space to re-trace."""
    scored = [c for c in curves if c.get("fidelity") is not None]
    if not scored:
        return
    worst = min(c["fidelity"] for c in scored)
    flagged = sorted((c for c in scored if c.get("fidelity_grade") in ("fair", "poor")),
                     key=lambda c: c["fidelity"])
    print(f"Ink-fidelity: worst {worst:.0f}/100 across {len(scored)} curve(s).", end="")
    if flagged:
        names = ", ".join(f"{c['color']}({c['fidelity']:.0f})" for c in flagged)
        print(f" Eyeball {names} in check.html — re-trace in trace_assist if a red-dashed"
              f" chord is off.")
    else:
        print(" Every trace sits on the ink.")


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
    results = [r for r in results if len(r["polyline"]) >= 20 or r.get("curves")]
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
            cnames = ", ".join(c["name"] for c in r.get("curves", [])) or "none"
            print(f"  [{i}] curves: {cnames} · "
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
    calib_mode = "none"
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
        calib_mode = "manual-ticks"
    elif len(xt) >= 2 and len(yt) >= 2:
        # No values typed: if a Tesseract OCR engine is installed, read the
        # four outer tick labels automatically (zero-typing calibration).
        from . import ocr
        if ocr.available():
            crops = crop_tick_labels(r["image"], r["frame_px"], r["ticks"])
            xp, yp = ocr.read_axis_values(crops)
            if xp and yp:
                calib = calibration_from_anchors(
                    x_anchor1=(xt[0], xp[0]), x_anchor2=(xt[-1], xp[1]),
                    y_anchor1=(yt[0], yp[0]), y_anchor2=(yt[-1], yp[1]),
                    x_unit=args.x_unit or "", y_unit=args.y_unit or "")
                calib_mode = "ocr"
                print(f"OCR read axis labels (VERIFY): x=[{xp[0]:g}, {xp[1]:g}], "
                      f"y=[{yp[0]:g}, {yp[1]:g}]. Pass --x-ticks/--y-ticks to override.")

    # Ticks are detected in the same pixel space as `image`/`polyline_px` (not
    # the PDF-point `polyline`) — calibrate and overlay against that directly.
    # One panel can hold several colour-coded curves (split by hue) plus a
    # dark one; fall back to the primary dark trace when splitting found none.
    curve_list = r.get("curves") or [
        {"name": "curve", "rgb": (0.1, 0.1, 0.1), "polyline_px": r["polyline_px"]}]

    plabel = f"r{idx}"
    pdir = os.path.join(rdir, f"panel_{plabel}")
    os.makedirs(pdir, exist_ok=True)

    report = {"pdf": pdf, "page": page, "panels_grid": [1, 1],
              "calibrated": calib is not None, "curves": []}
    processed = []
    for c in curve_list:
        poly = c["polyline_px"]
        if calib is not None:
            data = calib.apply(poly)
            xlab, ylab, xunit, yunit = calib.x_label, calib.y_label, calib.x_unit, calib.y_unit
        else:
            x, y = poly[:, 0], poly[:, 1]
            xr, yr = (x.max() - x.min()) or 1, (y.max() - y.min()) or 1
            data = np.column_stack([(x - x.min()) / xr, 1 - (y - y.min()) / yr])
            xlab, ylab, xunit, yunit = "x_norm", "y_norm", "0..1", "0..1 (up=+)"
        data = _resample(data, args.resample, getattr(args, "resample_mode", "arclength"))
        processed.append((c, data, xlab, ylab, xunit, yunit))

    fig, ax = plt.subplots(figsize=(11, 13)); ax.imshow(r["image"]); ax.axis("off")
    x0, y0, x1, y1 = r["frame_px"]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="lime", linewidth=2))
    for c, *_ in processed:
        ax.plot(c["polyline_px"][:, 0], c["polyline_px"][:, 1],
                color=_plot_color(c["rgb"]), lw=1.2, label=c["name"])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{stem} p{page} panel {plabel} (raster)")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "overlay.png"), dpi=100); plt.close(fig)

    from .qc import write_qc
    qc = write_qc(pdir, r["image"],
                  [{"name": c["name"], "xy": c["polyline_px"], "rgb": c["rgb"]}
                   for c, *_ in processed],
                  panel_stem="panel", title=f"{stem} p{page} panel {plabel}")
    fids = qc.get("fidelity", [])

    fig, ax = plt.subplots(figsize=(7, 5))
    for c, data, xlab, ylab, xunit, yunit in processed:
        ax.plot(data[:, 0], data[:, 1], color=_plot_color(c["rgb"]), lw=1.0,
                label=c["name"])
    ax.set_xlabel(f"{xlab} / {xunit}"); ax.set_ylabel(f"{ylab} / {yunit}")
    ax.set_title(("Digitized curves" if calib else "Digitized curves (normalised)")
                + f" — panel {plabel}")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "curves.png"), dpi=110); plt.close(fig)

    for k, (c, data, xlab, ylab, xunit, yunit) in enumerate(processed):
        name = f"{stem}_{plabel}_{c['name']}" if len(processed) > 1 else f"{stem}_{plabel}"
        meta = CurveMeta(name=name, figure=(args.figure or f"panel {plabel}"),
                         curve=c["name"], scan_rate=args.scan_rate or "",
                         x_label=xlab, x_unit=xunit, y_label=ylab, y_unit=yunit,
                         source_pdf=pdf, method="digitized",
                         comment="auto-extracted from a rasterized/scanned figure (colour-mask trace)"
                                 + ("" if calib else "; UNCALIBRATED (normalised coords)"))
        if calib is not None:
            paths = write_datapackage(pdir, data, meta, yaml=not args.no_yaml)
        else:
            p = os.path.join(pdir, name + ".csv")
            write_csv(p, data, meta)
            paths = {"csv": p}
        fid = fids[k] if k < len(fids) else {}
        report["curves"].append({
            "panel": plabel, "name": name, "color": c["name"], "rgb": list(c["rgb"]),
            "n_points": len(data), "units": [xunit, yunit],
            "loopiness": round(loop_metrics(data)["loopiness"], 3),
            "fidelity": fid.get("score"), "fidelity_grade": fid.get("grade"),
            "calibration": calib_mode,
            "files": {k2: os.path.relpath(v, rdir) for k2, v in paths.items()}})

    with open(os.path.join(rdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_html_report(rdir, stem, report)
    _print_fidelity_triage(report["curves"], pdir)

    print(f"Extracted {len(processed)} curve(s) from {stem} p{page} panel {plabel} "
          f"({'calibrated' if calib else 'UNCALIBRATED - pass --x-ticks/--y-ticks for real units'}).")
    print(f"Output: {rdir}")
    return 0


def _extract_vector(args, pdf, page, stem, out_dir):
    from .vector_extract import union_bbox, detect_panels
    from .autocalib import (find_axis_label_sets, match_calibration,
                            detect_ticks_for_bbox, assisted_tick_calibration)

    manual = Calibration.load(args.calibration) if args.calibration else None

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

    # Best-effort experimental metadata from the caption / page text (scan
    # rate, electrolyte, reference electrode ...). Pre-fills the datapackage so
    # a curator confirms rather than types — the thread's "longest part".
    from .metadata import extract_figure_metadata
    titles = None
    if label_sets:
        xs = next((s.title for s in label_sets if s.orientation == "x" and s.title), "")
        ys = next((s.title for s in label_sets if s.orientation == "y" and s.title), "")
        titles = (xs, ys)
    try:
        fig_meta = extract_figure_metadata(pdf, page, axis_titles=titles)
    except Exception:
        fig_meta = None
    from .metadata import parse_curve_legend, legend_for_panel, detect_plot_legend
    curve_legend = parse_curve_legend(fig_meta.caption) if fig_meta else {}
    if fig_meta and not fig_meta.is_empty:
        bits = []
        if fig_meta.scan_rate:
            bits.append(f"scan rate {fig_meta.scan_rate}")
        if fig_meta.electrolytes:
            bits.append("electrolyte(s): " + ", ".join(fig_meta.electrolytes))
        if fig_meta.reference_electrode:
            bits.append(f"ref: {fig_meta.reference_electrode}")
        if bits:
            print("Metadata from caption/text (verify): " + " · ".join(bits))

    # Build a uniform list of (label, curves) panels from one of three modes:
    #   auto  -> detect each plot's axes frame and assign curves to it
    #   NxM   -> fixed grid split
    #   1x1   -> the whole figure as a single panel
    panels_list: list[tuple[str, list]] = []
    grid = [1, 1]
    if args.panels.lower() == "auto":
        detected = detect_panels(pdf, page, min_points=args.min_points,
                                 min_curve_points=max(60, args.min_points))
        panels_list = [(p.label, p.curves) for p in detected]
        grid = [len(detected), 1]
        if len(panels_list) == 1:
            panels_list = [("", panels_list[0][1])]  # single plot: no letter
        if not panels_list:  # nothing detected -> whole figure
            panels_list = [("", extract_color_groups(pdf, page, min_points=args.min_points))]
    elif args.panels == "1x1":
        panels_list = [("", extract_color_groups(pdf, page, min_points=args.min_points))]
    else:
        nx, ny = _parse_grid(args.panels)
        grid = [nx, ny]
        pmap = extract_panels(pdf, page, nx, ny, min_curve_points=args.min_points)
        for key in sorted(pmap):
            panels_list.append((panel_label(key[0], key[1], nx), pmap[key]))

    if args.panel:
        panels_list = [(lbl, cs) for lbl, cs in panels_list
                       if lbl.lower() == args.panel.lower()]
        if not panels_list:
            print(f"Panel '{args.panel}' not found (available: "
                  f"{', '.join(l or '(main)' for l, _ in panels_list) or 'none'}).")

    # CV-likeness gate up front (loop score is calibration-invariant, so we can
    # judge and drop non-CV panels before doing any calibration or file I/O).
    if getattr(args, "cv_only", False):
        kept = []
        for lbl, cs in panels_list:
            score = max((loop_metrics(dedupe(order_curve(keep_main_components(c.polylines))))
                         ["loopiness"] for c in cs), default=0.0)
            if score >= args.cv_threshold:
                kept.append((lbl, cs))
            else:
                print(f"Panel {lbl or '(figure)'}: skipped — loop score {score:.2f} "
                      f"< {args.cv_threshold} (not CV-like).")
        panels_list = kept

    img = render_page(pdf, page, zoom=3.0)
    report = {"pdf": pdf, "page": page, "panels_grid": grid,
              "calibrated": manual is not None, "curves": []}
    total = 0
    modes_seen = []
    # A single --x-ticks/--y-ticks pair describes ONE panel's axes; applying it
    # to several panels that may have different ranges would silently mis-scale
    # them. So assisted-tick calibration only fires when exactly one panel is
    # being processed (whole-figure, a grid of one, or an explicit --panel).
    ticks_apply = bool(args.x_ticks and args.y_ticks) and len(panels_list) == 1
    if args.x_ticks and args.y_ticks and len(panels_list) > 1:
        print(f"Note: {len(panels_list)} panels detected — --x-ticks/--y-ticks describe "
              f"one panel's axes, so they are NOT applied blindly to all. Re-run with "
              f"--panel <letter> to calibrate a specific panel (per-panel tick-label "
              f"crops are written to each panel_*/calib_helper.png). Panels whose axis "
              f"text is machine-readable are auto-calibrated regardless.")

    for plabel, curves in panels_list:
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
        if calib is None and ticks_apply and bbox:
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

        processed = _process_curves(curves, calib, args.resample,
                                     getattr(args, "resample_mode", "arclength"))

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

        from .qc import write_qc
        from .postprocess import order_curve
        # one ordered polyline per curve (pixel space = PDF pts x3), so the
        # fidelity check sees the real curve, not jumps between shuffled sub-paths.
        qc_curves = [{"name": pc["group"].name,
                      "xy": order_curve(list(pc["group"].polylines)) * 3,
                      "rgb": pc["group"].rgb} for pc in processed]
        qc = write_qc(pdir, img, qc_curves, panel_stem="panel",
                      title=f"{stem} p{page}" + (f" panel {plabel}" if plabel else ""))
        vec_fids = qc.get("fidelity", [])

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
        # caption legend takes priority; the in-plot text-layer legend (colour
        # swatch + adjacent label) fills colours the caption did not describe.
        try:
            plot_legend = detect_plot_legend(pdf, page, curves)
        except Exception:
            plot_legend = {}
        panel_legend = {**plot_legend, **legend_for_panel(curve_legend, plabel or "")}
        for pk, pc in enumerate(processed):
            cg = pc["group"]
            name = f"{stem}_" + (f"{plabel}_" if plabel else "") + cg.name
            # if the caption or in-plot legend says what this colour is, use it
            sample = panel_legend.get(cg.name)
            curve_label = f"{cg.name}: {sample}" if sample else cg.name
            meta = CurveMeta(
                name=name, figure=(args.figure or (f"panel {plabel}" if plabel else "")),
                curve=curve_label,
                scan_rate=args.scan_rate or (fig_meta.scan_rate if fig_meta else ""),
                x_label=pc["xlab"], x_unit=pc["xunit"],
                y_label=pc["ylab"], y_unit=pc["yunit"],
                source_pdf=pdf, method="digitized",
                comment="auto-extracted from vector PDF" + cal_note,
                extracted=(fig_meta.as_dict() if fig_meta and not fig_meta.is_empty else {}),
            )
            if calib is not None:
                paths = write_datapackage(pdir, pc["data"], meta, yaml=not args.no_yaml)
            else:
                # only CSV when uncalibrated
                p = os.path.join(pdir, name + ".csv")
                write_csv(p, pc["data"], meta)
                paths = {"csv": p}
            fid = vec_fids[pk] if pk < len(vec_fids) else {}
            report["curves"].append({
                "panel": plabel, "name": name, "color": cg.name, "rgb": list(cg.rgb),
                "sample": sample or "", "n_points": len(pc["data"]),
                "units": [pc["xunit"], pc["yunit"]],
                "loopiness": pc["loopiness"], "calibration": mode,
                "fidelity": fid.get("score"), "fidelity_grade": fid.get("grade"),
                "files": {k: os.path.relpath(v, out_dir) for k, v in paths.items()},
            })
            total += 1

    calibrated = any(m != "none" for m in modes_seen)
    report["calibrated"] = calibrated
    report["calibration_modes"] = modes_seen
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_html_report(out_dir, stem, report)
    _print_fidelity_triage(report["curves"], out_dir)

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
            page, kind = _best_figure_page(pdf, vec, ras)
            paper_out = os.path.join(out_dir, stem)
            n_curves, overlay, note, best_loop = 0, "", "", 0.0
            try:
                a = argparse.Namespace(
                    pdf=pdf, page=page, panels="auto", panel=None, calibration=None,
                    resample=800, resample_mode="arclength", min_points=60,
                    figure=None, scan_rate=None, cv_only=False, cv_threshold=0.08,
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
                    fscores = [c["fidelity"] for c in rep["curves"]
                               if c.get("fidelity") is not None]
                    worst_fid = min(fscores) if fscores else None
                for cand in glob.glob(os.path.join(paper_out, "**", "overlay.png"), recursive=True):
                    overlay = cand
                    break
            except Exception as e:  # keep the batch going
                note = f"extract error: {type(e).__name__}"
            # Loopiness gauges CV-likeness: closed loops enclosing area score high;
            # schematic lines / Nyquist arcs / axis fragments score low.
            conf = "likely CV" if best_loop >= 0.15 else ("maybe" if best_loop >= 0.05 else "unlikely CV")
            conf_color = {"likely CV": "#137333", "maybe": "#b26a00", "unlikely CV": "#a50e0e"}[conf]
            # Reference-free ink-fidelity: how well the traces sit on the ink.
            fid_txt, fid_color = "n/a", "#888"
            if worst_fid is not None:
                fid_txt = f"worst {worst_fid:.0f}/100"
                fid_color = ("#137333" if worst_fid >= 85 else
                             "#b26a00" if worst_fid >= 65 else "#a50e0e")
            summary.append((stem, kind, len(vec), len(ras), n_curves, best_loop, conf, note))
            rows.append((best_loop, worst_fid, f"""
            <div class="card">
              <h3>{stem}</h3>
              <p class="meta">{len(infos)} pages · vector figs: {len(vec)} · raster figs: {len(ras)}
                 · page {page} ({kind}) · <b>{n_curves}</b> candidate curves
                 · <b style="color:{conf_color}">{conf}</b> (loop {best_loop:.2f})
                 · ink-fidelity <b style="color:{fid_color}">{fid_txt}</b>{(' · '+note) if note else ''}</p>
              {'<img src="'+embed(overlay)+'">' if overlay else '<p class="none">no overlay</p>'}
            </div>"""))
            fid_console = f"fid={worst_fid:.0f}" if worst_fid is not None else "fid=n/a"
            print(f"  {stem:48s} {kind:14s} vec={len(vec)} ras={len(ras)} "
                  f"curves={n_curves} loop={best_loop:.2f} {fid_console} [{conf}] {note}")
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
CV-like papers first, then <b>worst ink-fidelity first</b> so the traces most in
need of a look (or a trace_assist re-draw) are at the top. "candidate curves" are
unverified; a low loop score usually means the page is a schematic/other plot, not
a CV, and low ink-fidelity means a trace drifts off the ink (often a chord).</p>
<div class="grid">{''.join(r for _, _, r in sorted(rows, key=lambda t: (t[0] < 0.05, t[1] if t[1] is not None else 101)))}</div>"""
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
    pe.add_argument("--panels", default="auto",
                    help="'auto' (detect each plot's axes frame, default), a grid "
                         "like 2x2, or 1x1 for the whole figure as one panel")
    pe.add_argument("--panel", default=None, help="single panel letter a,b,c,... (default all)")
    pe.add_argument("--calibration", default=None, help="calibration JSON for real units")
    pe.add_argument("--resample", type=int, default=1000, help="resample point count (0=off)")
    pe.add_argument("--resample-mode", choices=["arclength", "uniform-E"], default="arclength",
                    help="arclength: even along the curve (smooth). uniform-E: even in "
                         "potential per anodic/cathodic branch — potentiostat-like 'raw' sampling")
    pe.add_argument("--min-points", type=int, default=60, help="min points per curve")
    pe.add_argument("--figure", default=None, help="figure label for metadata")
    pe.add_argument("--scan-rate", default=None, help="e.g. '50 mV/s'")
    pe.add_argument("--no-yaml", action="store_true", help="skip YAML metadata")
    pe.add_argument("--no-autocalib", action="store_true",
                    help="disable automatic calibration from the PDF text layer")
    pe.add_argument("--cv-only", action="store_true",
                    help="keep only CV-like panels (closed loops); skip schematics, "
                         "micrographs, spectra, Nyquist plots by their low loop score")
    pe.add_argument("--cv-threshold", type=float, default=0.08,
                    help="min loop score for --cv-only (0..1, default 0.08)")
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
