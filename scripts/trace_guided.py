"""Human-in-the-loop guided digitization (standalone; not wired into the CLI).

Guided tracing is for **raster / scanned** figures — the hard cases. Vector
figures already auto-extract near-perfectly, so send those through
``python -m cvdigitize`` instead; don't hand-trace them.

**Recommended workflow — crop first, then triage** (fixes the two failure
modes seen on composite multi-panel figures: a schematic's shapes read as
dozens of spurious plot frames, or several sub-plots with different axes get
lumped into one panel with one calibration):

  1. render whole pages, no panel auto-detection at all:

         python scripts/trace_guided.py pages data/papers -o data/out/pages

  2. open tools/trace_assist.html, load that folder, and in **Crop mode**
     drag a box around ONE real plot at a time; Alt+drag inside it to mask
     out a legend/text box sitting on the axes. Classify each crop:
     **Normal CV** (processed automatically), **Strange CV** (you'll trace it
     by hand), or **Not a CV** (excluded, just recorded). Export
     ``triage.json`` (bundles every crop: image + decision + any curves you
     traced + calibration).

  3. let normal CVs auto-extract, and get a fade-by-eye ``check.html`` for
     every one (same QC as the main CLI) to accept or send onward:

         python scripts/trace_guided.py triage triage.json -o data/out/triaged

     "Strange" crops you already traced in step 2 are finalized the same way;
     ones you only cropped/classified (no curves drawn yet) get their
     panel.png saved for tracing later. Curves whose auto-extraction check
     looks wrong: open trace_assist on that panel.png and re-trace by hand
     (step 4).

**Or, for a single figure you already know how to isolate** (the older,
still-supported path):

  a. export one panel directly:

         python scripts/trace_guided.py panel paper.pdf 4 -o panel.png

  b. (batch) pull every raster figure out of a folder of PDFs into a gallery,
     relying on automatic figure/frame detection (fine for simple, single-plot
     pages; use the crop-first workflow above for composite ones):

         python scripts/trace_guided.py gallery data/papers -o data/out/gallery

  c. after scribbling a rough guide along each curve (and, optionally, clicking
     the four axis calibration points) and exporting guides.json, turn the rough
     guides into pixel-accurate curves:

         python scripts/trace_guided.py run panel.png guides.json -o out/

     writes one CSV per curve plus overlay.png. If guides.json carries a
     calibration (per-curve, or one shared panel-level default) the CSVs are
     in real units (E, j); otherwise pixel coords.

The rough guide only resolves *which* ink is which curve and the sweep order;
the trace snaps to the real ink (see ``cvdigitize.guided.extract_near_guide``).
This is deliberately separate from ``python -m cvdigitize`` — merge later if
wanted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cmd_panel(args):
    import cv2
    from cvdigitize.raster_extract import find_image_regions, render_region
    from cvdigitize.ingest import render_page
    regions = find_image_regions(args.pdf, args.page)
    if regions and args.region is not None:
        img = render_region(args.pdf, args.page, regions[args.region].bbox, zoom=args.zoom)
    elif regions:
        # largest image region on the page by area
        r = max(regions, key=lambda r: (r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1]))
        img = render_region(args.pdf, args.page, r.bbox, zoom=args.zoom)
    else:
        img = render_page(args.pdf, args.page, zoom=args.zoom)
    out = args.out or f"panel_p{args.page}.png"
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {out}  ({img.shape[1]}x{img.shape[0]} px) — load it in tools/trace_assist.html")


def _cmd_pages(args):
    """Render whole PDF pages to PNG — no frame/panel auto-detection at all.

    ``gallery`` auto-crops each figure and auto-splits it into detected plot
    frames; that auto-detection is exactly what falls over on composite
    figures (a crystal-structure schematic's sphere shapes get misread as
    dozens of spurious plot frames, or several real sub-plots with different
    axes get lumped into one "panel" with one shared calibration — both
    measured on chen_2024_deconvolution_4958's Figure 1). This command instead
    hands the human the raw page, unmodified, to crop and triage by hand in
    trace_assist.html's Crop mode — slower per-figure, but immune to that
    whole failure class. Skips pages with no image and little vector drawing
    (unlikely to hold a figure at all); use --all to render every page."""
    import glob
    import cv2
    from cvdigitize.ingest import classify_pdf, render_page

    pdfs = ([args.src] if args.src.lower().endswith(".pdf")
            else sorted(glob.glob(os.path.join(args.src, "*.pdf"))))
    if not pdfs:
        sys.exit(f"no PDFs found at {args.src}")
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for pdf in pdfs:
        stem = os.path.splitext(os.path.basename(pdf))[0][:40]
        try:
            infos = classify_pdf(pdf)
        except Exception as e:
            print(f"  ! {os.path.basename(pdf)}: {e}"); continue
        for pi in infos:
            if not args.all and pi.n_images == 0 and pi.n_curve_items < 50:
                continue                                   # near-certainly no figure
            try:
                img = render_page(pdf, pi.number, zoom=args.zoom)
            except Exception:
                continue
            name = f"{stem}_p{pi.number}.png"
            cv2.imwrite(os.path.join(args.out, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            total += 1
    print(f"wrote {total} page render(s) to {args.out}/")
    print(f"open tools/trace_assist.html, load the {args.out}/ folder, and use "
          f"Crop mode to carve out + triage each real figure by hand")


def _cmd_gallery(args):
    """Extract every RASTER figure from a folder of PDFs into a PNG gallery.

    Vector-curve figure pages are skipped on purpose — they auto-extract well,
    so guided tracing is a waste of effort there. Full-page scans are located to
    a tight crop; embedded figure images are exported as-is."""
    import glob
    import cv2
    from cvdigitize.ingest import classify_pdf
    from cvdigitize.raster_extract import (find_image_regions, render_region,
                                           locate_plot_regions)

    pdfs = ([args.src] if args.src.lower().endswith(".pdf")
            else sorted(glob.glob(os.path.join(args.src, "*.pdf"))))
    if not pdfs:
        sys.exit(f"no PDFs found at {args.src}")
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for pdf in pdfs:
        stem = os.path.splitext(os.path.basename(pdf))[0][:40]
        try:
            infos = classify_pdf(pdf)
        except Exception as e:
            print(f"  ! {os.path.basename(pdf)}: {e}"); continue
        for pi in infos:
            # skip pages that are clearly vector CV figures (handled automatically)
            if pi.n_curve_items > 20 and pi.kind != "raster":
                continue
            try:
                regions = find_image_regions(pdf, pi.number)
            except Exception:
                regions = []
            if not (regions or pi.kind == "raster"):
                continue
            regions = regions or [None]
            for ri, region in enumerate(regions):
                bbox = region.bbox if region is not None else None
                try:
                    img = (render_region(pdf, pi.number, bbox, zoom=args.zoom)
                           if bbox is not None else
                           __import__("cvdigitize.ingest", fromlist=["render_page"])
                           .render_page(pdf, pi.number, zoom=args.zoom))
                except Exception:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                boxes = locate_plot_regions(gray) or [(0, 0, img.shape[1], img.shape[0])]
                for bi, (x0, y0, x1, y1) in enumerate(sorted(boxes, key=lambda b: b[1])):
                    crop = img[y0:y1, x0:x1]
                    if crop.shape[0] < 40 or crop.shape[1] < 40:
                        continue
                    name = f"{stem}_p{pi.number}_r{ri}_{bi}.png"
                    cv2.imwrite(os.path.join(args.out, name),
                                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    total += 1
    print(f"wrote {total} raster figure crop(s) to {args.out}/")
    print(f"open tools/trace_assist.html and load the whole {args.out}/ folder")


def _apply_calibration(poly: np.ndarray, cal: dict) -> tuple:
    """Map an (x, y) pixel polyline to real (E, j) using 4 axis anchor points.

    ``cal`` = {'E1':{'px':[x,y],'value':v}, 'E2':..., 'j1':..., 'j2':...,
    'E_unit','E_ref','j_unit'}. E uses the anchors' x-pixels, j their y-pixels
    (axes assumed aligned to the image, as the user places them)."""
    def lin(p_lo, v_lo, p_hi, v_hi):
        if p_hi == p_lo:
            return None
        s = (v_hi - v_lo) / (p_hi - p_lo)
        return lambda p: v_lo + (p - p_lo) * s
    try:
        fx = lin(cal["E1"]["px"][0], cal["E1"]["value"],
                 cal["E2"]["px"][0], cal["E2"]["value"])
        fy = lin(cal["j1"]["px"][1], cal["j1"]["value"],
                 cal["j2"]["px"][1], cal["j2"]["value"])
    except (KeyError, TypeError, IndexError):
        return None, None, None
    if fx is None or fy is None:
        return None, None, None
    E = fx(poly[:, 0]); j = fy(poly[:, 1])
    header = (f"E [{cal.get('E_unit','V')} vs {cal.get('E_ref','')}]".strip(),
              f"j [{cal.get('j_unit','')}]".strip())
    return np.column_stack([E, j]), header, True


def _cmd_run(args):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cvdigitize.guided import extract_guides

    bgr = cv2.imread(args.panel)
    if bgr is None:
        sys.exit(f"could not read image: {args.panel}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with open(args.guides, encoding="utf-8") as f:
        doc = json.load(f)
    guides = doc.get("guides", [])
    panel_calib = doc.get("calibration")
    # a curve's own calibration (set on it specifically in trace_assist) wins;
    # otherwise it falls back to the panel-default one — this is what makes
    # "one curve = one calibration, with an option to share" work: most curves
    # in a plot share one set of axes (no override needed), but a curve traced
    # from a different plot within the same session isn't silently mis-scaled.
    per_curve_calib = {g.get("name"): g["calibration"] for g in guides if g.get("calibration")}
    if not guides:
        sys.exit("no guides in the JSON")

    # a per-curve brush radius (from the tracer) overrides the global default
    results = extract_guides(rgb, guides, radius=args.radius, return_gaps=True)
    os.makedirs(args.out, exist_ok=True)
    calibrated = bool(panel_calib) or bool(per_curve_calib)

    qc_curves = []
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    for res in results:
        poly = res["polyline_px"]
        name = res["name"] or "curve"
        qc_curves.append({"name": name, "xy": poly, "rgb": None})
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "curve"
        cal = per_curve_calib.get(name, panel_calib)
        real, header, ok = (_apply_calibration(poly, cal) if cal else (None, None, False))
        with open(os.path.join(args.out, f"{safe}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            if ok:
                w.writerow(list(header))
                w.writerows(np.round(real, 5))
            else:
                w.writerow(["x_px", "y_px"])
                w.writerows(np.round(poly, 2))
        ax.plot(poly[:, 0], poly[:, 1], lw=1.2, label=name)
        gaps = res.get("gaps") or []
        for k, (p0, p1) in enumerate(gaps):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "--", color="0.5", lw=1.5,
                    label="untraced gap" if k == 0 else None)
        n_gap = len(gaps)
        gap_note = f", {n_gap} untraced gap(s) — see dashed grey lines" if n_gap else ""
        print(f"  {name}: {len(poly)} points -> {safe}.csv{gap_note}")
        if n_gap:
            print(f"    tip: your strokes don't cover the whole curve there; "
                  f"add more scribbles along the dashed spans and re-run for full accuracy")
    ax.legend(fontsize=8)
    ax.set_title("guided extraction (solid = traced on your guide, dashed grey = untraced gap)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "overlay.png"), dpi=110)
    plt.close(fig)
    units = "real units (E, j)" if calibrated else "pixel coords (no calibration in guides.json)"
    print(f"wrote {len(results)} curve(s) in {units} + overlay.png to {args.out}/")
    if not results:
        print("  (no ink found in any corridor — widen with --radius or redraw guides)")
        return

    from cvdigitize.qc import write_qc
    qc = write_qc(args.out, rgb, qc_curves,
                  panel_stem=os.path.splitext(os.path.basename(args.panel))[0],
                  title=os.path.splitext(os.path.basename(args.panel))[0])
    fids = qc.get("fidelity", [])
    worst = min((f["score"] for f in fids if f["score"] is not None), default=None)
    tag = f" — worst ink-fidelity {worst:.0f}/100" if worst is not None else ""
    print(f"  QC: open {os.path.basename(qc['check_html'])} to fade the trace over "
          f"the figure by eye{tag} (+ curve_overlay.png transparent PNG)")


def _decode_data_url(data_url: str):
    import base64
    import cv2
    _, b64 = data_url.split(",", 1)
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "curve"


def _write_curve_csv(path: str, poly: np.ndarray, cal: dict | None):
    real, header, ok = (_apply_calibration(poly, cal) if cal else (None, None, False))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if ok:
            w.writerow(list(header)); w.writerows(np.round(real, 5))
        else:
            w.writerow(["x_px", "y_px"]); w.writerows(np.round(poly, 2))
    return ok


def _cmd_triage(args):
    """Process a triage.json exported by trace_assist.html's Crop mode.

    "cv" crops are auto-extracted (the same detect-frame + colour-split
    pipeline the main CLI uses, just pointed at an already-cropped image
    instead of a PDF region — see ``raster_extract.extract_from_cropped_image``).
    "strange" crops with hand-drawn guides are finalized the same way
    ``run`` does; ones with no guides yet just get their panel.png saved for
    later tracing. "not_cv" entries (from the JSON's "rejected" list) are
    only recorded — nothing to extract from a shape that isn't a curve."""
    import cv2
    from cvdigitize.guided import extract_guides
    from cvdigitize.raster_extract import extract_from_cropped_image
    from cvdigitize.qc import write_qc

    with open(args.triage, encoding="utf-8") as f:
        doc = json.load(f)
    panels = doc.get("panels", [])
    rejected = doc.get("rejected", [])
    if not panels and not rejected:
        sys.exit("no panels or rejected crops in the triage JSON")

    os.makedirs(args.out, exist_ok=True)
    report = {"source": args.triage, "panels": [], "rejected": rejected}

    for p in panels:
        name = p.get("name") or "panel"
        pdir = os.path.join(args.out, _safe(name))
        os.makedirs(pdir, exist_ok=True)
        rgb = _decode_data_url(p["image"])
        decision = p.get("decision")
        panel_cal = p.get("calibration")
        entry = {"name": name, "decision": decision, "sourcePage": p.get("sourcePage")}

        if decision == "cv":
            result = extract_from_cropped_image(rgb)
            curves = result.get("curves", [])
            qc_curves, n_calibrated = [], 0
            for c in curves:
                poly = c["polyline_px"]
                ok = _write_curve_csv(os.path.join(pdir, f"{_safe(c['name'])}.csv"), poly, panel_cal)
                n_calibrated += ok
                qc_curves.append({"name": c["name"], "xy": poly, "rgb": c["rgb"]})  # already fractional 0..1
            if not qc_curves:
                entry["note"] = "automatic extraction found no curve — try Strange CV + hand-trace instead"
            else:
                qc = write_qc(pdir, rgb, qc_curves, panel_stem="panel", title=name)
                entry["fidelity"] = qc.get("fidelity", [])
                entry["calibrated"] = n_calibrated == len(curves) and len(curves) > 0
            entry["n_curves"] = len(curves)

        elif decision == "strange":
            guides = p.get("guides", [])
            traced = [g for g in guides if g.get("strokes")]
            if not traced:
                cv2.imwrite(os.path.join(pdir, "panel.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                entry["note"] = "no curves traced yet — open panel.png in trace_assist.html"
            else:
                per_curve_cal = {g.get("name"): g["calibration"] for g in guides if g.get("calibration")}
                results = extract_guides(rgb, guides, return_gaps=True)
                qc_curves, n_calibrated = [], 0
                for res in results:
                    poly = res["polyline_px"]; gname = res["name"] or "curve"
                    cal = per_curve_cal.get(gname, panel_cal)
                    ok = _write_curve_csv(os.path.join(pdir, f"{_safe(gname)}.csv"), poly, cal)
                    n_calibrated += ok
                    qc_curves.append({"name": gname, "xy": poly, "rgb": None})
                if qc_curves:
                    qc = write_qc(pdir, rgb, qc_curves, panel_stem="panel", title=name)
                    entry["fidelity"] = qc.get("fidelity", [])
                    entry["calibrated"] = n_calibrated == len(results) and len(results) > 0
                entry["n_curves"] = len(results)
        report["panels"].append(entry)

    with open(os.path.join(args.out, "triage_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    n_cv = sum(1 for e in report["panels"] if e["decision"] == "cv")
    n_strange = sum(1 for e in report["panels"] if e["decision"] == "strange")
    print(f"processed {n_cv} normal CV panel(s), {n_strange} strange panel(s), "
          f"{len(rejected)} rejected (not-a-CV) crop(s) recorded")
    print(f"wrote {args.out}/triage_report.json — open each panel's check.html to review by eye")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Guided (human-in-the-loop) digitization.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gallery", help="extract every raster figure from a folder of PDFs")
    g.add_argument("src", help="a PDF or a folder of PDFs")
    g.add_argument("--zoom", type=float, default=4.0)
    g.add_argument("-o", "--out", default="figure_gallery")
    g.set_defaults(func=_cmd_gallery)

    pg = sub.add_parser("pages", help="render whole pages (no panel auto-detection) for hand-cropping")
    pg.add_argument("src", help="a PDF or a folder of PDFs")
    pg.add_argument("--zoom", type=float, default=4.0)
    pg.add_argument("--all", action="store_true", help="render every page, even ones unlikely to hold a figure")
    pg.add_argument("-o", "--out", default="page_renders")
    pg.set_defaults(func=_cmd_pages)

    p = sub.add_parser("panel", help="export a single figure panel PNG for the tracer")
    p.add_argument("pdf")
    p.add_argument("page", type=int)
    p.add_argument("--region", type=int, default=None, help="image-region index on the page")
    p.add_argument("--zoom", type=float, default=4.0)
    p.add_argument("-o", "--out")
    p.set_defaults(func=_cmd_panel)

    r = sub.add_parser("run", help="guided extraction from panel.png + guides.json")
    r.add_argument("panel")
    r.add_argument("guides")
    r.add_argument("--radius", type=int, default=None, help="corridor half-width in px")
    r.add_argument("-o", "--out", default="guided_out")
    r.set_defaults(func=_cmd_run)

    tg = sub.add_parser("triage", help="process a triage.json from trace_assist.html's Crop mode")
    tg.add_argument("triage")
    tg.add_argument("-o", "--out", default="triaged_out")
    tg.set_defaults(func=_cmd_triage)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
