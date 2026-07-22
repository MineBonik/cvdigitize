"""Human-in-the-loop guided digitization (standalone; not wired into the CLI).

Guided tracing is for **raster / scanned** figures — the hard cases. Vector
figures already auto-extract near-perfectly, so send those through
``python -m cvdigitize`` instead; don't hand-trace them.

Workflow around the paint-like tracer in ``tools/trace_assist.html``:

  0. (batch) pull every raster figure out of a folder of PDFs into a gallery of
     PNGs the tracer can open directly:

         python scripts/trace_guided.py gallery data/papers -o data/out/gallery

  1. or export a single figure panel:

         python scripts/trace_guided.py panel paper.pdf 4 -o panel.png

  2. after scribbling a rough guide along each curve (and, optionally, clicking
     the four axis calibration points) and exporting guides.json, turn the rough
     guides into pixel-accurate curves:

         python scripts/trace_guided.py run panel.png guides.json -o out/

     writes one CSV per curve plus overlay.png. If guides.json carries a
     calibration the CSVs are in real units (E, j); otherwise pixel coords.

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
    calib = doc.get("calibration")
    if not guides:
        sys.exit("no guides in the JSON")

    # a per-curve brush radius (from the tracer) overrides the global default
    results = extract_guides(rgb, guides, radius=args.radius, return_gaps=True)
    os.makedirs(args.out, exist_ok=True)
    calibrated = bool(calib)

    qc_curves = []
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    for res in results:
        poly = res["polyline_px"]
        name = res["name"] or "curve"
        qc_curves.append({"name": name, "xy": poly, "rgb": None})
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "curve"
        real, header, ok = (_apply_calibration(poly, calib) if calibrated
                            else (None, None, False))
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Guided (human-in-the-loop) digitization.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gallery", help="extract every raster figure from a folder of PDFs")
    g.add_argument("src", help="a PDF or a folder of PDFs")
    g.add_argument("--zoom", type=float, default=4.0)
    g.add_argument("-o", "--out", default="figure_gallery")
    g.set_defaults(func=_cmd_gallery)

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

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
