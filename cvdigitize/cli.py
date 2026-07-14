"""Command-line interface for cvdigitize.

Examples
--------
    # inspect a PDF: which pages hold vector CV figures?
    python -m cvdigitize info paper.pdf

    # extract curves from the auto-detected figure page (whole figure)
    python -m cvdigitize extract paper.pdf

    # multi-panel figure: split 2x2 and process panel (a) only, with calibration
    python -m cvdigitize extract paper.pdf --panels 2x2 --panel a \\
        --calibration configs/paper_fig1a.calib.json --scan-rate "50 mV/s"

    # render a page with a pixel grid to read off axis-anchor pixels for calibration
    python -m cvdigitize grid paper.pdf --page 1
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
from .postprocess import order_curve, dedupe, resample_arclength, keep_main_components
from .calibrate import Calibration
from .package import write_datapackage, CurveMeta


# --------------------------------------------------------------------------- #
def _parse_grid(s: str) -> tuple[int, int]:
    s = s.lower().replace("×", "x")
    nx, ny = (int(v) for v in s.split("x"))
    return nx, ny


def _plot_color(rgb):
    return (max(0, min(1, rgb[0])), max(0, min(1, rgb[1])), max(0, min(1, rgb[2])))


def cmd_info(args):
    infos = classify_pdf(args.pdf)
    figs = find_figure_pages(args.pdf)
    print(f"{args.pdf}\n{'page':>4} {'kind':<15} {'draw':>6} {'items':>7} "
          f"{'colors':>7} {'imgs':>5} {'img%':>6}")
    for pi in infos:
        star = " *" if pi.number in figs else ""
        print(f"{pi.number:>4} {pi.kind:<15} {pi.n_drawings:>6} {pi.n_curve_items:>7} "
              f"{pi.n_stroke_colors:>7} {pi.n_images:>5} {pi.image_area_frac*100:>5.0f}%{star}")
    print(f"\nFigure-page candidates (*): {figs or 'none'}")
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
                    "xunit": xunit, "yunit": yunit})
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
    page = args.page if args.page is not None else (find_figure_pages(pdf) or [0])[0]
    out_dir = args.out or os.path.join("data", "out", stem)
    os.makedirs(out_dir, exist_ok=True)

    calib = Calibration.load(args.calibration) if args.calibration else None
    nx, ny = _parse_grid(args.panels)

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
              "calibrated": calib is not None, "curves": []}
    total = 0

    for key in sorted(panels):
        if wanted is not None and key != wanted:
            continue
        curves = panels[key]
        plabel = panel_label(key[0], key[1], nx) if (nx, ny) != (1, 1) else ""
        pdir = os.path.join(out_dir, f"panel_{plabel}") if plabel else out_dir
        os.makedirs(pdir, exist_ok=True)

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

        for pc in processed:
            cg = pc["group"]
            name = f"{stem}_" + (f"{plabel}_" if plabel else "") + cg.name
            meta = CurveMeta(
                name=name, figure=(args.figure or (f"panel {plabel}" if plabel else "")),
                curve=cg.name, scan_rate=args.scan_rate or "",
                x_label=pc["xlab"], x_unit=pc["xunit"],
                y_label=pc["ylab"], y_unit=pc["yunit"],
                source_pdf=pdf, method="digitized",
                comment="auto-extracted from vector PDF"
                        + ("" if calib else "; UNCALIBRATED (normalised coords)"),
            )
            if calib is not None:
                paths = write_datapackage(pdir, pc["data"], meta, yaml=not args.no_yaml)
            else:
                # only CSV when uncalibrated
                from .package import write_csv
                p = os.path.join(pdir, name + ".csv")
                write_csv(p, pc["data"], meta)
                paths = {"csv": p}
            report["curves"].append({
                "panel": plabel, "name": name, "color": cg.name, "rgb": list(cg.rgb),
                "n_points": len(pc["data"]), "units": [pc["xunit"], pc["yunit"]],
                "files": {k: os.path.relpath(v, out_dir) for k, v in paths.items()},
            })
            total += 1

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_html_report(out_dir, stem, report)

    print(f"Extracted {total} curve(s) from {stem} p{page} "
          f"({'calibrated' if calib else 'UNCALIBRATED — pass --calibration for real units'}).")
    print(f"Output: {out_dir}")
    if not calib:
        print("Tip: run `python -m cvdigitize grid "
              f"\"{pdf}\" --page {page}` to read axis anchors and build a calibration JSON.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cvdigitize",
                                description="Digitize CV curves from vector PDFs.")
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
    pe.add_argument("--out", default=None, help="output dir (default data/out/<pdf>)")
    pe.set_defaults(func=cmd_extract)

    pg = sub.add_parser("grid", help="render a page with a pixel grid for calibration")
    pg.add_argument("pdf")
    pg.add_argument("--page", type=int, default=0)
    pg.add_argument("--step", type=float, default=20.0, help="grid step in PDF points")
    pg.add_argument("--zoom", type=float, default=3.0)
    pg.add_argument("--out", default=None)
    pg.set_defaults(func=cmd_grid)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
