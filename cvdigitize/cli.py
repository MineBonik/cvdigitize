"""Command-line interface for cvdigitize (vector-only).

Two commands, because the workflow is two questions:

    cvdigitize info paper.pdf          # is there anything here for me?
    cvdigitize vector-calibrate --in <folder>   # do the work

``vector-calibrate`` is the tool. It scans a folder of PDFs for native-vector
CV panels, extracts every curve from the PDF's own path geometry, and then
walks you through the panels one at a time in a browser so you can confirm the
axis calibration — the one thing that cannot be read reliably from the file.
Each panel's CSV + frictionless JSON + echemdb YAML are written the moment you
save it, so an interrupted session never loses confirmed work.

Shorthand: a bare path does the obvious thing —

    cvdigitize data\\papers          # a folder  -> vector-calibrate --in ...
    cvdigitize paper.pdf            # a PDF     -> info

If you installed via the venv directly instead of the ``cvdigitize`` launcher,
replace ``cvdigitize`` above with ``.venv\\Scripts\\python.exe -m cvdigitize``
(Windows) or ``.venv/bin/python -m cvdigitize`` (macOS/Linux).
"""
from __future__ import annotations

import argparse
import os
import sys

from .ingest import classify_pdf


def cmd_info(args) -> int:
    """Report what a PDF holds, and whether this tool can digitize it.

    The page table is cheap (one pass over each page's drawings). The panel
    summary underneath runs the *same* detection ``vector-calibrate`` uses, so
    what it reports is exactly what a scan would find — worth the few seconds
    of page rendering, since "will this paper yield anything?" is the whole
    reason to run this command. ``--quick`` skips it.
    """
    from .postprocess import dedupe, keep_main_components, loop_metrics, order_curve
    from .veccal.scan import candidate_pages

    pdf = args.pdf
    if not os.path.isfile(pdf):
        print(f"Not a file: {pdf}")
        return 2

    infos = classify_pdf(pdf)
    candidates = candidate_pages(pdf)

    print(f"{pdf}  ({len(infos)} page{'s' if len(infos) != 1 else ''})\n")
    print(f"{'page':>4}  {'kind':<14} {'draws':>6} {'items':>7} {'colours':>8} {'img%':>5}")
    for pi in infos:
        mark = "  <- candidate" if pi.number in candidates else ""
        print(f"{pi.number:>4}  {pi.kind:<14} {pi.n_drawings:>6} {pi.n_curve_items:>7} "
              f"{pi.n_stroke_colors:>8} {pi.image_area_frac * 100:>4.0f}%{mark}")

    if not candidates:
        raster = [pi.number for pi in infos if pi.kind == "raster"]
        print("\nNo pages carry enough vector geometry to hold a figure.")
        if raster:
            print(f"Pages {raster} look like raster (scanned/embedded-image) figures. "
                  f"This tool is vector-only and cannot digitize those.")
        return 1

    print(f"\nCandidate figure pages: {candidates}")
    if args.quick:
        print("(--quick: skipped panel detection)")
        return 0

    from .vector_extract import detect_panels
    from .veccal.scan import MIN_PANEL_H_PT, MIN_PANEL_W_PT

    print("\nCV panels found (same detection vector-calibrate uses):")
    total_panels = total_curves = 0
    for page in candidates:
        try:
            panels = detect_panels(pdf, page, min_points=60, min_curve_points=60)
        except Exception as exc:
            print(f"  page {page:>3}: detection failed ({type(exc).__name__})")
            continue
        keep = []
        for p in panels:
            # Same two gates the scan applies, in the same order: a plot too
            # small to read tick labels off is not a work unit, then loop score.
            x0, y0, x1, y1 = p.frame_pdf
            if (x1 - x0) < MIN_PANEL_W_PT or (y1 - y0) < MIN_PANEL_H_PT:
                continue
            score = max((loop_metrics(dedupe(order_curve(
                keep_main_components(c.polylines))))["loopiness"]
                for c in p.curves), default=0.0)
            if score >= args.cv_threshold:
                keep.append((p, score))
        if not keep:
            continue
        for p, score in keep:
            total_panels += 1
            total_curves += len(p.curves)
            label = f"panel {p.label}" if p.label else "(single plot)"
            print(f"  page {page:>3}: {label:<14} {len(p.curves)} curve(s), "
                  f"loop score {score:.2f}")

    if not total_panels:
        print("  none — vector content is present, but nothing scored as a CV.")
        print(f"  (lower the bar with --cv-threshold, currently {args.cv_threshold})")
        return 1

    print(f"\n{total_panels} CV panel(s), {total_curves} curve(s) total.")
    print(f'Next:  cvdigitize vector-calibrate --in "{os.path.dirname(pdf) or "."}"')
    return 0


def cmd_vector_calibrate(args) -> int:
    """Scan a folder for vector CV panels, then calibrate them by hand.

    Deliberately the only extraction path. It stops at the one thing the
    machine cannot read reliably — what the axis numbers are — and presents
    each panel with its curves drawn over the original figure so a human
    confirms them against the picture.
    """
    from .veccal.server import run
    return run(args.source, args.work,
               args.out or os.path.join(args.work, "curves"),
               port=args.port, rescan=args.rescan,
               cv_threshold=args.cv_threshold,
               open_browser=not args.no_open,
               resample=args.resample, resample_mode=args.resample_mode,
               workers=args.workers)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cvdigitize",
        description="Digitize Cyclic Voltammetry curves out of native-vector PDFs.",
        epilog='Quick start: cvdigitize vector-calibrate --in "data\\papers"',
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="what does this PDF hold, and can I use it?")
    pi.add_argument("pdf")
    pi.add_argument("--quick", action="store_true",
                    help="page table only; skip the (slower) panel detection")
    pi.add_argument("--cv-threshold", type=float, default=0.08,
                    help="min loop score for a panel to count as a CV (0..1)")
    pi.set_defaults(func=cmd_info)

    pv = sub.add_parser("vector-calibrate",
                        help="find every vector CV panel in a folder, then "
                             "calibrate and name them one by one in the browser")
    pv.add_argument("--in", dest="source", default=None, metavar="FOLDER",
                    help="folder of PDFs to scan (needed on the first run; "
                         "omit afterwards to resume the existing scan)")
    pv.add_argument("--work", default=os.path.join("data", "out", "vector_curated"),
                    help="where the scan and your progress live "
                         "(default data/out/vector_curated)")
    pv.add_argument("--out", default=None,
                    help="where calibrated curves are written (default <work>/curves)")
    pv.add_argument("--rescan", action="store_true",
                    help="re-detect panels even if a scan exists (keeps the "
                         "status, names and calibrations you already entered)")
    pv.add_argument("--cv-threshold", type=float, default=0.08,
                    help="min loop score for a panel to count as a CV (0..1)")
    pv.add_argument("--resample", type=int, default=1000,
                    help="resample point count per curve (0=off)")
    pv.add_argument("--resample-mode", choices=["arclength", "uniform-E"],
                    default="arclength",
                    help="arclength: even along the curve. uniform-E: even in "
                         "potential per branch — potentiostat-like sampling")
    pv.add_argument("--workers", type=int, default=None,
                    help="parallel scan workers (default: CPU count - 1; "
                         "1 to scan one paper at a time)")
    pv.add_argument("--port", type=int, default=8756)
    pv.add_argument("--no-open", action="store_true",
                    help="don't auto-open a browser tab")
    pv.set_defaults(func=cmd_vector_calibrate)
    return p


_KNOWN_COMMANDS = {"info", "vector-calibrate", "-h", "--help"}


def _with_implicit_command(argv: list[str]) -> list[str]:
    """Let a bare path work: a folder means scan it, a PDF means inspect it.

    Typing the subcommand is the first thing a new user gets wrong, and the
    right command is unambiguous from what the path *is* — there is exactly one
    thing to do with a folder and one with a single file.
    """
    if not argv or argv[0] in _KNOWN_COMMANDS or argv[0].startswith("-"):
        return argv
    if os.path.isdir(argv[0]):
        return ["vector-calibrate", "--in", *argv]
    return ["info", *argv]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(_with_implicit_command(argv))
    return args.func(args)
