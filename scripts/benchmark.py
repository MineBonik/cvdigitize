"""Accuracy benchmark: compare cvdigitize output against reference CSVs.

Point it at a PDF and a folder of hand-digitized reference curves (e.g. the
existing echemdb CSVs for that paper). It extracts every curve, matches each
reference to the extracted curve it best fits, and reports a normalised-shape
error (symmetric nearest-neighbour / Chamfer distance) per curve plus an
aggregate — turning "looks right" into a measured number.

Why normalised shape? It isolates *extraction* fidelity from calibration: both
curves are min-max scaled to a unit box before comparison, so a difference in
axis calibration does not masquerade as an extraction error. (Chamfer < ~0.01
is an excellent overlay; the rizo reference reads 0.001-0.007.)

Usage:
    .venv\\Scripts\\python.exe scripts\\benchmark.py <paper.pdf> <ref_dir> [--page N]
    .venv\\Scripts\\python.exe scripts\\benchmark.py --suite <dir>

  <ref_dir>  folder of reference CSVs (2 columns, optional header). In --suite
             mode, <dir> holds <stem>.pdf next to a <stem>_ref/ folder each.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvdigitize.vector_extract import detect_panels, extract_color_groups
from cvdigitize.postprocess import order_curve, dedupe, resample_arclength, keep_main_components


def _load_csv(path: str) -> np.ndarray:
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except ValueError:
                continue  # header line
    return np.asarray(rows, dtype=float)


def _norm(xy: np.ndarray) -> np.ndarray:
    out = xy.astype(float).copy()
    for k in (0, 1):
        lo, hi = out[:, k].min(), out[:, k].max()
        out[:, k] = (out[:, k] - lo) / (hi - lo) if hi > lo else 0.0
    return out


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    ta, tb = cKDTree(a), cKDTree(b)
    return float((tb.query(a)[0].mean() + ta.query(b)[0].mean()) / 2)


def _best_page(pdf: str) -> int:
    """Figure page with the most vector curve content (skips logo/header pages
    that carry a few coloured marks — the same trap the CLI's page picker avoids)."""
    from cvdigitize.vector_extract import find_figure_pages
    from cvdigitize.ingest import classify_pdf
    figs = find_figure_pages(pdf)
    if not figs:
        return 0
    items = {pi.number: pi.n_curve_items for pi in classify_pdf(pdf)}
    return max(figs, key=lambda p: items.get(p, 0))


def extracted_curves(pdf: str, page: int | None) -> list[np.ndarray]:
    """All digitized curves in the paper as ordered, resampled polylines."""
    if page is None:
        page = _best_page(pdf)
    curves = []
    panels = detect_panels(pdf, page)
    groups = ([c for p in panels for c in p.curves]
              if panels else extract_color_groups(pdf, page, min_points=60))
    for cg in groups:
        loop = dedupe(order_curve(keep_main_components(cg.polylines)))
        if len(loop) >= 20:
            curves.append(resample_arclength(loop, n=600))
    return curves


def benchmark_pdf(pdf: str, ref_dir: str, page: int | None = None) -> dict:
    refs = {os.path.basename(p): _load_csv(p)
            for p in sorted(glob.glob(os.path.join(ref_dir, "*.csv")))}
    refs = {k: v for k, v in refs.items() if len(v) >= 10}
    curves = extracted_curves(pdf, page)
    if not curves:
        return {"pdf": pdf, "error": "no curves extracted", "results": []}

    norm_curves = [_norm(c) for c in curves]
    results = []
    for name, ref in refs.items():
        rn = _norm(ref)
        dists = [chamfer(rn, nc) for nc in norm_curves]
        best = int(np.argmin(dists))
        results.append({"reference": name, "chamfer": round(dists[best], 5),
                        "matched_extracted_index": best})
    ch = [r["chamfer"] for r in results]
    return {"pdf": os.path.basename(pdf), "n_reference": len(refs),
            "n_extracted": len(curves),
            "mean_chamfer": round(float(np.mean(ch)), 5) if ch else None,
            "max_chamfer": round(float(np.max(ch)), 5) if ch else None,
            "results": sorted(results, key=lambda r: r["chamfer"])}


def _print_report(rep: dict):
    print(f"\n=== {rep['pdf']} ===")
    if rep.get("error"):
        print(f"  {rep['error']}")
        return
    print(f"  extracted {rep['n_extracted']} curves, {rep['n_reference']} references")
    print(f"  {'reference':44s} {'Chamfer':>9}  {'grade'}")
    for r in rep["results"]:
        grade = ("excellent" if r["chamfer"] < 0.01 else
                 "good" if r["chamfer"] < 0.03 else
                 "fair" if r["chamfer"] < 0.08 else "poor")
        print(f"  {r['reference'][:44]:44s} {r['chamfer']:>9.4f}  {grade}")
    print(f"  --> mean {rep['mean_chamfer']}  max {rep['max_chamfer']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Accuracy benchmark vs reference CSVs.")
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("ref_dir", nargs="?")
    ap.add_argument("--page", type=int, default=None)
    ap.add_argument("--suite", help="dir with <stem>.pdf + <stem>_ref/ pairs")
    args = ap.parse_args(argv)

    reports = []
    if args.suite:
        for pdf in sorted(glob.glob(os.path.join(args.suite, "*.pdf"))):
            stem = os.path.splitext(os.path.basename(pdf))[0]
            ref_dir = os.path.join(args.suite, stem + "_ref")
            if os.path.isdir(ref_dir):
                reports.append(benchmark_pdf(pdf, ref_dir))
    elif args.pdf and args.ref_dir:
        reports.append(benchmark_pdf(args.pdf, args.ref_dir, args.page))
    else:
        ap.error("give <pdf> <ref_dir>, or --suite <dir>")

    for rep in reports:
        _print_report(rep)
    all_ch = [r["chamfer"] for rep in reports for r in rep.get("results", [])]
    if all_ch:
        print(f"\nOVERALL: {len(all_ch)} curves across {len(reports)} paper(s) · "
              f"mean Chamfer {np.mean(all_ch):.4f} · "
              f"{100*np.mean([c < 0.03 for c in all_ch]):.0f}% good-or-better (<0.03)")


if __name__ == "__main__":
    main()
