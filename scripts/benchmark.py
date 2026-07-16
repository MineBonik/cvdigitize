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
    .venv\\Scripts\\python.exe scripts\\benchmark.py --echemdb <pdf_dir> <echemdb_svgdigitizer_root>
    .venv\\Scripts\\python.exe scripts\\benchmark.py --echemdb-one <paper.pdf> <echemdb_entry_dir>

  <ref_dir>  folder of reference CSVs (2 columns, optional header). In --suite
             mode, <dir> holds <stem>.pdf next to a <stem>_ref/ folder each.

  --echemdb  ground-truth mode: reference curves are parsed straight from the
             echemdb svgdigitizer SVGs (calibration markers + traced path), so
             no hand-made CSVs are needed. Each PDF is auto-matched to its
             echemdb entry by DOI (text/metadata) or Elsevier PII, and every
             reference figure is matched to the globally best-fitting extracted
             curve across the whole paper (vector + raster pooled), by
             flip-invariant normalised-shape Chamfer. echemdb page numbers are
             not trusted — they refer to the curators' source PDF pagination.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvdigitize.vector_extract import detect_panels, extract_color_groups
from cvdigitize.postprocess import order_curve, dedupe, resample_arclength, keep_main_components
from cvdigitize.echemdb_ref import load_entry_references


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


# ---------------------------------------------------------------------------
# echemdb ground-truth mode: benchmark a publisher PDF against the echemdb
# svgdigitizer entry for the same paper (reference curves parsed from the SVGs).
# ---------------------------------------------------------------------------
_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)
_PII = re.compile(r"S\d{16,}[0-9Xx]?")  # Elsevier PII embedded in "1-s2.0-<PII>" names


def _norm_doi(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _dois_from_pdf(pdf: str) -> set[str]:
    """Every DOI-like / Elsevier-PII token we can find in a PDF (normalised)."""
    import fitz
    out: set[str] = set()
    doc = fitz.open(pdf)
    for pg in doc:
        for d in _DOI.findall(pg.get_text()):
            out.add(_norm_doi(d))
    for v in doc.metadata.values():
        for d in _DOI.findall(str(v or "")):
            out.add(_norm_doi(d))
    doc.close()
    for pii in _PII.findall(os.path.basename(pdf)):
        out.add(_norm_doi(pii))
    return out


def _entry_doi(entry_dir: str) -> str:
    for y in glob.glob(os.path.join(entry_dir, "*.yaml")):
        with open(y, encoding="utf-8") as f:
            m = re.search(r"url:\s*https?://doi\.org/(\S+)", f.read())
            if m:
                return _norm_doi(m.group(1))
    return ""


# Fallback for scanned papers that carry no embedded/text DOI: map a filename
# substring to the echemdb entry directory name. Only consulted when DOI/PII
# matching fails, so it never overrides a real DOI match.
_FILENAME_OVERRIDES = {
    "oxygen-reduction-on-platinum-low-index": "markovic_1996_oxygen_6715",
}


def map_pdf_to_entry(pdf: str, echemdb_root: str) -> str | None:
    """Find the echemdb entry directory whose DOI a PDF references.

    Matches on DOI (from text/metadata) or, for Elsevier scans with no text DOI,
    on the PII token embedded in both the filename and the DOI. Substring match
    both ways so an Elsevier PII ("s0022072879800224") aligns with the DOI that
    contains it. Falls back to an explicit filename map for scans with no DOI at
    all.
    """
    pdf_ids = _dois_from_pdf(pdf)
    for entry in sorted(glob.glob(os.path.join(echemdb_root, "*"))):
        if not os.path.isdir(entry):
            continue
        edoi = _entry_doi(entry)
        if not edoi:
            continue
        for pid in pdf_ids:
            if len(pid) >= 8 and (pid in edoi or edoi in pid):
                return entry
    base = os.path.basename(pdf).lower()
    for frag, entry_name in _FILENAME_OVERRIDES.items():
        if frag in base:
            cand = os.path.join(echemdb_root, entry_name)
            if os.path.isdir(cand):
                return cand
    return None


def _flip_y(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    out[:, 1] = -out[:, 1]
    return out


def chamfer_oriented(ref_norm: np.ndarray, curve: np.ndarray) -> float:
    """Normalised-shape Chamfer, invariant to a vertical flip.

    Extracted curves live in y-down pixel/PDF space; echemdb references are in
    y-up real units. We don't calibrate the extraction here, so we compare the
    curve and its vertical mirror and keep the better — matching the *same*
    physical curve regardless of coordinate convention.
    """
    a = _norm(curve)
    b = _norm(_flip_y(curve))
    return min(chamfer(ref_norm, a), chamfer(ref_norm, b))


def raster_curves(pdf: str, page: int) -> list[np.ndarray]:
    """All curves the raster pipeline finds on a page, resampled like the
    vector path so the two pools are directly comparable.

    Tries frame-based multi-panel extraction first; where a region yields no
    framed panel (classic crossing-axis figures with no box), falls back to the
    frameless dominant-curve extractor."""
    from cvdigitize.raster_extract import (find_image_regions, extract_all_panel_curves,
                                           render_region, extract_frameless_curve)
    out: list[np.ndarray] = []
    try:
        regions = find_image_regions(pdf, page)
    except Exception:
        regions = []
    for region in regions:
        got = False
        try:
            results = extract_all_panel_curves(pdf, page, region.bbox)
        except Exception:
            results = []
        for res in results:
            poly = res.get("polyline_px")
            if poly is None or len(poly) < 20:
                continue
            loop = dedupe(order_curve(keep_main_components([np.asarray(poly, float)])))
            if len(loop) >= 20:
                out.append(resample_arclength(loop, n=600))
                got = True
        if not got:
            # No framed panel here -> frameless dominant-curve fallback. Reliable
            # for single-plot classic figures (bare crossing axes); we skip it
            # when a frame was found, so multi-panel composites (which frameless
            # would smear into one path) stay with the framed pipeline.
            try:
                img = render_region(pdf, page, region.bbox, zoom=4.0)
                loop = extract_frameless_curve(img)
            except Exception:
                loop = np.empty((0, 2))
            if len(loop) >= 20:
                out.append(resample_arclength(loop, n=600))
    return out


def all_paper_curves(pdf: str) -> list[dict]:
    """Pool every extractable curve in a paper: vector figure pages first, then
    raster figure pages (only where vector found nothing). Each item records the
    source page and branch so a match can be traced back."""
    from cvdigitize.ingest import classify_pdf
    from cvdigitize.raster_extract import find_image_regions
    infos = classify_pdf(pdf)
    pool: list[dict] = []
    vec_pages = sorted({pi.number for pi in infos if pi.n_curve_items > 20})
    vec_hit_pages = set()
    for pg in vec_pages:
        try:
            cs = extracted_curves(pdf, pg)
        except Exception:
            cs = []
        if cs:
            vec_hit_pages.add(pg)
        pool += [{"xy": c, "page": pg, "branch": "vector"} for c in cs]
    # Any page carrying an embedded figure image is a raster candidate — not
    # only those classified "raster" (a page can hold a figure image yet read as
    # "sparse" when the image doesn't dominate the page area, as in older scans).
    ras_pages = []
    for pi in infos:
        if pi.number in vec_hit_pages:
            continue
        try:
            has_img = bool(find_image_regions(pdf, pi.number))
        except Exception:
            has_img = False
        if pi.kind == "raster" or has_img:
            ras_pages.append(pi.number)
    for pg in ras_pages:
        pool += [{"xy": c, "page": pg, "branch": "raster"} for c in raster_curves(pdf, pg)]
    return pool


def benchmark_echemdb(pdf: str, entry_dir: str) -> dict:
    """Compare curves extracted from ``pdf`` to the echemdb references in
    ``entry_dir`` (parsed from that paper's svgdigitizer SVGs).

    Reference figures are matched to the *globally* best-fitting extracted curve
    across the whole paper (vector + raster pooled), by flip-invariant
    normalised-shape Chamfer. We do not rely on echemdb's page numbering — it
    refers to the curators' source PDF, which need not paginate like the
    publisher PDF in hand.
    """
    refs = load_entry_references(entry_dir)
    if not refs:
        return {"pdf": os.path.basename(pdf), "entry": os.path.basename(entry_dir),
                "error": "no references parsed", "results": []}
    pool = all_paper_curves(pdf)
    if not pool:
        return {"pdf": os.path.basename(pdf), "entry": os.path.basename(entry_dir),
                "error": "no curves extracted from PDF", "n_reference": len(refs),
                "n_matched": 0, "results": [
                    {"reference": r.name, "chamfer": None, "page": None, "branch": None}
                    for r in refs]}

    rows = []
    for r in refs:
        rn = _norm(r.xy)
        dists = [chamfer_oriented(rn, c["xy"]) for c in pool]
        best = int(np.argmin(dists))
        rows.append({"reference": r.name, "chamfer": round(float(dists[best]), 5),
                     "page": pool[best]["page"], "branch": pool[best]["branch"]})
    ch = [r["chamfer"] for r in rows if r["chamfer"] is not None]
    return {"pdf": os.path.basename(pdf), "entry": os.path.basename(entry_dir),
            "n_reference": len(refs), "n_matched": len(ch), "n_pool": len(pool),
            "mean_chamfer": round(float(np.mean(ch)), 5) if ch else None,
            "max_chamfer": round(float(np.max(ch)), 5) if ch else None,
            "results": sorted(rows, key=lambda r: (r["chamfer"] is None, r["chamfer"] or 0))}


def _print_echemdb_report(rep: dict):
    print(f"\n=== {rep['pdf']}  ->  {rep.get('entry','?')} ===")
    if rep.get("error"):
        print(f"  {rep['error']}")
        return
    print(f"  {rep['n_matched']}/{rep['n_reference']} references matched "
          f"(pool of {rep.get('n_pool', 0)} extracted curves)")
    print(f"  {'reference':44s} {'pg':>3} {'branch':>7} {'Chamfer':>9}  grade")
    for r in rep["results"]:
        if r["chamfer"] is None:
            print(f"  {r['reference'][:44]:44s} {'-':>3} {'-':>7} {'  --   ':>9}  no curve extracted")
            continue
        grade = ("excellent" if r["chamfer"] < 0.01 else
                 "good" if r["chamfer"] < 0.03 else
                 "fair" if r["chamfer"] < 0.08 else "poor")
        print(f"  {r['reference'][:44]:44s} {r['page']:>3} {r['branch']:>7} "
              f"{r['chamfer']:>9.4f}  {grade}")
    print(f"  --> mean {rep['mean_chamfer']}  max {rep['max_chamfer']}")


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
    ap = argparse.ArgumentParser(description="Accuracy benchmark vs reference curves.")
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("ref_dir", nargs="?")
    ap.add_argument("--page", type=int, default=None)
    ap.add_argument("--suite", help="dir with <stem>.pdf + <stem>_ref/ pairs")
    ap.add_argument("--echemdb", nargs=2, metavar=("PDF_DIR", "ECHEMDB_ROOT"),
                    help="benchmark every PDF in PDF_DIR against its echemdb "
                         "svgdigitizer entry (auto-matched by DOI)")
    ap.add_argument("--echemdb-one", nargs=2, metavar=("PDF", "ENTRY_DIR"),
                    help="benchmark one PDF against one echemdb entry directory")
    args = ap.parse_args(argv)

    # ---- echemdb ground-truth mode -----------------------------------------
    if args.echemdb or args.echemdb_one:
        ereports = []
        if args.echemdb_one:
            pdf, entry = args.echemdb_one
            ereports.append(benchmark_echemdb(pdf, entry))
        else:
            pdf_dir, root = args.echemdb
            for pdf in sorted(glob.glob(os.path.join(pdf_dir, "*.pdf"))):
                entry = map_pdf_to_entry(pdf, root)
                if entry is None:
                    print(f"\n=== {os.path.basename(pdf)} ===\n  no echemdb entry matched (DOI not found)")
                    continue
                ereports.append(benchmark_echemdb(pdf, entry))
        for rep in ereports:
            _print_echemdb_report(rep)
        all_ch = [r["chamfer"] for rep in ereports
                  for r in rep.get("results", []) if r.get("chamfer") is not None]
        n_ref = sum(rep.get("n_reference", 0) for rep in ereports)
        if all_ch:
            print(f"\nOVERALL: {len(all_ch)}/{n_ref} references matched across "
                  f"{len(ereports)} paper(s) · mean Chamfer {np.mean(all_ch):.4f} · "
                  f"{100*np.mean([c < 0.03 for c in all_ch]):.0f}% good-or-better (<0.03) · "
                  f"{100*np.mean([c < 0.01 for c in all_ch]):.0f}% excellent (<0.01)")
        return

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
        ap.error("give <pdf> <ref_dir>, --suite <dir>, or --echemdb <pdf_dir> <echemdb_root>")

    for rep in reports:
        _print_report(rep)
    all_ch = [r["chamfer"] for rep in reports for r in rep.get("results", [])]
    if all_ch:
        print(f"\nOVERALL: {len(all_ch)} curves across {len(reports)} paper(s) · "
              f"mean Chamfer {np.mean(all_ch):.4f} · "
              f"{100*np.mean([c < 0.03 for c in all_ch]):.0f}% good-or-better (<0.03)")


if __name__ == "__main__":
    main()
