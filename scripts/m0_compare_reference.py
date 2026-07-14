"""M0 verification: prove the auto-extracted panel-(a) curves reproduce
Vladislav's hand-digitized reference CSVs.

Uses panel-first extraction (localise to the 2x2 grid, then group by colour
within panel a). Each extracted curve is overlaid on its matching reference
CSV; both are min-max normalised so the check is calibration-independent. A
symmetric nearest-neighbour (Chamfer) distance in normalised units gives a
quantitative match score (smaller = better; <0.02 is an excellent overlay).

Run:
    .venv\\Scripts\\python.exe scripts\\m0_compare_reference.py
Output: data/out/m0/compare_reference.png  (+ printed scores)
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvdigitize.vector_extract import extract_panels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "data", "in", "rizo_2025_analysis_351.pdf")
REF = os.path.join(ROOT, "data", "reference")
OUT = os.path.join(ROOT, "data", "out", "m0")
os.makedirs(OUT, exist_ok=True)

# Panel (a) exact stroke RGB -> (reference csv, plot colour, sample label).
# NB panel (a)'s Pt(331) is drawn in (0.502,0,1.0), a bluer purple than the
# violet used in the other panels — see the M0 investigation.
PANEL_A = [
    ((0.0, 0.0, 0.0),     "rizo_2025_analysis_351_black_111.csv",   "#000000", "Pt(111)"),
    ((1.0, 0.0, 0.0),     "rizo_2025_analysis_351_red_151514.csv",  "#e0002b", "Pt(15,15,14)"),
    ((0.0, 0.0, 1.0),     "rizo_2025_analysis_351_blue_776.csv",    "#0000ff", "Pt(776)"),
    ((1.0, 0.0, 1.0),     "rizo_2025_analysis_351_pink_554.csv",    "#ff00ff", "Pt(554)"),
    ((1.0, 0.502, 0.0),   "rizo_2025_analysis_351_orange_553.csv",  "#ff8000", "Pt(553)"),
    ((0.502, 0.502, 0.0), "rizo_2025_analysis_351_khaki_221.csv",   "#808000", "Pt(221)"),
    ((0.502, 0.0, 1.0),   "rizo_2025_analysis_351_violet_331.csv",  "#8000ff", "Pt(331)"),
]


def load_ref_csv(path):
    E, j = [], []
    with open(path, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) >= 2:
                E.append(float(row[0])); j.append(float(row[1]))
    return np.array(E), np.array(j)


def normalize_xy(x, y):
    def n(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if hi > lo else a * 0
    return n(x), n(y)


def chamfer(a_xy, b_xy):
    """Symmetric mean nearest-neighbour distance between two point clouds."""
    ta, tb = cKDTree(a_xy), cKDTree(b_xy)
    da, _ = tb.query(a_xy)
    db, _ = ta.query(b_xy)
    return float((da.mean() + db.mean()) / 2)


def main():
    panels = extract_panels(PDF, 1, nx=2, ny=2)
    panel_a = {c.rgb: c for c in panels[(0, 0)]}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    scores = []

    for ax, (rgb, ref_name, col, label) in zip(axes, PANEL_A):
        E, j = load_ref_csv(os.path.join(REF, ref_name))
        rEx, rjy = normalize_xy(E, j)
        ax.plot(rEx, rjy, color="0.65", lw=3.2, alpha=0.8, label="reference (manual)", zorder=1)

        cg = panel_a.get(rgb)
        if cg is not None:
            allx = np.concatenate([p[:, 0] for p in cg.polylines])
            ally = np.concatenate([p[:, 1] for p in cg.polylines])
            xlo, xhi, ylo, yhi = allx.min(), allx.max(), ally.min(), ally.max()
            ex_pts = []
            for pl in cg.polylines:
                xn = (pl[:, 0] - xlo) / (xhi - xlo)
                yn = 1.0 - (pl[:, 1] - ylo) / (yhi - ylo)  # flip y: up = higher current
                ax.plot(xn, yn, color=col, lw=0.8, zorder=2)
                ex_pts.append(np.column_stack([xn, yn]))
            ex_pts = np.vstack(ex_pts)
            d = chamfer(np.column_stack([rEx, rjy]), ex_pts)
            scores.append((label, d))
            ax.plot([], [], color=col, lw=1.5, label=f"extracted (auto)")
            ax.text(0.5, -0.02, f"Chamfer = {d:.4f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9,
                    color=("green" if d < 0.02 else "darkorange"))
        else:
            ax.text(0.5, 0.5, "no curve in panel (a)", transform=ax.transAxes,
                    ha="center", color="red")

        ax.set_title(f"{label}   [{ref_name.split('_')[-1].replace('.csv','')}]", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper center", fontsize=7, framealpha=0.85)

    axes[-1].axis("off")
    fig.suptitle("M0 verification: auto-extracted panel (a) vs manual reference "
                 "(normalised; Chamfer distance in normalised units)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT, "compare_reference.png")
    fig.savefig(out, dpi=115)
    plt.close(fig)

    print("Per-curve shape-match (Chamfer distance, normalised units):")
    for label, d in scores:
        flag = "OK " if d < 0.02 else "!! "
        print(f"  {flag}{label:14s} {d:.4f}")
    if scores:
        print(f"  mean = {np.mean([d for _, d in scores]):.4f}   "
              f"max = {np.max([d for _, d in scores]):.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
