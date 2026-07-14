"""M0 demo: auto-split the CV curves from a vector PDF by stroke colour, and
prove the split is correct by overlaying the extracted curves on a render of
the original figure page.

Run:
    .venv\\Scripts\\python.exe scripts\\m0_demo.py

Outputs (data/out/m0/):
    overlay.png            extracted curves drawn over the original figure
    separated_grid.png     each colour on its own axes
    summary.txt            per-colour point counts
"""
from __future__ import annotations

import os
import sys

import fitz
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvdigitize.vector_extract import extract_color_groups, find_figure_pages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "data", "in", "rizo_2025_analysis_351.pdf")
OUT = os.path.join(ROOT, "data", "out", "m0")
os.makedirs(OUT, exist_ok=True)

# Matplotlib colour for each named curve (magenta for "pink" to match RGB).
PLOT_COLORS = {
    "black": "#000000", "blue": "#0000ff", "khaki": "#808000",
    "orange": "#ff8000", "pink": "#ff00ff", "red": "#ff0000",
    "violet": "#800080",
}


def render_page_png(page, zoom=3.0):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    return img[:, :, :3], zoom


def main():
    pages = find_figure_pages(PDF)
    print(f"Candidate figure pages: {pages}")
    page_no = pages[0] if pages else 1

    groups = extract_color_groups(PDF, page_no, min_points=50)
    # Keep only the 7 known CV colours (drops tiny 8th artefact colour).
    known = {"black", "blue", "khaki", "orange", "pink", "red", "violet"}
    groups = [g for g in groups if g.name in known]

    lines = [f"Figure page (0-indexed): {page_no}", f"Curves found: {len(groups)}", ""]
    for g in groups:
        x0, y0, x1, y1 = g.bbox
        lines.append(f"  {g.name:8s} rgb={g.rgb}  points={g.n_points:5d}  "
                     f"subpaths={len(g.polylines):3d}  bbox=({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # ---- Overlay on the original page render -----------------------------
    doc = fitz.open(PDF)
    page = doc[page_no]
    img, zoom = render_page_png(page, zoom=3.0)

    fig, ax = plt.subplots(figsize=(12, 14))
    ax.imshow(img)
    for g in groups:
        c = PLOT_COLORS.get(g.name, "#333333")
        for i, pl in enumerate(g.polylines):
            ax.plot(pl[:, 0] * zoom, pl[:, 1] * zoom, color=c, lw=1.2,
                    label=g.name if i == 0 else None)
    ax.set_title(f"M0: auto-extracted curves overlaid on original (page {page_no})")
    ax.legend(loc="upper right", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overlay.png"), dpi=110)
    plt.close(fig)

    # ---- Separated grid: each colour alone (PDF coords, y flipped) --------
    n = len(groups)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, g in zip(axes, groups):
        c = PLOT_COLORS.get(g.name, "#333333")
        for pl in g.polylines:
            ax.plot(pl[:, 0], pl[:, 1], color=c, lw=0.9)
        ax.set_title(f"{g.name}  ({g.n_points} pts)")
        ax.invert_yaxis()  # PDF y grows downward
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("M0: each CV curve separated automatically by stroke colour", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "separated_grid.png"), dpi=110)
    plt.close(fig)
    doc.close()

    print(f"\nWrote overlay.png, separated_grid.png, summary.txt to {OUT}")


if __name__ == "__main__":
    main()
