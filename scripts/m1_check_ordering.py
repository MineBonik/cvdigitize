"""Validate loop ordering + resampling on real extracted curves."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvdigitize.vector_extract import extract_panels
from cvdigitize.postprocess import order_curve, dedupe, split_branches, resample_arclength

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "data", "in", "rizo_2025_analysis_351.pdf")
OUT = os.path.join(ROOT, "data", "out", "m1"); os.makedirs(OUT, exist_ok=True)

panels = extract_panels(PDF, 1, nx=2, ny=2)
curves = {c.name: c for c in panels[(0, 0)]}
targets = ["black", "red", "c_8000ff"]  # incl. the tricky violet-a

fig, axes = plt.subplots(len(targets), 3, figsize=(15, 4 * len(targets)))
for row, name in enumerate(targets):
    cg = curves[name]
    raw = np.vstack(cg.polylines)
    loop = dedupe(order_curve(cg.polylines))
    fwd, rev = split_branches(loop)
    smooth = resample_arclength(loop, n=800)

    # col 0: raw subpaths (each a different shade) -> shows shuffled segments
    ax = axes[row, 0]
    for k, pl in enumerate(cg.polylines):
        ax.plot(pl[:, 0], pl[:, 1], lw=0.6)
    ax.invert_yaxis(); ax.set_title(f"{name}: raw sub-paths ({len(cg.polylines)})")
    ax.set_aspect("equal", "datalim")

    # col 1: ordered loop coloured by traversal index -> should flow smoothly
    ax = axes[row, 1]
    t = np.linspace(0, 1, len(loop))
    ax.scatter(loop[:, 0], loop[:, 1], c=t, cmap="viridis", s=2)
    ax.invert_yaxis(); ax.set_title(f"ordered loop ({len(loop)} pts, colour=order)")
    ax.set_aspect("equal", "datalim")

    # col 2: forward/reverse branches + smooth resample
    ax = axes[row, 2]
    ax.plot(fwd[:, 0], fwd[:, 1], color="tab:red", lw=1, label="anodic")
    ax.plot(rev[:, 0], rev[:, 1], color="tab:blue", lw=1, label="cathodic")
    ax.invert_yaxis(); ax.set_title(f"branches (fwd {len(fwd)}, rev {len(rev)})")
    ax.legend(fontsize=8); ax.set_aspect("equal", "datalim")

fig.suptitle("M1 ordering check: shuffled sub-paths -> single ordered CV loop", fontsize=13)
fig.tight_layout()
out = os.path.join(OUT, "ordering_check.png")
fig.savefig(out, dpi=110); plt.close(fig)

# quantitative: max jump between consecutive ordered points (should be small
# relative to curve size, i.e. no teleporting across the plot)
for name in targets:
    cg = curves[name]
    loop = dedupe(order_curve(cg.polylines))
    steps = np.hypot(np.diff(loop[:, 0]), np.diff(loop[:, 1]))
    diag = np.hypot(*(loop.max(0) - loop.min(0)))
    print(f"{name:9s}: pts={len(loop):5d}  max step={steps.max():7.2f}  "
          f"median step={np.median(steps):5.2f}  (plot diag={diag:6.1f}, "
          f"max step = {100*steps.max()/diag:4.1f}% of diag)")
print(f"Wrote {out}")
