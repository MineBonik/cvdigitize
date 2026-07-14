"""Test M2 raster extraction on the arXiv page 6 (index 5) CV panel."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvdigitize.raster_extract import find_image_regions, extract_raster_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "data", "in", "arxiv_photoec_transistor.pdf")
OUT = os.path.join(ROOT, "data", "out", "m2"); os.makedirs(OUT, exist_ok=True)

regions = find_image_regions(PDF, 5)
print(f"Found {len(regions)} image region(s) on page 6:")
for r in regions:
    print(f"  xref={r.xref} bbox={tuple(round(v,1) for v in r.bbox)}")

region = regions[0]
result = extract_raster_curve(PDF, 5, region.bbox, zoom=4.0)
poly = result["polyline"]
print(f"\nExtracted polyline: {len(poly)} points")
print(f"frame_px (within region): {result['frame_px']}")

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
axes[0].imshow(result["image"]); axes[0].set_title("rendered region (whole figure block)")
axes[1].imshow(result["mask"], cmap="gray"); axes[1].set_title("curve mask (largest component)")
axes[2].imshow(result["skeleton"], cmap="gray"); axes[2].set_title("skeleton")
axes[3].plot(poly[:, 0], poly[:, 1], "-", lw=1, color="black")
axes[3].invert_yaxis(); axes[3].set_aspect("equal", "datalim")
axes[3].set_title(f"ordered polyline ({len(poly)} pts, PDF-pt coords)")
fig.tight_layout()
out = os.path.join(OUT, "raster_extract_check.png")
fig.savefig(out, dpi=110)
print(f"\nWrote {out}")
