"""M2 end-to-end: multi-panel raster figure -> per-panel curves, calibrated.

Runs on the arXiv page-6 composite figure, auto-detects all plot panels,
picks panel (c) (the CV), calibrates using auto-detected tick pixel positions
+ known tick label values (read from the printed figure), and overlays the
result on the original rendered figure for visual verification.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvdigitize.raster_extract import find_image_regions, extract_all_panel_curves
from cvdigitize.calibrate import calibration_from_anchors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "data", "in", "arxiv_photoec_transistor.pdf")
OUT = os.path.join(ROOT, "data", "out", "m2"); os.makedirs(OUT, exist_ok=True)
PAGE = 5  # 0-indexed page 6

regions = find_image_regions(PDF, PAGE)
region = regions[0]
results = extract_all_panel_curves(PDF, PAGE, region.bbox)
print(f"Detected {len(results)} panel(s) on page {PAGE+1}:")
for i, r in enumerate(results):
    print(f"  [{i}] frame_px={r['frame_px']} points={len(r['polyline'])} "
          f"x_ticks={r['ticks']['x_ticks']} y_ticks={r['ticks']['y_ticks']}")

# panel c = the CV: identify by its frame position (matches earlier finding)
panel = next(r for r in results if 500 < r["frame_px"][1] < 600)
xt, yt = panel["ticks"]["x_ticks"], panel["ticks"]["y_ticks"]
print(f"\nUsing panel with frame {panel['frame_px']}")
print(f"x ticks (px, relative to region): {xt} -> labels -0.8 .. +0.2 V (step 0.2)")
print(f"y ticks (px, relative to region): {yt} -> labels +25 .. -50 uA (step -25, top to bottom)")

# calibration: first/last detected tick <-> known printed label values
calib = calibration_from_anchors(
    x_anchor1=(xt[0], -0.8), x_anchor2=(xt[-1], 0.2),
    y_anchor1=(yt[0], 25.0), y_anchor2=(yt[-1], -50.0),
    x_unit="V vs Ag/AgCl", y_unit="uA",
)
os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
calib.save(os.path.join(ROOT, "configs", "arxiv_photoec_p6_panelc.calib.json"))

poly_px = panel["polyline_px"]
data = calib.apply(poly_px)
print(f"\nCalibrated curve: {len(data)} points, "
      f"E range [{data[:,0].min():.2f}, {data[:,0].max():.2f}] V, "
      f"j range [{data[:,1].min():.1f}, {data[:,1].max():.1f}] uA")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
axes[0].imshow(panel["image"]); axes[0].set_title("full composite region (all panels)")
x0, y0, x1, y1 = panel["frame_px"]
import matplotlib.patches as patches
axes[0].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                     edgecolor="lime", linewidth=2))

axes[1].imshow(panel["image"][y0:y1, x0:x1])
axes[1].plot(poly_px[:, 0] - x0, poly_px[:, 1] - y0, "-", color="red", lw=1)
axes[1].set_title("panel (c) crop + extracted skeleton (red)")

axes[2].plot(data[:, 0], data[:, 1], "-", color="black", lw=1.2)
axes[2].set_xlabel("E / V vs Ag/AgCl"); axes[2].set_ylabel("j / uA")
axes[2].set_title("calibrated result (compare to original figure)")
axes[2].grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(OUT, "m2_full_pipeline.png")
fig.savefig(out, dpi=110)
print(f"\nWrote {out}")
