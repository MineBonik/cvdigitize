"""End-to-end accuracy test on the synthetic vector CV PDF against ground truth.

Pipeline: extract -> order loop -> global bbox calibration (pooled extracted
pixels <-> pooled true data extent) -> arc-length resample. Reports Chamfer
distance to ground truth, normalised by each axis range.
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvdigitize.vector_extract import extract_color_groups
from cvdigitize.postprocess import order_curve, dedupe, resample_arclength, keep_main_components
from cvdigitize.calibrate import calibration_from_bboxes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "data", "in", "synthetic_cv.pdf")
TRUTH = os.path.join(ROOT, "data", "synthetic_truth")
OUT = os.path.join(ROOT, "data", "out", "synthetic"); os.makedirs(OUT, exist_ok=True)

# synthetic stroke rgb -> (truth name, plot colour)
MAP = [
    ((0.816, 0.0, 0.0),   "sample_A", "#d00000"),
    ((0.0, 0.314, 0.816), "sample_B", "#0050d0"),
    ((0.0, 0.627, 0.0),   "sample_C", "#00a000"),
]


def load_truth(name):
    return np.loadtxt(os.path.join(TRUTH, name + ".csv"), delimiter=",", skiprows=1)


def chamfer(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return float((tb.query(a)[0].mean() + ta.query(b)[0].mean()) / 2)


def main():
    groups = {g.rgb: g for g in extract_color_groups(PDF, 0, min_points=50)}
    truths = {name: load_truth(name) for _, name, _ in MAP}

    loops = {rgb: dedupe(order_curve(keep_main_components(groups[rgb].polylines)))
             for rgb, *_ in MAP}
    allpx = np.vstack(list(loops.values()))
    alltruth = np.vstack(list(truths.values()))
    calib = calibration_from_bboxes(
        (allpx[:, 0].min(), allpx[:, 1].min(), allpx[:, 0].max(), allpx[:, 1].max()),
        (alltruth[:, 0].min(), alltruth[:, 1].min(), alltruth[:, 0].max(), alltruth[:, 1].max()),
    )

    Erange = np.ptp(alltruth[:, 0]); jrange = np.ptp(alltruth[:, 1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2)); scores = []
    for ax, (rgb, name, col) in zip(axes, MAP):
        truth = truths[name]
        clean = resample_arclength(calib.apply(loops[rgb]), n=1000)
        # normalise both axes before Chamfer so it's a fair % error
        def nrm(a): return np.column_stack([a[:, 0] / Erange, a[:, 1] / jrange])
        d = chamfer(nrm(truth), nrm(clean))
        scores.append((name, d, len(clean)))
        ax.plot(truth[:, 0], truth[:, 1], color="0.6", lw=3, alpha=.8, label="ground truth")
        ax.plot(clean[:, 0], clean[:, 1], color=col, lw=1, label="auto")
        ax.set_title(f"{name}  (Chamfer={d:.4f})"); ax.legend(fontsize=8)
        ax.set_xlabel("E / V"); ax.set_ylabel("j / mA cm$^{-2}$")
    fig.suptitle("Synthetic PDF: auto pipeline vs ground truth", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "synthetic_vs_truth.png"), dpi=115); plt.close(fig)

    print("curve      Chamfer(norm)  n_points")
    for name, d, n in scores:
        print(f"  {name:9s}  {d:.5f}      {n}")
    print(f"  mean Chamfer = {np.mean([d for _,d,_ in scores]):.5f}")
    print(f"Wrote {os.path.join(OUT, 'synthetic_vs_truth.png')}")


if __name__ == "__main__":
    main()
