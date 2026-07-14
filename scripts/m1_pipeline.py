"""Full M1 vector pipeline on rizo panel (a):
  extract -> order loop -> calibrate (fit from reference) -> resample -> package,
then compare the cleaned, calibrated curve to the manual reference in REAL units
and quantify the spacing improvement (Albert's "data is not raw" issue).

Run: .venv\\Scripts\\python.exe scripts\\m1_pipeline.py
"""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvdigitize.vector_extract import extract_panels
from cvdigitize.postprocess import order_curve, dedupe, resample_arclength
from cvdigitize.calibrate import calibration_from_bboxes
from cvdigitize.package import write_datapackage, CurveMeta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "data", "in", "rizo_2025_analysis_351.pdf")
REF = os.path.join(ROOT, "data", "reference")
OUT = os.path.join(ROOT, "data", "out", "m1")
DP = os.path.join(OUT, "datapackages")
os.makedirs(DP, exist_ok=True)

# panel(a) rgb -> (reference csv, sample label, plot colour)
MAP = [
    ((0.0, 0.0, 0.0),     "black_111",   "Pt(111)",      "#000000"),
    ((1.0, 0.0, 0.0),     "red_151514",  "Pt(15,15,14)", "#e0002b"),
    ((0.0, 0.0, 1.0),     "blue_776",    "Pt(776)",      "#0000ff"),
    ((1.0, 0.0, 1.0),     "pink_554",    "Pt(554)",      "#ff00ff"),
    ((1.0, 0.502, 0.0),   "orange_553",  "Pt(553)",      "#ff8000"),
    ((0.502, 0.502, 0.0), "khaki_221",   "Pt(221)",      "#808000"),
    ((0.502, 0.0, 1.0),   "violet_331",  "Pt(331)",      "#8000ff"),
]
PREFIX = "rizo_2025_analysis_351_"


def load_ref(tag):
    E, j = [], []
    with open(os.path.join(REF, PREFIX + tag + ".csv"), newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            E.append(float(row[0])); j.append(float(row[1]))
    return np.column_stack([E, j])


def main():
    panels = extract_panels(PDF, 1, nx=2, ny=2)
    by_rgb = {c.rgb: c for c in panels[(0, 0)]}

    # order every curve once
    loops = {rgb: dedupe(order_curve(by_rgb[rgb].polylines)) for rgb, *_ in MAP}
    refs = {tag: load_ref(tag) for _, tag, *_ in MAP}

    # ONE global calibration for the whole panel: pooled pixel bbox of all
    # curves <-> pooled data bbox of all references (shared axes).
    allpx = np.vstack(list(loops.values()))
    allref = np.vstack(list(refs.values()))
    pixel_bbox = (allpx[:, 0].min(), allpx[:, 1].min(), allpx[:, 0].max(), allpx[:, 1].max())
    data_bbox = (allref[:, 0].min(), allref[:, 1].min(), allref[:, 0].max(), allref[:, 1].max())
    calib = calibration_from_bboxes(pixel_bbox, data_bbox, x_unit="V vs RHE", y_unit="uA/cm2")
    os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
    calib.save(os.path.join(ROOT, "configs", "rizo_2025_analysis_351_fig1a.calib.json"))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.ravel()
    fig2, ax2 = plt.subplots(figsize=(7, 5))  # spacing histogram
    stats = []

    for ax, (rgb, tag, label, col) in zip(axes, MAP):
        ref = refs[tag]
        cal = calib.apply(loops[rgb])                 # -> (E, j)
        clean = resample_arclength(cal, n=1200)       # even, smooth

        # package as echemdb datapackage
        meta = CurveMeta(name=PREFIX + tag, figure="1a", curve=label,
                         scan_rate="50 mV/s", x_unit="V vs RHE", y_unit="uA/cm2",
                         source_pdf=PDF, method="digitized",
                         comment="auto-extracted from vector PDF; arc-length resampled")
        write_datapackage(DP, clean, meta, yaml=True)

        # overlay in real units
        ax.plot(ref[:, 0], ref[:, 1], color="0.65", lw=3, alpha=.8, label="reference")
        ax.plot(clean[:, 0], clean[:, 1], color=col, lw=0.9, label="auto (clean)")
        ax.set_title(f"{label}"); ax.set_xlabel("E / V vs RHE"); ax.set_ylabel("j / uA cm$^{-2}$")
        ax.legend(fontsize=7)

        # spacing stats: reference vs cleaned (consecutive point distances in E)
        dref = np.abs(np.diff(ref[:, 0]))
        dcln = np.abs(np.diff(clean[:, 0]))
        # arc-length spacing (true evenness metric)
        aref = np.hypot(np.diff(ref[:, 0]), np.diff(ref[:, 1]) / 100.0)
        acln = np.hypot(np.diff(clean[:, 0]), np.diff(clean[:, 1]) / 100.0)
        cv_ref = np.std(aref) / np.mean(aref)
        cv_cln = np.std(acln) / np.mean(acln)
        dup_ref = int(np.sum(dref < 1e-9))
        stats.append((label, len(ref), len(clean), cv_ref, cv_cln, dup_ref))
        if tag == "black_111":
            ax2.hist(aref, bins=60, alpha=.6, label=f"reference (CoV={cv_ref:.2f})", color="0.5")
            ax2.hist(acln, bins=60, alpha=.6, label=f"auto clean (CoV={cv_cln:.2f})", color=col)

    axes[-1].axis("off")
    fig.suptitle("M1: cleaned + calibrated auto curves vs manual reference (real units)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "m1_compare_realunits.png"), dpi=115); plt.close(fig)

    ax2.set_title("Point-spacing evenness: Pt(111)\n(lower CoV = more uniform = more 'raw'-like)")
    ax2.set_xlabel("step length between consecutive points"); ax2.set_ylabel("count")
    ax2.legend(); fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "m1_spacing_hist.png"), dpi=115); plt.close(fig2)

    print("curve            n_ref  n_clean  CoV_ref  CoV_clean  dup_x_ref")
    for label, nr, nc, cr, cc, dr in stats:
        print(f"  {label:13s} {nr:6d} {nc:8d}   {cr:6.2f}    {cc:6.2f}    {dr:6d}")
    print(f"\nWrote datapackages to {DP}")
    print(f"Wrote m1_compare_realunits.png, m1_spacing_hist.png to {OUT}")
    # show one packaged CSV head
    ex = os.path.join(DP, PREFIX + "black_111.csv")
    print(f"\nSample of {os.path.basename(ex)}:")
    with open(ex) as f:
        for i, line in enumerate(f):
            if i > 4: break
            print("   " + line.rstrip())


if __name__ == "__main__":
    main()
