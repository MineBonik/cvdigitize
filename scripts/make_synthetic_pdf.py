"""Generate a synthetic *vector* CV figure PDF with known ground-truth curves.

This is a controlled generality test: a different figure layout, colours and
axis ranges from the rizo paper, with exact ground truth so we can measure the
full pipeline's accuracy (extract -> order -> calibrate -> resample).

Outputs:
    data/in/synthetic_cv.pdf
    data/synthetic_truth/<name>.csv   (ground-truth E,j)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# keep full point resolution in the vector PDF (no path simplification)
matplotlib.rcParams["path.simplify"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "data", "in")
TRUTH = os.path.join(ROOT, "data", "synthetic_truth")
os.makedirs(IN, exist_ok=True); os.makedirs(TRUTH, exist_ok=True)

# axis ranges (deliberately different from rizo)
E_MIN, E_MAX = -0.2, 0.8
J_MIN, J_MAX = -2.5, 2.5


def synth_cv(cap, peak_amp, peak_E, sep=0.08, width=0.03, n=500, tilt=0.0):
    """A duck-shaped CV loop: capacitive box + anodic/cathodic Gaussian peaks."""
    Ef = np.linspace(E_MIN, E_MAX, n)
    jf = cap + tilt * (Ef - E_MIN) + peak_amp * np.exp(-((Ef - (peak_E + sep / 2)) / width) ** 2)
    Er = np.linspace(E_MAX, E_MIN, n)
    jr = -cap + tilt * (Er - E_MIN) - peak_amp * np.exp(-((Er - (peak_E - sep / 2)) / width) ** 2)
    E = np.concatenate([Ef, Er]); j = np.concatenate([jf, jr])
    return np.column_stack([E, j])


CURVES = {
    "sample_A": (dict(cap=0.6, peak_amp=1.6, peak_E=0.25, tilt=0.3), "#d00000"),
    "sample_B": (dict(cap=0.4, peak_amp=1.0, peak_E=0.40, tilt=0.2), "#0050d0"),
    "sample_C": (dict(cap=0.8, peak_amp=2.0, peak_E=0.15, tilt=0.5), "#00a000"),
}


def main():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (params, color) in CURVES.items():
        xy = synth_cv(**params)
        np.savetxt(os.path.join(TRUTH, name + ".csv"), xy, delimiter=",",
                   header="E,j", comments="")
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.2, label=name)

    ax.set_xlim(E_MIN, E_MAX); ax.set_ylim(J_MIN, J_MAX)
    ax.set_xlabel("E / V vs Ref"); ax.set_ylabel("j / mA cm$^{-2}$")
    ax.set_title("Synthetic cyclic voltammograms (vector)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    pdf = os.path.join(IN, "synthetic_cv.pdf")
    fig.savefig(pdf)  # vector PDF
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote ground truth for {list(CURVES)} to {TRUTH}")
    print(f"Axis ranges: E[{E_MIN},{E_MAX}] V, j[{J_MIN},{J_MAX}] mA/cm2")


if __name__ == "__main__":
    main()
