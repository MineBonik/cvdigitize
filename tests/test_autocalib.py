"""Tests for text-layer auto-calibration and the assisted-tick helpers."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvdigitize.autocalib import (
    TickLabel, _fit_axis, _normalize, find_axis_label_sets, match_calibration,
)


def _mk(value, cx, cy, w=12.0, h=8.0):
    return TickLabel(value=value, cx=cx, cy=cy,
                     bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def test_normalize_unicode_minus():
    assert _normalize("−0.8") == "-0.8"
    assert _normalize("–0.5") == "-0.5"


def test_fit_axis_accepts_clean_row():
    labels = [_mk(v, 100 + i * 50, 300) for i, v in enumerate([0.0, 0.2, 0.4, 0.6])]
    s = _fit_axis(labels, "x")
    assert s is not None
    assert abs(s.slope - 0.2 / 50) < 1e-9
    assert s.residual < 1e-9


def test_fit_axis_rejects_duplicates_and_nonmonotonic():
    dup = [_mk(v, 100 + i * 50, 300) for i, v in enumerate([0.0, 0.0, 1.0, 2.0])]
    assert _fit_axis(dup, "x") is None
    # caption-style numbers scattered in x, not on a line
    scatter = [_mk(0.1, 100, 300), _mk(0.9, 120, 300), _mk(0.3, 400, 300)]
    assert _fit_axis(scatter, "x") is None


def test_fit_axis_rejects_poor_linear_fit():
    labels = [_mk(v, p, 300) for v, p in
              [(0.0, 100), (1.0, 150), (2.0, 380), (3.0, 400)]]
    assert _fit_axis(labels, "x") is None


def test_end_to_end_on_matplotlib_pdf(tmp_path):
    """A matplotlib PDF keeps live text: detection must recover the true axes."""
    pdf = str(tmp_path / "fig.pdf")
    fig, ax = plt.subplots(figsize=(4, 3))
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(0.3 + 0.25 * np.cos(t), 5 * np.sin(t), color="red")
    ax.set_xlim(-0.2, 0.8); ax.set_ylim(-8, 8)
    ax.set_xlabel("E / V"); ax.set_ylabel("j / uA")
    fig.savefig(pdf); plt.close(fig)

    sets = find_axis_label_sets(pdf, 0)
    assert {s.orientation for s in sets} == {"x", "y"}

    # curve bbox in PDF points: derive from the known data limits is awkward;
    # instead use a generous box in the middle of the page — the matcher only
    # needs the labels to sit below/left of it.
    xs = [s for s in sets if s.orientation == "x"][0]
    ys = [s for s in sets if s.orientation == "y"][0]
    lo, hi = xs.pixel_span
    ylo, yhi = ys.pixel_span
    bbox = (lo, ylo, hi, yhi)
    cal = match_calibration(sets, bbox)
    assert cal is not None
    # x calibration must map the label row's own pixel positions back to values
    for lab in xs.labels:
        E, _ = cal.to_data(np.array([lab.cx]), np.array([0.0]))
        assert abs(E[0] - lab.value) < 0.02
    for lab in ys.labels:
        _, j = cal.to_data(np.array([0.0]), np.array([lab.cy]))
        assert abs(j[0] - lab.value) < 0.2


def test_regular_subset_filters_stray():
    from cvdigitize.raster_extract import _regular_subset
    ticks = [100, 150, 200, 250, 300]
    with_stray = sorted(ticks + [137, 262])
    out = _regular_subset(with_stray)
    assert out == ticks
    # fewer than 4: untouched
    assert _regular_subset([10, 97]) == [10, 97]
