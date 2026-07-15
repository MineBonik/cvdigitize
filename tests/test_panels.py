"""Tests for automatic frame-based panel detection."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvdigitize.vector_extract import detect_panels


def _two_panel_pdf(path):
    """Two side-by-side axes, each with a distinct-coloured closed loop."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8, 3))
    t = np.linspace(0, 2 * np.pi, 400)
    axl.plot(0.3 + 0.25 * np.cos(t), 5 * np.sin(t), color="red")
    axl.set_xlim(-0.2, 0.8); axl.set_ylim(-8, 8)
    axr.plot(0.5 + 0.3 * np.cos(t), 2 * np.sin(t), color="blue")
    axr.set_xlim(0, 1); axr.set_ylim(-3, 3)
    for a in (axl, axr):
        a.set_xlabel("E / V"); a.set_ylabel("j")
    fig.savefig(path); plt.close(fig)


def test_detect_two_panels(tmp_path):
    pdf = str(tmp_path / "two.pdf")
    _two_panel_pdf(pdf)
    panels = detect_panels(pdf, 0, min_points=40, min_curve_points=40)
    assert len(panels) == 2
    # panels are lettered in reading order, left-to-right here
    assert [p.label for p in panels] == ["a", "b"]
    # left panel holds the red loop, right the blue one
    left_colors = {c.name for c in panels[0].curves}
    right_colors = {c.name for c in panels[1].curves}
    assert "red" in left_colors
    assert "blue" in right_colors
    # each panel's curves stay within its own frame
    for p in panels:
        fx0, fy0, fx1, fy1 = p.frame_pdf
        for c in p.curves:
            for pl in c.polylines:
                cx, cy = pl[:, 0].mean(), pl[:, 1].mean()
                assert fx0 <= cx <= fx1 and fy0 <= cy <= fy1


def test_detect_single_panel(tmp_path):
    pdf = str(tmp_path / "one.pdf")
    fig, ax = plt.subplots(figsize=(4, 3))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(0.3 + 0.25 * np.cos(t), np.sin(t), color="green")
    ax.set_xlabel("E"); ax.set_ylabel("j")
    fig.savefig(pdf); plt.close(fig)
    panels = detect_panels(pdf, 0, min_points=40, min_curve_points=40)
    assert len(panels) == 1
