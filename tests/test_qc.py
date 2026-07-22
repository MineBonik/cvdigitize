"""Tests for the eyeball-QC artifacts (transparent overlay + check.html handoff)."""
import os

import cv2
import numpy as np

from cvdigitize.qc import write_qc, _qc_hex


def _panel_and_curve():
    img = np.full((120, 200, 3), 255, np.uint8)
    xy = np.column_stack([np.arange(20, 180), 60 + 30 * np.sin(np.arange(160) / 25)])
    return img, xy


def test_qc_palette_is_contrasting_not_curve_colour():
    # QC must never reuse the curve's own colour (would vanish over same-colour ink)
    assert _qc_hex(0) != _qc_hex(1)
    assert all(c.startswith("#") and len(c) == 7 for c in (_qc_hex(0), _qc_hex(9)))


def test_write_qc_emits_all_three_artifacts(tmp_path):
    img, xy = _panel_and_curve()
    out = write_qc(str(tmp_path), img, [{"name": "curve", "xy": xy, "rgb": (0, 0, 0)}],
                   panel_stem="panel", title="demo")
    for k in ("panel_png", "overlay_png", "check_html"):
        assert os.path.exists(out[k]), k


def test_overlay_matches_panel_dimensions_and_has_alpha(tmp_path):
    img, xy = _panel_and_curve()
    out = write_qc(str(tmp_path), img, [{"name": "c", "xy": xy, "rgb": None}])
    ov = cv2.imread(out["overlay_png"], cv2.IMREAD_UNCHANGED)
    assert ov.shape[:2] == img.shape[:2]      # registers pixel-for-pixel
    assert ov.shape[2] == 4                    # transparent background
    assert ov[:, :, 3].max() > 0               # something was drawn


def test_check_html_links_to_trace_assist_with_this_panel(tmp_path):
    img, xy = _panel_and_curve()
    out = write_qc(str(tmp_path), img, [{"name": "c", "xy": xy, "rgb": None}],
                   panel_stem="panel", title="demo")
    html = open(out["check_html"], encoding="utf-8").read()
    assert "trace_assist.html?panel=" in html
    assert "panel.png" in html                 # the handoff targets this panel
    assert 'id="op"' in html                    # the fade-by-eye slider is present


def test_write_qc_scores_fidelity_and_flags_a_chord(tmp_path):
    # white panel with a black curve on the left; the trace chords across to the right
    img = np.full((200, 400, 3), 255, np.uint8)
    img[98:103, 20:180] = 0                       # real ink on the left
    trace = np.column_stack([np.arange(20, 380), np.full(360, 100, float)])
    out = write_qc(str(tmp_path), img, [{"name": "c", "xy": trace, "rgb": (0, 0, 0)}])
    fid = out["fidelity"][0]
    assert fid["score"] < 70 and fid["n_defects"] >= 1   # chord detected + scored
    html = open(out["check_html"], encoding="utf-8").read()
    assert "#e74c3c" in html                       # a red chord marker is drawn
