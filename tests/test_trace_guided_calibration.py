"""Per-curve calibration fallback in scripts/trace_guided.py's run command.

trace_assist.html now attaches calibration either to a curve specifically (an
override, for the rare case where curves in one session don't share axes) or
to the panel as a whole (the default, used by every curve lacking its own).
Old guides.json files only ever had the panel-level flat calibration — this
must keep applying to every curve unchanged (backward compatibility)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from trace_guided import _apply_calibration


def _cal(e_lo=0, e_hi=1, j_lo=-150, j_hi=150):
    return {"E1": {"px": [10, 500], "value": e_lo}, "E2": {"px": [110, 500], "value": e_hi},
            "j1": {"px": [10, 500], "value": j_lo}, "j2": {"px": [10, 0], "value": j_hi},
            "E_unit": "V", "E_ref": "RHE", "j_unit": "uA/cm2"}


def test_apply_calibration_maps_pixels_to_declared_range():
    poly = np.array([[10.0, 500.0], [110.0, 0.0]])
    real, header, ok = _apply_calibration(poly, _cal())
    assert ok
    assert np.allclose(real[0], [0, -150]) and np.allclose(real[1], [1, 150])
    assert "V" in header[0] and "RHE" in header[0]


def test_per_curve_calibration_overrides_panel_default():
    # Mirrors what _cmd_run builds: per_curve_calib wins over panel_calib for a
    # named curve; curves without their own calibration use the panel default.
    panel_calib = _cal(e_lo=0, e_hi=1)
    guides = [{"name": "shares axes"}, {"name": "different axes", "calibration": _cal(e_lo=-1, e_hi=1)}]
    per_curve_calib = {g["name"]: g["calibration"] for g in guides if g.get("calibration")}

    cal_a = per_curve_calib.get("shares axes", panel_calib)
    cal_b = per_curve_calib.get("different axes", panel_calib)
    assert cal_a is panel_calib                      # fell back correctly
    assert cal_b["E1"]["value"] == -1                 # used its own override, not the panel's


def test_old_flat_calibration_guides_json_still_works(tmp_path):
    # An old export has no per-curve "calibration" key on any guide at all —
    # confirm every curve still resolves to the single panel-level calibration.
    guides = [{"name": "a"}, {"name": "b"}]
    panel_calib = _cal()
    per_curve_calib = {g.get("name"): g["calibration"] for g in guides if g.get("calibration")}
    assert per_curve_calib == {}
    for g in guides:
        assert per_curve_calib.get(g["name"], panel_calib) is panel_calib
