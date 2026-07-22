"""Tests for `trace_guided.py triage` — processing a triage.json exported by
trace_assist.html's Crop mode (crop each real plot by hand, classify it CV /
strange / not-a-CV, let normal CVs auto-extract)."""
import base64
import json
import os
import sys
from argparse import Namespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from trace_guided import _cmd_triage


def _data_url(img_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _cv_panel():
    """A clean single dark curve inside a drawn frame — a normal, auto-traceable CV."""
    img = np.full((220, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (280, 200), (0, 0, 0), 2)
    xs = np.arange(30, 270)
    ys = 110 + 60 * np.sin(xs / 40)
    for x, y in zip(xs, ys.astype(int)):
        cv2.circle(img, (int(x), y), 2, (0, 0, 0), -1)
    return img


def test_cv_panel_auto_extracts_with_calibration(tmp_path):
    img = _cv_panel()
    triage = {
        "panels": [{
            "name": "fig1_panelA", "decision": "cv", "sourcePage": "paper_p1.png",
            "image": _data_url(img),
            "calibration": {"E1": {"px": [20, 210], "value": 0.0}, "E2": {"px": [280, 210], "value": 1.0},
                           "j1": {"px": [10, 200], "value": -100}, "j2": {"px": [10, 20], "value": 100},
                           "E_unit": "V", "E_ref": "RHE", "j_unit": "uA/cm2"},
            "guides": [],
        }],
        "rejected": [],
    }
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage), encoding="utf-8")
    out = tmp_path / "out"

    _cmd_triage(Namespace(triage=str(triage_path), out=str(out)))

    report = json.loads((out / "triage_report.json").read_text(encoding="utf-8"))
    assert report["panels"][0]["decision"] == "cv"
    assert report["panels"][0]["n_curves"] >= 1
    assert report["panels"][0]["calibrated"] is True
    pdir = out / "fig1_panelA"
    assert (pdir / "check.html").exists()
    assert (pdir / "curve_overlay.png").exists()
    csvs = list(pdir.glob("*.csv"))
    assert csvs, "expected at least one calibrated curve CSV"
    header = csvs[0].read_text().splitlines()[0]
    assert "V" in header and "uA/cm2" in header      # real units, not pixel coords


def test_strange_panel_with_guides_is_finalized(tmp_path):
    img = _cv_panel()
    triage = {
        "panels": [{
            "name": "fig1_panelB", "decision": "strange", "sourcePage": "paper_p1.png",
            "image": _data_url(img), "calibration": None,
            "guides": [{"name": "curve1", "radius": 10,
                       "strokes": [[[30, 110], [100, 130], [200, 100], [269, 108]]]}],
        }],
        "rejected": [],
    }
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage), encoding="utf-8")
    out = tmp_path / "out"

    _cmd_triage(Namespace(triage=str(triage_path), out=str(out)))

    report = json.loads((out / "triage_report.json").read_text(encoding="utf-8"))
    entry = report["panels"][0]
    assert entry["decision"] == "strange"
    assert entry["n_curves"] == 1
    assert (out / "fig1_panelB" / "check.html").exists()


def test_strange_panel_without_guides_just_saves_panel_png(tmp_path):
    img = _cv_panel()
    triage = {"panels": [{"name": "fig1_panelC", "decision": "strange", "sourcePage": "paper_p1.png",
                          "image": _data_url(img), "calibration": None, "guides": []}],
              "rejected": []}
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage), encoding="utf-8")
    out = tmp_path / "out"

    _cmd_triage(Namespace(triage=str(triage_path), out=str(out)))

    assert (out / "fig1_panelC" / "panel.png").exists()
    assert not (out / "fig1_panelC" / "check.html").exists()   # nothing traced yet — no QC to show


def test_not_cv_rejections_are_recorded_without_processing(tmp_path):
    triage = {"panels": [], "rejected": [{"sourcePage": "paper_p1.png", "rect": [0, 0, 100, 50]}]}
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage), encoding="utf-8")
    out = tmp_path / "out"

    _cmd_triage(Namespace(triage=str(triage_path), out=str(out)))

    report = json.loads((out / "triage_report.json").read_text(encoding="utf-8"))
    assert report["rejected"] == triage["rejected"]
    assert report["panels"] == []
