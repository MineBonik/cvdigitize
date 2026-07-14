"""Unit tests for axis calibration."""
import numpy as np
import pytest

from cvdigitize.calibrate import (
    Calibration, calibration_from_anchors, calibration_from_bboxes,
)


def test_linear_map_endpoints():
    c = Calibration(x1=100, ex1=0.0, x2=300, ex2=1.0,
                    y1=50, jy1=150.0, y2=250, jy2=-150.0)
    E, j = c.to_data(np.array([100.0, 300.0]), np.array([50.0, 250.0]))
    assert E[0] == pytest.approx(0.0)
    assert E[1] == pytest.approx(1.0)
    assert j[0] == pytest.approx(150.0)
    assert j[1] == pytest.approx(-150.0)


def test_bbox_calibration_y_flip():
    # pixel bbox (x0,y0,x1,y1); data bbox (E0,j0,E1,j1)
    c = calibration_from_bboxes((10, 20, 110, 220), (0.0, -5.0, 1.0, 5.0))
    # top of plot (min pixel y) => max current
    _, j_top = c.to_data(np.array([10.0]), np.array([20.0]))
    _, j_bot = c.to_data(np.array([10.0]), np.array([220.0]))
    assert j_top[0] == pytest.approx(5.0)
    assert j_bot[0] == pytest.approx(-5.0)


def test_apply_shape():
    c = calibration_from_anchors((0, 0.0), (100, 1.0), (0, 0.0), (100, 10.0))
    xy = np.column_stack([np.linspace(0, 100, 5), np.linspace(0, 100, 5)])
    out = c.apply(xy)
    assert out.shape == (5, 2)
    assert out[-1, 0] == pytest.approx(1.0)
    assert out[-1, 1] == pytest.approx(10.0)


def test_save_load_roundtrip(tmp_path):
    c = Calibration(x1=1, ex1=2, x2=3, ex2=4, y1=5, jy1=6, y2=7, jy2=8,
                    x_unit="V vs RHE", y_unit="mA/cm2")
    p = tmp_path / "cal.json"
    c.save(str(p))
    c2 = Calibration.load(str(p))
    assert c2 == c
