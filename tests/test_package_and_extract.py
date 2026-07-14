"""Tests for datapackage writing and colour helpers."""
import json

import numpy as np

from cvdigitize.package import build_descriptor, write_datapackage, CurveMeta
from cvdigitize.vector_extract import color_name, is_light_gray


def test_color_name_known_and_fallback():
    assert color_name((0.0, 0.0, 0.0)) == "black"
    assert color_name((0.0, 0.0, 1.0)) == "blue"
    assert color_name((0.5, 0.0, 0.5)) == "violet"
    # unknown colour -> hex fallback
    assert color_name((0.0, 0.63, 0.0)).startswith("c_")


def test_is_light_gray():
    assert is_light_gray((0.8, 0.8, 0.8))       # grid line
    assert not is_light_gray((0.0, 0.0, 0.0))   # black curve
    assert not is_light_gray((1.0, 0.0, 0.0))   # saturated red


def test_build_descriptor_structure():
    meta = CurveMeta(name="demo", figure="1a", curve="Pt(111)",
                     scan_rate="50 mV/s", x_unit="V vs RHE", y_unit="uA/cm2")
    d = build_descriptor("demo.csv", meta)
    res = d["resources"][0]
    assert res["path"] == "demo.csv"
    fields = res["schema"]["fields"]
    assert [f["name"] for f in fields] == ["E", "j"]
    assert res["metadata"]["echemdb"]["source"]["figure"] == "1a"
    assert res["metadata"]["echemdb"]["figureDescription"]["scanRate"] == "50 mV/s"


def test_write_datapackage(tmp_path):
    data = np.column_stack([np.linspace(0, 1, 10), np.linspace(-1, 1, 10)])
    meta = CurveMeta(name="demo", figure="1a", curve="black")
    paths = write_datapackage(str(tmp_path), data, meta, yaml=True)
    assert set(paths) == {"csv", "json", "yaml"}
    # csv has header + 10 rows
    lines = open(paths["csv"]).read().strip().splitlines()
    assert lines[0] == "E,j"
    assert len(lines) == 11
    # json is valid frictionless descriptor
    with open(paths["json"]) as f:
        desc = json.load(f)
    assert desc["resources"][0]["name"] == "demo"
