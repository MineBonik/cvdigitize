"""Tests for datapackage writing and colour helpers."""
import json

import numpy as np
import pytest

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


# --------------------------------------------------------------------------- #
# YAML sidecar validity. The emitter is hand-rolled (no PyYAML dependency), and
# it used to write every string as a bare plain scalar -- so a curve label like
# "red: Pt(111)" or a caption containing a newline produced a file that would
# not parse at all ("mapping values are not allowed here"). The sidecar exists
# to be machine-read by echemdb, so an unparseable one is worse than none.
# --------------------------------------------------------------------------- #
def test_scalar_quotes_only_what_needs_it():
    from cvdigitize.package import _scalar
    assert _scalar("plain text") == "plain text"
    assert _scalar("10.1021/acs.jpclett.4c01056") == "10.1021/acs.jpclett.4c01056"
    for unsafe in ("red: Pt(111)", "a #b", "-lead", "end:", "  pad  ", ""):
        assert _scalar(unsafe).startswith('"'), unsafe


def test_scalar_keeps_numeric_looking_strings_as_strings():
    from cvdigitize.package import _scalar
    assert _scalar("2017") == '"2017"'
    assert _scalar("0.5") == '"0.5"'
    assert _scalar(0.5) == "0.5"          # a real number stays unquoted


def test_scalar_emits_yaml_booleans_and_null():
    from cvdigitize.package import _scalar
    assert _scalar(True) == "true"
    assert _scalar(False) == "false"
    assert _scalar(None) == "null"
    assert _scalar("true") == '"true"'    # the string must not become a bool


def test_written_yaml_parses_and_round_trips(tmp_path):
    yaml = pytest.importorskip("yaml")
    from cvdigitize.package import CurveMeta, write_yaml

    meta = CurveMeta(
        name="t", figure="1", curve="red: Pt(111) in 0.1 M HClO4",
        scan_rate="50 mV/s", x_unit="V", y_unit="q (uC cm-2)", source_pdf="p.pdf",
        extracted={"caption": "Figure 1.13 Comparison\nof charges: see (i).",
                   "electrolytes": ["0.1 M HClO4"],
                   "paper": {"title": "Surface Electrochemistry: Pt",
                             "year": "2017", "doi": "10.1021/x"}},
    )
    path = str(tmp_path / "t.yaml")
    write_yaml(path, "t.csv", meta)
    with open(path, encoding="utf-8") as f:
        parsed = yaml.safe_load(f)

    res = parsed["resources"][0]
    echemdb = res["metadata"]["echemdb"]
    assert echemdb["source"]["curve"] == "red: Pt(111) in 0.1 M HClO4"
    assert "\n" in echemdb["autoExtracted"]["caption"]
    assert echemdb["autoExtracted"]["paper"]["year"] == "2017"     # not int 2017
    assert echemdb["autoExtracted"]["paper"]["doi"] == "10.1021/x"
    assert res["schema"]["fields"][1]["unit"] == "q (uC cm-2)"
