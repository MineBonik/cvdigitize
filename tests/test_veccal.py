"""Tests for the vector-only curate/calibrate workflow (``vector-calibrate``).

Three layers, cheapest first: the pure naming/colour helpers, then
``finalize.save_panel`` driven by hand-built units so calibrated values can be
checked exactly, then the HTTP API against a background server. The one test
that needs a real figure builds a matplotlib PDF, which stores genuine vector
paths and real text tick labels — the same two things the scanner reads.
"""
import json
import os
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvdigitize.veccal import finalize
from cvdigitize.veccal.scan import (_describe_color, _slug, _stamp_tick_values,
                                    load_index, scan_folder)
from cvdigitize.veccal.server import make_server


# --------------------------------------------------------------------------- #
# colour naming
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rgb,expected", [
    ((0, 0, 0), "black"),
    ((1, 1, 1), "white"),
    ((0.318, 0.318, 0.318), "dark-gray"),     # the #515151 curve in luo_2022
    ((0.55, 0.55, 0.55), "gray"),
    ((1.0, 0.0, 0.0), "red"),
    ((0.0, 0.0, 1.0), "blue"),
    ((1.0, 0.5, 0.0), "orange"),
    ((0.5, 0.0, 1.0), "violet"),
    ((0.38, 0.0, 0.0), "dark-red"),           # #620000
])
def test_describe_color(rgb, expected):
    assert _describe_color(rgb) == expected


def test_describe_color_never_returns_hex():
    """The whole point of this helper: a word for every colour, never a hex."""
    rng = np.random.default_rng(0)
    for rgb in rng.random((60, 3)):
        assert "_" not in _describe_color(tuple(rgb)).replace("dark-", "").replace("light-", "")


def test_describe_color_clamps_out_of_range():
    assert _describe_color((-0.5, 2.0, 0.0))          # must not raise


# --------------------------------------------------------------------------- #
# filename slugs
# --------------------------------------------------------------------------- #
def test_slug_joins_concentration_units():
    assert _slug("0.02 M HClO4") == "0.02M-HClO4"


def test_slug_keeps_plus_in_mixtures():
    assert _slug("0.02 M HClO4 + 0.08 M KClO4") == "0.02M-HClO4+0.08M-KClO4"


def test_slug_strips_path_hostile_characters():
    out = _slug('a/b\\c:d*e?"f')
    assert not any(ch in out for ch in '/\\:*?"')


def test_slug_of_empty_is_empty():
    assert _slug("  ") == ""


# --------------------------------------------------------------------------- #
# tick value stamping
# --------------------------------------------------------------------------- #
def test_stamp_tick_values_interpolates_unlabelled_ticks():
    ticks = [{"pos": 0.0, "value": None}, {"pos": 5.0, "value": None},
             {"pos": 10.0, "value": None}]
    _stamp_tick_values(ticks, 0.0, 0.0, 10.0, 1.0)
    assert [t["value"] for t in ticks] == [0.0, 0.5, 1.0]


def test_stamp_tick_values_leaves_known_values_alone():
    ticks = [{"pos": 0.0, "value": 99.0}, {"pos": 10.0, "value": None}]
    _stamp_tick_values(ticks, 0.0, 0.0, 10.0, 1.0)
    assert ticks[0]["value"] == 99.0


def test_stamp_tick_values_degenerate_span_is_noop():
    ticks = [{"pos": 3.0, "value": None}]
    _stamp_tick_values(ticks, 5.0, 0.0, 5.0, 1.0)
    assert ticks[0]["value"] is None


# --------------------------------------------------------------------------- #
# finalize: calibration -> datapackages
# --------------------------------------------------------------------------- #
def _unit(loop=None, names=("curve_black",), colours=("c_000000",)):
    """A minimal panel unit; geometry defaults to a 10x10 pixel-space square."""
    if loop is None:
        loop = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    return {
        "uid": "paper_p0_a", "pdf": "paper.pdf", "page": 0, "stem": "paper",
        "panel": "a", "figure": "1a",
        "curves": [{"color": c, "color_label": "black", "name": n, "sample": "",
                    "rgb": [0, 0, 0], "loop_pdf": loop}
                   for n, c in zip(names, colours)],
        "figure_meta": {"scanRate": "50 mV/s"},
        "paper_meta": {"doi": "10.1/x", "title": "A paper"},
    }


def _cal(**over):
    """Maps pixel x 0..10 -> E 0..1 and pixel y 0..10 -> j +100..-100 (y flips)."""
    base = {"x1": 0, "ex1": 0.0, "x2": 10, "ex2": 1.0,
            "y1": 0, "jy1": 100.0, "y2": 10, "jy2": -100.0,
            "x_unit": "V vs RHE", "y_unit": "uA/cm2"}
    base.update(over)
    return base


def test_calibration_from_payload_maps_pixels_to_units():
    cal = finalize.calibration_from_payload(_cal())
    E, j = cal.to_data(np.array([0.0, 5.0, 10.0]), np.array([0.0, 5.0, 10.0]))
    assert np.allclose(E, [0.0, 0.5, 1.0])
    assert np.allclose(j, [100.0, 0.0, -100.0])       # y flip preserved


@pytest.mark.parametrize("bad,reason", [
    ({"x2": 0}, "same position"),
    ({"y2": 0}, "same position"),
    ({"ex2": 0.0}, "same number"),
    ({"jy2": 100.0}, "same number"),
])
def test_validate_rejects_degenerate_calibrations(bad, reason):
    problems = finalize.validate_calibration(finalize.calibration_from_payload(_cal(**bad)))
    assert problems and any(reason in p for p in problems)


def test_validate_accepts_a_good_calibration():
    assert finalize.validate_calibration(finalize.calibration_from_payload(_cal())) == []


def test_save_panel_writes_datapackage_in_real_units(tmp_path):
    out = str(tmp_path / "out")
    res = finalize.save_panel(_unit(), _cal(), out, resample=0)

    assert len(res["curves"]) == 1
    csv_path = os.path.join(out, "curve_black.csv")
    assert os.path.exists(csv_path)
    assert os.path.exists(os.path.join(out, "curve_black.json"))
    assert os.path.exists(os.path.join(out, "curve_black.yaml"))

    rows = [l.split(",") for l in open(csv_path, encoding="utf-8").read().splitlines()]
    assert rows[0] == ["E", "j"]
    xs = [float(r[0]) for r in rows[1:]]
    ys = [float(r[1]) for r in rows[1:]]
    assert min(xs) == pytest.approx(0.0) and max(xs) == pytest.approx(1.0)
    assert min(ys) == pytest.approx(-100.0) and max(ys) == pytest.approx(100.0)


def test_save_panel_records_doi_in_the_descriptor(tmp_path):
    out = str(tmp_path / "out")
    finalize.save_panel(_unit(), _cal(), out, resample=0)
    desc = json.load(open(os.path.join(out, "curve_black.json"), encoding="utf-8"))
    echemdb = desc["resources"][0]["metadata"]["echemdb"]
    assert echemdb["autoExtracted"]["paper"]["doi"] == "10.1/x"
    assert echemdb["figureDescription"]["scanRate"] == "50 mV/s"
    assert desc["resources"][0]["schema"]["fields"][0]["unit"] == "V vs RHE"


def test_save_panel_units_come_from_the_confirmed_calibration(tmp_path):
    out = str(tmp_path / "out")
    finalize.save_panel(_unit(), _cal(x_unit="V vs Ag/AgCl", y_unit="mA/cm2"),
                        out, resample=0)
    desc = json.load(open(os.path.join(out, "curve_black.json"), encoding="utf-8"))
    fields = desc["resources"][0]["schema"]["fields"]
    assert [f["unit"] for f in fields] == ["V vs Ag/AgCl", "mA/cm2"]


def test_save_panel_resamples_to_requested_count(tmp_path):
    out = str(tmp_path / "out")
    res = finalize.save_panel(_unit(), _cal(), out, resample=250)
    assert res["curves"][0]["n_points"] == 250


def test_save_panel_rejects_bad_calibration_before_writing(tmp_path):
    out = str(tmp_path / "out")
    with pytest.raises(ValueError, match="not usable"):
        finalize.save_panel(_unit(), _cal(x2=0), out)
    assert not os.path.exists(os.path.join(out, "curve_black.csv"))


def test_save_panel_disambiguates_colliding_user_names(tmp_path):
    """The name fields are free text, so two curves can be typed the same."""
    out = str(tmp_path / "out")
    unit = _unit(names=("same", "same"), colours=("c_000000", "c_ff0000"))
    res = finalize.save_panel(unit, _cal(), out, resample=0)
    assert [c["name"] for c in res["curves"]] == ["same", "same-2"]
    assert os.path.exists(os.path.join(out, "same.csv"))
    assert os.path.exists(os.path.join(out, "same-2.csv"))


def test_save_panel_skips_empty_geometry_but_keeps_the_rest(tmp_path):
    out = str(tmp_path / "out")
    unit = _unit(names=("good", "empty"), colours=("c_000000", "c_ff0000"))
    unit["curves"][1]["loop_pdf"] = []
    res = finalize.save_panel(unit, _cal(), out, resample=0)
    assert [c["name"] for c in res["curves"]] == ["good"]
    assert any("empty" in w for w in res["warnings"])


def test_save_panel_all_geometry_missing_raises(tmp_path):
    unit = _unit()
    unit["curves"][0]["loop_pdf"] = []
    with pytest.raises(ValueError, match="usable geometry"):
        finalize.save_panel(unit, _cal(), str(tmp_path / "out"))


# --------------------------------------------------------------------------- #
# scan over a real (synthetic) vector figure
# --------------------------------------------------------------------------- #
def _cv_pdf(path, *, colours=("black", "red")):
    """A vector PDF holding a framed CV-shaped loop with real text tick labels."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    t = np.linspace(0, 2 * np.pi, 900)
    for k, colour in enumerate(colours):
        # a lens/loop shape: closed, fills its bbox -> reads as CV-like
        ax.plot(0.5 + 0.42 * np.cos(t),
                (0.9 + 0.1 * k) * np.sin(t) * (0.6 + 0.4 * np.cos(t)),
                color=colour, lw=1.0)
    ax.set_xlim(0, 1); ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("E / V"); ax.set_ylabel("j")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def test_scan_folder_finds_a_cv_panel_and_names_its_curves(tmp_path):
    papers = tmp_path / "papers"; papers.mkdir()
    _cv_pdf(str(papers / "smith_2020_thing_1.pdf"))
    work = str(tmp_path / "work")

    index = scan_folder(str(papers), work)
    assert index["units"], "expected at least one CV panel"
    unit = index["units"][0]

    assert unit["stem"] == "smith_2020_thing_1"
    assert unit["status"] == "pending"
    assert len(unit["curves"]) >= 1
    # geometry is stored, in PDF points, ready to calibrate
    assert len(unit["curves"][0]["loop_pdf"]) > 10
    # a suggested filename that leads with the paper and is a real word
    assert unit["curves"][0]["name"].startswith("smith_2020_thing_1")
    assert "c_" not in unit["curves"][0]["color_label"]
    # the panel picture the user calibrates against exists
    assert os.path.exists(os.path.join(work, unit["image"]))


def test_scan_prefills_calibration_from_text_tick_labels(tmp_path):
    """Matplotlib writes tick labels as text, so this path must fire."""
    papers = tmp_path / "papers"; papers.mkdir()
    _cv_pdf(str(papers / "p_2020_a_1.pdf"))
    index = scan_folder(str(papers), str(tmp_path / "work"))
    unit = index["units"][0]
    assert unit["calib_source"] in ("auto-text", "auto-ocr")
    assert unit["calib_guess"] is not None
    assert unit["x_ticks"] and unit["y_ticks"]


def test_rescan_preserves_status_and_edited_names(tmp_path):
    papers = tmp_path / "papers"; papers.mkdir()
    _cv_pdf(str(papers / "p_2020_a_1.pdf"))
    work = str(tmp_path / "work")

    index = scan_folder(str(papers), work)
    uid = index["units"][0]["uid"]
    index["units"][0]["status"] = "saved"
    index["units"][0]["curves"][0]["name"] = "my-own-name"
    index["units"][0]["curves"][0]["sample"] = "Pt(111)"
    with open(os.path.join(work, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)

    again = scan_folder(str(papers), work)
    unit = next(u for u in again["units"] if u["uid"] == uid)
    assert unit["status"] == "saved"
    assert unit["curves"][0]["name"] == "my-own-name"
    assert unit["curves"][0]["sample"] == "Pt(111)"


def test_scan_empty_folder_yields_no_units(tmp_path):
    empty = tmp_path / "empty"; empty.mkdir()
    index = scan_folder(str(empty), str(tmp_path / "work"))
    assert index["units"] == [] and index["n_pdfs"] == 0


# --------------------------------------------------------------------------- #
# the HTTP API
# --------------------------------------------------------------------------- #
def _write_work(work: str, out: str):
    """A work dir with one pending unit and a stand-in panel image."""
    os.makedirs(os.path.join(work, "panels"), exist_ok=True)
    cv2.imwrite(os.path.join(work, "panels", "paper_p0_a.png"),
                np.full((40, 60, 3), 255, np.uint8))
    unit = _unit()
    unit.update({"image": "panels/paper_p0_a.png", "zoom": 3.0,
                 "origin": [0, 0], "size": [60, 40], "frame_pdf": [0, 0, 10, 10],
                 "calib_guess": None, "calib_source": "none",
                 "x_ticks": [], "y_ticks": [], "axis_titles": ["", ""],
                 "status": "pending"})
    with open(os.path.join(work, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"source_folder": "papers", "n_pdfs": 1, "units": [unit]}, f)


@pytest.fixture
def api(tmp_path):
    work, out = str(tmp_path / "work"), str(tmp_path / "curves")
    _write_work(work, out)
    srv = make_server(work, out, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", work, out
    srv.shutdown(); srv.server_close()


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_api_index_reports_counts(api):
    base, _, out = api
    status, body = _get(base, "/api/index")
    assert status == 200
    j = json.loads(body)
    assert j["counts"] == {"total": 1, "saved": 0, "skipped": 0,
                          "pending": 1, "prefilled": 0}
    assert j["units"][0]["uid"] == "paper_p0_a"
    assert j["out_dir"] == out


def test_api_serves_the_ui_and_panel_image(api):
    base, _, _ = api
    assert _get(base, "/")[0] == 200
    status, body = _get(base, "/panels/paper_p0_a.png")
    assert status == 200 and body[:4] == b"\x89PNG"


def test_api_save_writes_files_and_flips_status(api):
    base, work, out = api
    status, j = _post(base, "/api/save", {
        "uid": "paper_p0_a", "calibration": _cal(),
        "curves": [{"color": "c_000000", "name": "renamed", "sample": "Pt(111)"}],
        "scan_rate": "20 mV/s",
    })
    assert status == 200 and j["ok"]
    assert j["counts"]["saved"] == 1 and j["counts"]["pending"] == 0

    assert os.path.exists(os.path.join(out, "paper_p0_a", "renamed.csv"))
    desc = json.load(open(os.path.join(out, "paper_p0_a", "renamed.json"),
                          encoding="utf-8"))
    echemdb = desc["resources"][0]["metadata"]["echemdb"]
    assert echemdb["figureDescription"]["scanRate"] == "20 mV/s"
    assert echemdb["source"]["curve"] == "black: Pt(111)"

    # progress is persisted, so a reload resumes correctly
    assert load_index(work)["units"][0]["status"] == "saved"


def test_api_save_rejects_bad_calibration(api):
    base, _, out = api
    status, j = _post(base, "/api/save",
                      {"uid": "paper_p0_a", "calibration": _cal(x2=0)})
    assert status == 400 and "not usable" in j["error"]
    assert not os.path.exists(os.path.join(out, "paper_p0_a"))


def test_api_save_unknown_panel_is_404(api):
    base, _, _ = api
    status, j = _post(base, "/api/save", {"uid": "nope", "calibration": _cal()})
    assert status == 404 and "nope" in j["error"]


def test_api_skip_and_reopen(api):
    base, work, _ = api
    status, j = _post(base, "/api/skip", {"uid": "paper_p0_a", "reason": "not a CV"})
    assert status == 200 and j["counts"]["skipped"] == 1
    assert load_index(work)["units"][0]["skip_reason"] == "not a CV"

    status, j = _post(base, "/api/reopen", {"uid": "paper_p0_a"})
    assert status == 200 and j["counts"]["pending"] == 1


def test_api_edit_persists_names_without_writing_curves(api):
    base, work, out = api
    status, _ = _post(base, "/api/edit", {
        "uid": "paper_p0_a",
        "curves": [{"color": "c_000000", "name": "typed", "sample": "s"}]})
    assert status == 200
    unit = load_index(work)["units"][0]
    assert unit["curves"][0]["name"] == "typed"
    assert unit["status"] == "pending"          # editing is not saving
    assert not os.path.exists(os.path.join(out, "paper_p0_a"))


def test_api_rejects_path_traversal(api):
    base, _, _ = api
    status, _ = _get(base, "/panels/../../index.json")
    assert status in (403, 404)


def test_api_unknown_route_is_404(api):
    base, _, _ = api
    assert _get(base, "/api/nope")[0] == 404
