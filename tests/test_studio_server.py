"""Server tests for CV Studio's API (STUDIO_PLAN.md §12): drive each endpoint
with urllib against a background ThreadingHTTPServer bound to an OS-assigned
port (``--no-open`` equivalent), and assert response shape + files land in
the workspace. No browser needed."""
import base64
import json
import os
import threading
import urllib.error
import urllib.request

import cv2
import fitz
import numpy as np
import pytest

from cvdigitize.studio.server import make_server


@pytest.fixture
def server(tmp_path):
    workspace = str(tmp_path / "workspace")
    srv = make_server(workspace, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    port = srv.server_address[1]
    yield f"http://127.0.0.1:{port}", workspace
    srv.shutdown()
    srv.server_close()


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return r.status, json.loads(r.read())


def _post(base, path, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _make_pdf_with_embedded_image(path: str):
    """A one-page PDF whose page is dominated by one embedded raster image,
    so find_image_regions picks it up (mirrors a scanned-figure paper)."""
    img = np.full((240, 320, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (300, 220), (0, 0, 0), 2)
    xs = np.arange(30, 290)
    ys = 120 + 70 * np.sin(xs / 40)
    for x, y in zip(xs, ys.astype(int)):
        cv2.circle(img, (int(x), y), 2, (0, 0, 0), -1)
    png_path = path + ".png"
    cv2.imwrite(png_path, img)

    doc = fitz.open()
    page = doc.new_page(width=320, height=240)
    page.insert_image(fitz.Rect(0, 0, 320, 240), filename=png_path)
    doc.save(path)
    doc.close()
    os.remove(png_path)


def test_open_paper_extracts_sources_and_writes_analysis(server, tmp_path):
    base, workspace = server
    pdf = str(tmp_path / "hoshi_2008_surface_6070.pdf")
    _make_pdf_with_embedded_image(pdf)

    status, out = _post(base, "/api/open_paper", {"pdf_path": pdf})
    assert status == 200
    assert out["paper"] == "hoshi_2008_surface_6070"
    assert len(out["pages"]) == 1
    page0 = out["pages"][0]
    assert page0["n_embedded"] >= 1
    assert page0["sources"], "expected at least one source image"

    analysis_path = os.path.join(workspace, "hoshi_2008_surface_6070", "analysis.json")
    assert os.path.exists(analysis_path)
    for name in page0["sources"]:
        assert os.path.exists(os.path.join(workspace, "hoshi_2008_surface_6070", "sources", name))


def test_open_paper_missing_pdf_returns_404(server):
    base, _ = server
    status, out = _post(base, "/api/open_paper", {"pdf_path": "does_not_exist.pdf"})
    assert status == 404
    assert "error" in out


def test_open_paper_missing_field_returns_400(server):
    base, _ = server
    status, out = _post(base, "/api/open_paper", {})
    assert status == 400
    assert "error" in out


def _tiny_crop_data_url() -> str:
    img = np.full((40, 60, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def test_save_list_delete_crop_roundtrip(server, tmp_path):
    base, workspace = server
    pdf = str(tmp_path / "paperX.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})

    status, out = _post(base, "/api/save_crop", {
        "paper": "paperX", "type": "single_cv", "source": "p0_img0.png",
        "bbox": [10, 10, 50, 50], "excludeRects": [], "image": _tiny_crop_data_url(),
    })
    assert status == 200
    crop_name = out["crop"]
    assert crop_name == "paperX_crop1"
    png_path = os.path.join(workspace, "paperX", "crops", crop_name + ".png")
    json_path = os.path.join(workspace, "paperX", "crops", crop_name + ".json")
    assert os.path.exists(png_path)
    assert os.path.exists(json_path)
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["type"] == "single_cv"
    assert meta["bbox"] == [10, 10, 50, 50]

    status, out = _get(base, "/api/list_crops?paper=paperX")
    assert status == 200
    assert [c["name"] for c in out["crops"]] == [crop_name]

    # a re-crop chains off a parent crop, not a raw source
    status, out2 = _post(base, "/api/save_crop", {
        "paper": "paperX", "type": "single_cv", "source": crop_name,
        "bbox": [0, 0, 30, 30], "excludeRects": [], "image": _tiny_crop_data_url(),
        "parentCrop": crop_name,
    })
    assert status == 200
    assert out2["crop"] == "paperX_crop2"
    assert out2["meta"]["parentCrop"] == crop_name

    status, out = _post(base, "/api/delete_crop", {"paper": "paperX", "crop": crop_name})
    assert status == 200
    assert out["ok"] is True
    assert not os.path.exists(png_path)
    assert not os.path.exists(json_path)

    status, out = _get(base, "/api/list_crops?paper=paperX")
    assert [c["name"] for c in out["crops"]] == ["paperX_crop2"]


def test_save_crop_bad_type_returns_400(server, tmp_path):
    base, _ = server
    pdf = str(tmp_path / "paperY.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})

    status, out = _post(base, "/api/save_crop", {
        "paper": "paperY", "type": "not_a_real_type", "source": "p0_img0.png",
        "bbox": [0, 0, 10, 10], "excludeRects": [], "image": _tiny_crop_data_url(),
    })
    assert status == 400
    assert "error" in out


def test_workspace_static_serving_and_traversal_guard(server, tmp_path):
    base, workspace = server
    pdf = str(tmp_path / "paperZ.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})

    with urllib.request.urlopen(base + "/workspace/paperZ/sources/p0_img0.png") as r:
        assert r.status == 200
        assert r.headers["Content-Type"] == "image/png"

    # a traversal attempt must not escape the workspace root
    try:
        with urllib.request.urlopen(base + "/workspace/../../../../etc/passwd") as r:
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    assert status in (400, 404)


def test_unknown_route_is_404(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/api/does_not_exist")
    assert exc.value.code == 404


def _framed_axes_data_url() -> str:
    """A crop with a real axes frame + evenly-spaced ticks (same synthetic
    pattern as tests/test_raster_extract.py's detect_axis_ticks test), so
    detect_frame_bbox/detect_axis_ticks reliably find something to report."""
    img = np.full((300, 400, 3), 255, np.uint8)
    box = (60, 40, 340, 240)
    cv2.rectangle(img, box[:2], box[2:], (0, 0, 0), 2)
    x0, y0, x1, y1 = box
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        x = x0 + int((x1 - x0) * f)
        cv2.line(img, (x, y1 + 3), (x, y1 + 8), (0, 0, 0), 1)
    for f in (0.15, 0.5, 0.85):
        y = y0 + int((y1 - y0) * f)
        cv2.line(img, (x0 - 8, y), (x0 - 3, y), (0, 0, 0), 1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _seed_paper_with_crop(base, tmp_path, paper_name="paperCal"):
    pdf = str(tmp_path / f"{paper_name}.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})
    status, out = _post(base, "/api/save_crop", {
        "paper": paper_name, "type": "single_cv", "source": "p0_img0.png",
        "bbox": [0, 0, 400, 300], "excludeRects": [], "image": _framed_axes_data_url(),
    })
    assert status == 200
    return out["crop"]


def test_autocalibrate_finds_frame_and_tick_positions(server, tmp_path):
    base, _ = server
    crop_name = _seed_paper_with_crop(base, tmp_path)

    status, out = _post(base, "/api/autocalibrate", {"paper": "paperCal", "crop": crop_name})
    assert status == 200
    for anchor in ("E1", "E2", "j1", "j2"):
        assert anchor in out["points"]
        assert len(out["points"][anchor]) == 2
        assert anchor in out["labelCrops"]
        assert out["labelCrops"][anchor].startswith("data:image/png;base64,")
    assert len(out["frame"]) == 4


def test_autocalibrate_missing_crop_is_404(server, tmp_path):
    base, _ = server
    pdf = str(tmp_path / "paperNo.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})
    status, out = _post(base, "/api/autocalibrate", {"paper": "paperNo", "crop": "nope_crop1"})
    assert status == 404
    assert "error" in out


def test_save_calibration_persists_into_crop_json(server, tmp_path):
    base, workspace = server
    crop_name = _seed_paper_with_crop(base, tmp_path, paper_name="paperCal2")
    calibration = {
        "E1": {"px": [70, 200], "value": 0.0}, "E2": {"px": [330, 200], "value": 1.0},
        "j1": {"px": [60, 70], "value": 100}, "j2": {"px": [60, 210], "value": -100},
        "E_unit": "V", "E_ref": "RHE", "j_unit": "uA/cm2",
    }
    status, out = _post(base, "/api/save_calibration",
                        {"paper": "paperCal2", "crop": crop_name, "calibration": calibration})
    assert status == 200
    assert out["ok"] is True

    json_path = os.path.join(workspace, "paperCal2", "crops", crop_name + ".json")
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["calibration"] == calibration
    # the rest of the crop's metadata must survive the read-modify-write
    assert meta["type"] == "single_cv"

    status, out = _get(base, "/api/list_crops?paper=paperCal2")
    assert out["crops"][0]["calibration"] == calibration


def test_save_calibration_missing_crop_is_404(server, tmp_path):
    base, _ = server
    pdf = str(tmp_path / "paperNo2.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})
    status, out = _post(base, "/api/save_calibration",
                        {"paper": "paperNo2", "crop": "nope_crop1", "calibration": {}})
    assert status == 404
    assert "error" in out


def test_require_crop_image_rejects_empty_array_cleanly(tmp_path, monkeypatch):
    """A decoded-but-empty image (e.g. a degenerate 0-size crop that somehow
    made it to disk) must raise a clean, catchable error at this boundary --
    not propagate into OpenCV and crash with a cryptic native assertion."""
    from cvdigitize.studio import pipeline
    from cvdigitize.studio import workspace as ws

    workspace = str(tmp_path / "workspace")
    meta = ws.save_crop(workspace, "paperEmpty", type_="single_cv", source="p0_img0.png",
                        bbox=[0, 0, 10, 10], exclude_rects=[], image_data_url=_tiny_crop_data_url())
    crop_name = meta["name"]

    monkeypatch.setattr(pipeline.ws, "load_crop_image", lambda *a, **k: np.zeros((0, 0, 3), np.uint8))
    with pytest.raises(FileNotFoundError):
        pipeline._require_crop_image(workspace, "paperEmpty", crop_name)


# --------------------------------------------------------------------------- #
# G4: measure / autoextract / trace / accept_curves
# --------------------------------------------------------------------------- #
_CV_BOX = (60, 40, 340, 240)   # x0, y0, x1, y1 - the drawn axes frame
_CV_CALIBRATION = {
    "E1": {"px": [60, 240], "value": 0.0}, "E2": {"px": [340, 240], "value": 1.0},
    "j1": {"px": [60, 240], "value": -100}, "j2": {"px": [60, 40], "value": 100},
    "E_unit": "V", "E_ref": "RHE", "j_unit": "uA/cm2",
}


def _cv_crop_data_url() -> str:
    """A framed axes box (thin, 2px-drawn line) with a wavy dark curve (3px
    radius, so a visibly thicker stroke) inside - line_width should measure
    noticeably larger than axis_width, and the curve should auto-extract."""
    img = np.full((300, 400, 3), 255, np.uint8)
    x0, y0, x1, y1 = _CV_BOX
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)
    xs = np.arange(x0 + 10, x1 - 10)
    ys = (140 + 60 * np.sin((xs - x0) / 40)).astype(int)
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 0), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _seed_calibrated_cv_crop(base, tmp_path, paper_name="paperExtract"):
    pdf = str(tmp_path / f"{paper_name}.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})
    status, out = _post(base, "/api/save_crop", {
        "paper": paper_name, "type": "single_cv", "source": "p0_img0.png",
        "bbox": [0, 0, 400, 300], "excludeRects": [], "image": _cv_crop_data_url(),
    })
    assert status == 200
    crop_name = out["crop"]
    status, out = _post(base, "/api/save_calibration",
                        {"paper": paper_name, "crop": crop_name, "calibration": _CV_CALIBRATION})
    assert status == 200
    return crop_name


def test_measure_reports_line_wider_than_axis_and_persists(server, tmp_path):
    base, workspace = server
    crop_name = _seed_calibrated_cv_crop(base, tmp_path)

    status, out = _post(base, "/api/measure", {"paper": "paperExtract", "crop": crop_name})
    assert status == 200
    assert out["line_width"] > out["axis_width"] > 0
    assert out["suggested_brush"] == round(max(3.0, 1.8 * out["line_width"]), 2)
    assert "exclusion_band" in out

    json_path = os.path.join(workspace, "paperExtract", "crops", crop_name + ".json")
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["lineWidth"] == out["line_width"]
    assert meta["axisWidth"] == out["axis_width"]


def test_measure_without_calibration_is_400(server, tmp_path):
    base, _ = server
    pdf = str(tmp_path / "paperNoCalib.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})
    status, out = _post(base, "/api/save_crop", {
        "paper": "paperNoCalib", "type": "single_cv", "source": "p0_img0.png",
        "bbox": [0, 0, 10, 10], "excludeRects": [], "image": _tiny_crop_data_url(),
    })
    crop_name = out["crop"]
    status, out = _post(base, "/api/measure", {"paper": "paperNoCalib", "crop": crop_name})
    assert status == 400
    assert "error" in out


def test_autoextract_finds_the_curve_with_good_fidelity(server, tmp_path):
    base, _ = server
    crop_name = _seed_calibrated_cv_crop(base, tmp_path, paper_name="paperExtract2")

    status, out = _post(base, "/api/autoextract", {"paper": "paperExtract2", "crop": crop_name})
    assert status == 200
    assert out["curves"], "expected at least one extracted curve"
    c = out["curves"][0]
    assert len(c["xy_px"]) == len(c["xy_real"])
    assert c["fidelity"]["score"] is not None and c["fidelity"]["score"] >= 70
    assert "measurement" in out and out["measurement"]["line_width"] > 0


def test_autoextract_missing_crop_is_404(server, tmp_path):
    base, _ = server
    pdf = str(tmp_path / "paperExtractNo.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})
    status, out = _post(base, "/api/autoextract", {"paper": "paperExtractNo", "crop": "nope_crop1"})
    assert status == 404
    assert "error" in out


def test_trace_snaps_guide_to_ink(server, tmp_path):
    base, _ = server
    crop_name = _seed_calibrated_cv_crop(base, tmp_path, paper_name="paperTrace")

    x0, y0, x1, y1 = _CV_BOX
    xs = np.linspace(x0 + 10, x1 - 10, 12)
    guide_pts = [[float(x), float(140 + 60 * np.sin((x - x0) / 40))] for x in xs]
    guides = [{"name": "hand", "radius": 10, "strokes": [{"radius": 10, "pts": guide_pts}]}]

    status, out = _post(base, "/api/trace",
                        {"paper": "paperTrace", "crop": crop_name, "guides": guides})
    assert status == 200
    assert out["curves"], "expected the guide to snap onto the drawn curve"
    c = out["curves"][0]
    assert c["name"] == "hand"
    assert c["fidelity"]["score"] is not None


def test_trace_masks_frame_border_so_guide_cannot_snap_onto_it(server, tmp_path):
    """A hand-trace guide that overshoots toward the plot's own outer frame
    must never snap onto that border ink -- the real bug (reported live): a
    hand-traced curve escaped along the frame border after the real ink
    faded out near a corner, chording across empty space onto it. trace_crop
    masks the axis + frame border out of the ink before extraction, same as
    autoextract_crop, so the brush is structurally forbidden from ever
    landing on either."""
    base, _ = server
    paper_name = "paperTraceFrame"
    h, w = 300, 500
    img = np.full((h, w, 3), 255, np.uint8)
    frame_box = (20, 20, 480, 280)   # a thick 6px outer frame
    cv2.rectangle(img, frame_box[:2], frame_box[2:], (0, 0, 0), 6)
    axis_row, axis_col = 200, 60
    cv2.line(img, (axis_col, axis_row), (460, axis_row), (0, 0, 0), 2)
    cv2.line(img, (axis_col, 30), (axis_col, axis_row), (0, 0, 0), 2)
    xs = np.arange(80, 420)   # the curve's real ink ends well short of the frame
    ys = (axis_row - 80 * np.exp(-((xs - 180.0) ** 2) / (2 * 40.0 ** 2))).astype(int)
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 0), -1)
    ok, buf = cv2.imencode(".png", img)
    data_url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    pdf = str(tmp_path / f"{paper_name}.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})
    status, out = _post(base, "/api/save_crop", {
        "paper": paper_name, "type": "single_cv", "source": "p0_img0.png",
        "bbox": [0, 0, w, h], "excludeRects": [], "image": data_url,
    })
    crop_name = out["crop"]
    calibration = {
        "E1": {"px": [axis_col, axis_row], "value": 0.0}, "E2": {"px": [440, axis_row], "value": 1.0},
        "j1": {"px": [axis_col, axis_row], "value": -100}, "j2": {"px": [axis_col, 30], "value": 100},
        "E_unit": "V", "E_ref": "RHE", "j_unit": "uA/cm2",
    }
    _post(base, "/api/save_calibration",
         {"paper": paper_name, "crop": crop_name, "calibration": calibration})

    # a guide that deliberately overshoots well past the curve's real end,
    # running close to and roughly parallel to the frame's right border
    guide_xs = np.linspace(85, 478, 40)
    guide_ys = np.full_like(guide_xs, float(axis_row))
    guides = [{"name": "hand", "radius": 15,
              "strokes": [{"radius": 15,
                          "pts": [[float(x), float(y)] for x, y in zip(guide_xs, guide_ys)]}]}]

    status, out = _post(base, "/api/trace",
                        {"paper": paper_name, "crop": crop_name, "guides": guides})
    assert status == 200
    assert out["curves"]
    pts = np.asarray(out["curves"][0]["xy_px"])
    x0f, y0f, x1f, y1f = frame_box
    margin = 8
    on_border = ((pts[:, 0] <= x0f + margin) | (pts[:, 0] >= x1f - margin)
                | (pts[:, 1] <= y0f + margin) | (pts[:, 1] >= y1f - margin))
    assert not on_border.any(), "trace must never snap onto the frame's own border ink"


def test_accept_curves_persists_onto_crop(server, tmp_path):
    base, workspace = server
    crop_name = _seed_calibrated_cv_crop(base, tmp_path, paper_name="paperAccept")

    curves = [{"name": "dark", "rgb": [0.1, 0.1, 0.1],
              "xy_px": [[0, 0], [1, 1]], "xy_real": [[0.0, -100.0], [0.01, -99.0]]}]
    status, out = _post(base, "/api/accept_curves",
                        {"paper": "paperAccept", "crop": crop_name, "curves": curves})
    assert status == 200
    assert out["ok"] is True

    status, out = _get(base, "/api/list_crops?paper=paperAccept")
    assert out["crops"][0]["curves"] == curves


# --------------------------------------------------------------------------- #
# G6: save_curves (echemdb naming) + resume
# --------------------------------------------------------------------------- #
def test_save_curves_writes_datapackage_with_echemdb_naming(server, tmp_path):
    base, workspace = server
    crop_name = _seed_calibrated_cv_crop(base, tmp_path, paper_name="paperSave")

    curves = [
        {"name": "dark", "label": "Pt(111) 0.1 M HClO4",
         "xy_real": [[0.0, -100.0], [0.5, 0.0], [1.0, 100.0]]},
        {"name": "red", "figtag": "f2a", "label": "red",
         "xy_real": [[0.0, 50.0], [1.0, -50.0]]},
    ]
    status, out = _post(base, "/api/save_curves",
                        {"paper": "paperSave", "crop": crop_name, "curves": curves})
    assert status == 200
    csvs = sorted(w["csv"] for w in out["written"])
    assert csvs == [
        "curves/paperSave_crop1_Pt_111_0.1_M_HClO4.csv",
        "curves/paperSave_f2a_red.csv",
    ]
    for rel in csvs:
        assert os.path.exists(os.path.join(workspace, "paperSave", rel))
        # sibling JSON + YAML frictionless descriptors also land alongside the CSV
        assert os.path.exists(os.path.join(workspace, "paperSave", rel[:-4] + ".json"))
        assert os.path.exists(os.path.join(workspace, "paperSave", rel[:-4] + ".yaml"))

    csv_text = open(os.path.join(workspace, "paperSave", csvs[0]), encoding="utf-8").read()
    assert "E,j" in csv_text.splitlines()[0]

    index_path = os.path.join(workspace, "paperSave", "index.html")
    assert os.path.exists(index_path)
    index_html = open(index_path, encoding="utf-8").read()
    assert crop_name in index_html
    assert "paperSave_crop1_Pt_111_0.1_M_HClO4.csv" in index_html


def test_save_curves_missing_crop_is_404(server, tmp_path):
    base, _ = server
    pdf = str(tmp_path / "paperSaveNo.pdf")
    _make_pdf_with_embedded_image(pdf)
    _post(base, "/api/open_paper", {"pdf_path": pdf})
    status, out = _post(base, "/api/save_curves", {
        "paper": "paperSaveNo", "crop": "nope_crop1",
        "curves": [{"name": "x", "xy_real": [[0, 0], [1, 1]]}],
    })
    assert status == 404
    assert "error" in out


def test_save_curves_empty_list_is_400(server, tmp_path):
    base, _ = server
    crop_name = _seed_calibrated_cv_crop(base, tmp_path, paper_name="paperSaveEmpty")
    status, out = _post(base, "/api/save_curves",
                        {"paper": "paperSaveEmpty", "crop": crop_name, "curves": []})
    assert status == 400
    assert "error" in out


def test_open_paper_resume_reports_last_crop_and_step(server, tmp_path):
    base, _ = server
    pdf = str(tmp_path / "paperResume.pdf")
    _make_pdf_with_embedded_image(pdf)

    status, out = _post(base, "/api/open_paper", {"pdf_path": pdf})
    assert out["last_crop"] is None
    assert out["last_step"] == 1

    status, out = _post(base, "/api/save_crop", {
        "paper": "paperResume", "type": "single_cv", "source": "p0_img0.png",
        "bbox": [0, 0, 10, 10], "excludeRects": [], "image": _tiny_crop_data_url(),
    })
    crop_name = out["crop"]
    _post(base, "/api/save_calibration",
         {"paper": "paperResume", "crop": crop_name, "calibration": _CV_CALIBRATION})

    status, out = _post(base, "/api/open_paper", {"pdf_path": pdf})
    assert status == 200
    assert out["last_crop"] == crop_name
    assert out["last_step"] == 2   # save_calibration is the most recent action -> step 2
