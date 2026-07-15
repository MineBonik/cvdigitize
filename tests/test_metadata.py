"""Tests for caption / text-layer metadata extraction."""
import fitz

from cvdigitize.metadata import extract_figure_metadata, FigureMetadata


def _text_pdf(path, text):
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in text.splitlines():
        page.insert_text((72, y), line, fontsize=10)
        y += 14
    doc.save(path); doc.close()


def test_extracts_scan_rate_electrolyte_ref(tmp_path):
    pdf = str(tmp_path / "cap.pdf")
    _text_pdf(pdf,
              "Figure 2. Cyclic voltammograms of Pt(111) in 0.1 M HClO4 and\n"
              "0.5 M H2SO4 recorded at 50 mV/s vs RHE at 25 C.\n"
              "Some following body text that should not be in the caption.")
    m = extract_figure_metadata(pdf, 0)
    assert m.scan_rate == "50 mV/s"
    assert "0.1 M HClO4" in m.electrolytes
    assert "0.5 M H2SO4" in m.electrolytes
    assert m.reference_electrode == "RHE"
    assert m.temperature == "25 °C"
    assert m.caption.startswith("Figure 2.")


def test_scan_rate_unit_variants(tmp_path):
    for raw, want in [("100 mV s-1", "100 mV/s"), ("0.05 V/s", "0.05 V/s"),
                      ("20mVs", "20 mV/s")]:
        pdf = str(tmp_path / "s.pdf")
        _text_pdf(pdf, f"Fig. 1. Something at {raw} in 1 M KOH.")
        m = extract_figure_metadata(pdf, 0)
        assert m.scan_rate == want


def test_reference_electrode_from_axis_title(tmp_path):
    pdf = str(tmp_path / "noref.pdf")
    _text_pdf(pdf, "Figure 1. A plot with no reference in the caption text.")
    m = extract_figure_metadata(pdf, 0, axis_titles=("E vs Ag/AgCl / V", "j"))
    assert m.reference_electrode == "Ag/AgCl"


def test_empty_when_no_cues(tmp_path):
    pdf = str(tmp_path / "plain.pdf")
    _text_pdf(pdf, "Just some ordinary prose with no experimental parameters.")
    m = extract_figure_metadata(pdf, 0)
    assert m.scan_rate == ""
    assert m.electrolytes == []
    assert m.is_empty or m.caption == ""


def test_as_dict_shape():
    m = FigureMetadata(scan_rate="50 mV/s", electrolytes=["1 M KOH"],
                       reference_electrode="RHE")
    d = m.as_dict()
    assert d["scanRate"] == "50 mV/s"
    assert d["electrolytes"] == ["1 M KOH"]
    assert "note" in d
