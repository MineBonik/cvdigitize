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


def test_curve_legend_flat():
    from cvdigitize.metadata import parse_curve_legend, legend_for_panel
    cap = "Figure 1. CVs of Pt: black line: Pt(111), red line: Pt(100), blue: Pt(110)."
    leg = parse_curve_legend(cap)
    m = legend_for_panel(leg, "")
    assert m["black"] == "Pt(111)"
    assert m["red"] == "Pt(100)"
    assert m["blue"] == "Pt(110)"


def test_curve_legend_per_panel_keeps_decimals():
    from cvdigitize.metadata import parse_curve_legend, legend_for_panel
    cap = ("Fig. 1. Profiles of Pt(111) in (A) black line: 0.10 M HClO4, pH 1.10 "
           "red line: 0.10 M MSA pH 1.10, (B) black line: 0.30 M HClO4 pH 0.64, "
           "red line: 0.30 M MSA pH 0.66. Scan rate=50 mV s-1.")
    leg = parse_curve_legend(cap)
    assert legend_for_panel(leg, "a") == {"black": "0.10 M HClO4", "red": "0.10 M MSA"}
    assert legend_for_panel(leg, "b") == {"black": "0.30 M HClO4", "red": "0.30 M MSA"}


def test_curve_legend_colour_aliases():
    from cvdigitize.metadata import parse_curve_legend, legend_for_panel
    leg = parse_curve_legend("Fig 1. purple: sample X, magenta line: sample Y.")
    m = legend_for_panel(leg, "")
    assert m.get("violet") == "sample X"   # purple -> violet
    assert m.get("pink") == "sample Y"     # magenta -> pink


def test_curve_legend_empty():
    from cvdigitize.metadata import parse_curve_legend
    assert parse_curve_legend("") == {}
    assert parse_curve_legend("A plain caption with no colours.") == {}
