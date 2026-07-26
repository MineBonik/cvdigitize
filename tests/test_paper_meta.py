"""Tests for paper-level bibliographic extraction (DOI, title, SI inheritance)."""
import fitz
import pytest

from cvdigitize.paper_meta import (PaperMetadata, _looks_like_title, _squash,
                                   extract_paper_metadata)


def _pdf(path, text="", *, title=None, subject=None, author=None):
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_textbox(fitz.Rect(40, 40, 550, 500), text, fontsize=9)
    md = {k: "" for k in ("title", "author", "subject", "keywords",
                          "creator", "producer")}
    if title is not None:
        md["title"] = title
    if subject is not None:
        md["subject"] = subject
    if author is not None:
        md["author"] = author
    doc.set_metadata(md)
    doc.save(path)
    doc.close()


# --- DOI ------------------------------------------------------------------
def test_doi_from_page_text(tmp_path):
    p = str(tmp_path / "a.pdf")
    _pdf(p, "Some journal header\nhttps://doi.org/10.1021/acs.jpclett.4c01056\nAbstract")
    assert extract_paper_metadata(p).doi == "10.1021/acs.jpclett.4c01056"


def test_doi_from_pdf_metadata(tmp_path):
    p = str(tmp_path / "b.pdf")
    _pdf(p, "no doi in the body", subject="Journal 2019.5:1-9 doi:10.1039/c0cp00108b")
    assert extract_paper_metadata(p).doi == "10.1039/c0cp00108b"


def test_doi_trailing_punctuation_is_stripped(tmp_path):
    p = str(tmp_path / "c.pdf")
    _pdf(p, "cited as 10.1016/j.electacta.2008.01.083.")
    assert extract_paper_metadata(p).doi == "10.1016/j.electacta.2008.01.083"


def test_url_property_and_no_doi(tmp_path):
    p = str(tmp_path / "d.pdf")
    _pdf(p, "a paper with no identifier at all")
    meta = extract_paper_metadata(p)
    assert meta.doi == ""
    assert meta.url == ""
    assert "doi" not in meta.as_dict()


def test_missing_file_returns_blanks_not_raises():
    meta = extract_paper_metadata("does/not/exist.pdf")
    assert meta.is_empty and meta.doi == ""


# --- titles ---------------------------------------------------------------
@pytest.mark.parametrize("junk", [
    "RSC_CP_C4CP00260A 3..7",          # RSC production filename
    "PII: S0013-4686(08)00123-4",      # Elsevier PII
    "untitled",
    "jp1088146",                       # bare article id
    "short",                           # too short to be a title
])
def test_junk_titles_rejected(junk):
    assert not _looks_like_title(junk)


def test_real_title_accepted():
    assert _looks_like_title("Deconvolution of the Voltammetric Features of Pt(100)")


def test_junk_metadata_title_falls_back_to_layout(tmp_path):
    p = str(tmp_path / "e.pdf")
    # A junk metadata title plus a plausible title drawn large near the top.
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 90), "Structural effects on the oxygen reduction reaction",
                     fontsize=17)
    page.insert_text((60, 160), "body text set much smaller than the title",
                     fontsize=8)
    doc.set_metadata({"title": "RSC_CP_C4CP00260A 3..7"})
    doc.save(p)
    doc.close()
    assert "Structural effects" in extract_paper_metadata(p).title


def test_squash_collapses_justified_spacing():
    assert _squash("Elucidating   the   activity") == "Elucidating the activity"


def test_title_whitespace_is_normalised(tmp_path):
    p = str(tmp_path / "f.pdf")
    _pdf(p, "body", title="Enthalpic   and   Entropic   Effects   on   Adsorption")
    assert "  " not in extract_paper_metadata(p).title


# --- journal / year -------------------------------------------------------
def test_journal_and_year_from_acs_subject(tmp_path):
    p = str(tmp_path / "g.pdf")
    _pdf(p, "body", subject="J. Phys. Chem. Lett. 2024.15:4958-4964")
    meta = extract_paper_metadata(p)
    assert meta.journal == "J. Phys. Chem. Lett"
    assert meta.year == "2024"


# --- supplementary information -------------------------------------------
def test_si_inherits_doi_from_parent(tmp_path):
    main = tmp_path / "rizo_2025_analysis_351.pdf"
    si = tmp_path / "rizo_2025_analysis_si.pdf"
    _pdf(str(main), "doi 10.1021/acselectrochem.4c00107", title="Analysis of the OH Coverage")
    _pdf(str(si), "Supporting Information, no identifier here")

    meta = extract_paper_metadata(str(si))
    assert meta.doi == "10.1021/acselectrochem.4c00107"
    assert meta.inherited_from.endswith("rizo_2025_analysis_351.pdf")
    assert meta.as_dict()["doiInheritedFrom"] == "rizo_2025_analysis_351.pdf"


def test_si_without_parent_stays_empty(tmp_path):
    si = str(tmp_path / "lonely_2020_thing_si.pdf")
    _pdf(si, "Supporting Information with nothing to inherit")
    meta = extract_paper_metadata(si)
    assert meta.doi == "" and meta.inherited_from == ""


def test_si_does_not_recurse_past_one_level(tmp_path):
    """An SI whose only sibling is another SI must not loop."""
    a = str(tmp_path / "x_2020_a_si.pdf")
    b = str(tmp_path / "x_2020_a_supp.pdf")
    _pdf(a, "no doi")
    _pdf(b, "no doi either")
    assert extract_paper_metadata(a).doi == ""


def test_as_dict_omits_blanks_but_keeps_note():
    meta = PaperMetadata(doi="10.1/x", pdf="some/paper.pdf")
    d = meta.as_dict()
    assert d["doi"] == "10.1/x"
    assert d["url"] == "https://doi.org/10.1/x"
    assert d["pdf"] == "paper.pdf"
    assert "title" not in d and "journal" not in d
    assert "verify" in d["note"]
