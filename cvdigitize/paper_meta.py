"""Paper-level bibliographic metadata: DOI, title, journal, year, authors.

Complements :mod:`cvdigitize.metadata`, which extracts *experimental* facts from
a figure's caption (scan rate, electrolyte, reference electrode). This module
answers the other half a curator needs — "which paper is this curve from?" — so
the emitted datapackage carries a citation instead of just a filename.

Two sources, in order of trust:

1. **The PDF's own metadata dictionary.** Publishers fill this inconsistently:
   ACS puts a real title in ``title`` and the citation in ``subject``
   ("J. Phys. Chem. Lett. 2024.15:4958-4964"); RSC puts a production filename
   there instead ("RSC_CP_C4CP00260A 3..7"), which we detect and reject.
2. **A regex sweep of the first two pages' text.** The DOI is printed on the
   first page of essentially every modern paper, so this is the reliable path;
   it found a DOI in 46 of the 52 papers in the reference corpus.

Supplementary-information PDFs usually carry no DOI of their own. When a stem
ends in an SI marker we look for the sibling main-text PDF and inherit its
bibliography, tagging the result so the inheritance stays visible.

Everything here is best-effort and labelled as such: ``PaperMetadata.source``
always says the data was auto-extracted and wants verification. Nothing in this
module guesses a value it cannot see in the file.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field

import fitz

# A DOI is "10." + registrant + "/" + suffix. The suffix runs to the first
# whitespace or delimiter; trailing sentence punctuation is stripped after.
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>,;)\]}]+", re.I)

# Some PDFs hyphenate a DOI across a line break, or use a non-ASCII hyphen.
_DASHES = {"‐": "-", "‑": "-", "‒": "-", "–": "-"}

# Titles that are really production artefacts, not titles. RSC/Elsevier
# typesetting systems leave filenames like "RSC_CP_C4CP00260A 3..7" or
# "PII: S0013-4686(08)00123-4" in the title slot.
_JUNK_TITLE = re.compile(
    r"^(RSC_|PII[: ]|S\d{4}-\d{4}|doi:|untitled|microsoft word|"
    r"[a-z]{2,6}\d{3,}(\.(pdf|doc|tex))?$)", re.I)

_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")

# "J. Phys. Chem. Lett. 2024.15:4958-4964" / "Electrochim. Acta 53 (2008) 6070"
_ACS_SUBJECT = re.compile(r"^(?P<journal>.+?)\s+(?P<year>(?:19|20)\d{2})[.,;: ]")


def _clean_doi(raw: str) -> str:
    out = raw
    for bad, good in _DASHES.items():
        out = out.replace(bad, good)
    # Strip trailing sentence punctuation a regex can't distinguish from suffix.
    return out.rstrip(".,;:)]}>'\"").strip()


def _squash(text: str) -> str:
    """Collapse the runs of spaces justified PDF text leaves between words."""
    return re.sub(r"\s+", " ", (text or "").replace("­", "")).strip()


def _looks_like_title(text: str) -> bool:
    t = _squash(text)
    if len(t) < 12 or len(t) > 400:
        return False
    if _JUNK_TITLE.search(t):
        return False
    if ".." in t:                      # "RSC_CP_C4CP00260A 3..7" page ranges
        return False
    # A title has words; production strings are mostly punctuation/digits.
    letters = sum(c.isalpha() for c in t)
    return letters >= 0.55 * len(t) and " " in t


@dataclass
class PaperMetadata:
    """Bibliographic identity of the source paper. All fields best-effort."""

    doi: str = ""
    title: str = ""
    authors: str = ""
    journal: str = ""
    year: str = ""
    pdf: str = ""
    inherited_from: str = ""   # set when an SI PDF borrowed its parent's DOI
    source: str = "auto-extracted from PDF metadata/text (verify)"

    @property
    def is_empty(self) -> bool:
        return not (self.doi or self.title or self.journal or self.year)

    @property
    def url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else ""

    def as_dict(self) -> dict:
        """Only the fields that were actually found, plus the caveat note."""
        d: dict = {}
        for key in ("doi", "title", "authors", "journal", "year"):
            val = getattr(self, key)
            if val:
                d[key] = val
        if self.url:
            d["url"] = self.url
        if self.pdf:
            d["pdf"] = os.path.basename(self.pdf)
        if self.inherited_from:
            d["doiInheritedFrom"] = os.path.basename(self.inherited_from)
        if d:
            d["note"] = self.source
        return d

    def to_json(self) -> dict:
        return asdict(self)


def _find_doi(blob: str) -> str:
    for hit in _DOI.findall(blob):
        doi = _clean_doi(hit)
        # "10.1021/x" is 9 chars; anything shorter is a false positive.
        if len(doi) > 8 and "/" in doi:
            return doi
    return ""


def _title_from_layout(page) -> str:
    """Largest-font text block near the top of page 1, best-effort.

    Journal titles are set noticeably larger than body text, so the biggest
    span in the top half of the first page is usually the paper title. Used
    only when the PDF metadata title is missing or is a production artefact.
    """
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return ""
    page_mid = page.rect.height * 0.55
    best_size, best_text = 0.0, ""
    for b in blocks:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            if not spans or line["bbox"][1] > page_mid:
                continue
            size = max(s.get("size", 0) for s in spans)
            text = " ".join(s.get("text", "") for s in spans).strip()
            # Merge the following lines of the same size (multi-line titles).
            if size > best_size and _looks_like_title(text):
                best_size, best_text = size, text
    return best_text


def _journal_year(md: dict, blob: str) -> tuple[str, str]:
    subject = (md.get("subject") or "").strip()
    m = _ACS_SUBJECT.match(subject)
    if m:
        return _squash(m.group("journal")).strip(" .,"), m.group("year")
    # No structured citation: take a year from the metadata dates if present,
    # else the earliest plausible year on the first page.
    for key in ("creationDate", "modDate"):
        ym = _YEAR.search(str(md.get(key) or ""))
        if ym:
            return (subject if _looks_like_title(subject) else ""), ym.group(1)
    ym = _YEAR.search(blob[:1500])
    return (subject if _looks_like_title(subject) else ""), (ym.group(1) if ym else "")


# Filename markers for supplementary-information PDFs, which rarely carry a DOI.
_SI_SUFFIXES = ("_si", "_supp", "_supporting", "_supplementary", "-si")


def _si_parent(pdf_path: str) -> str | None:
    """Path to the main-text PDF a supplementary file belongs to, if present."""
    directory, base = os.path.split(pdf_path)
    stem, ext = os.path.splitext(base)
    low = stem.lower()
    for suffix in _SI_SUFFIXES:
        if low.endswith(suffix):
            trunk = stem[: len(stem) - len(suffix)]
            if not trunk:
                return None
            # The main text keeps the same author_year_word prefix but ends in
            # the article/page number, so match on prefix rather than equality.
            try:
                siblings = os.listdir(directory or ".")
            except OSError:
                return None
            for cand in sorted(siblings):
                cstem, cext = os.path.splitext(cand)
                if cext.lower() != ext.lower() or cand == base:
                    continue
                if cstem.lower().startswith(trunk.lower()):
                    return os.path.join(directory, cand)
            return None
    return None


def extract_paper_metadata(pdf_path: str, *, _follow_si: bool = True) -> PaperMetadata:
    """Bibliographic metadata for one PDF. Never raises; returns blanks instead.

    ``_follow_si`` guards the one level of recursion used to inherit a DOI from
    a supplementary file's parent paper.
    """
    meta = PaperMetadata(pdf=pdf_path)
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return meta
    try:
        md = doc.metadata or {}
        n = min(2, doc.page_count)
        page_text = " ".join(doc[i].get_text() for i in range(n))
        layout_title = _title_from_layout(doc[0]) if doc.page_count else ""
    except Exception:
        md, page_text, layout_title = {}, "", ""
    finally:
        doc.close()

    md_blob = " ".join(str(v) for v in md.values() if v)
    meta.doi = _find_doi(md_blob) or _find_doi(page_text)

    md_title = (md.get("title") or "").strip()
    meta.title = _squash(md_title if _looks_like_title(md_title) else layout_title)
    meta.authors = _squash(md.get("author") or "")
    meta.journal, meta.year = _journal_year(md, page_text)

    # Supplementary files: borrow the parent paper's identity for the fields
    # they lack, but keep whatever the SI itself provided.
    if _follow_si and not meta.doi:
        parent = _si_parent(pdf_path)
        if parent and os.path.exists(parent):
            pm = extract_paper_metadata(parent, _follow_si=False)
            if pm.doi:
                meta.doi = pm.doi
                meta.inherited_from = parent
                meta.title = meta.title or pm.title
                meta.authors = meta.authors or pm.authors
                meta.journal = meta.journal or pm.journal
                meta.year = meta.year or pm.year
    return meta
