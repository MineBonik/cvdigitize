"""Extract experimental metadata from a figure's caption / page text layer.

The email thread that motivated this project flagged metadata entry as "the
longest part" of digitizing a CV. A lot of it is written in plain text right
next to the figure — scan rate, electrolyte, reference electrode, temperature
— so when the PDF has a text layer we can pre-fill those fields and let the
curator verify rather than type. Everything here is best-effort and clearly
marked auto-extracted; nothing overrides a value the user passed explicitly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz


_MINUSES = {"−": "-", "–": "-", "—": "-"}


def _clean(text: str) -> str:
    for a, b in _MINUSES.items():
        text = text.replace(a, b)
    return re.sub(r"[ \t]+", " ", text)


# scan rate: "50 mV/s", "50 mV s-1", "50mVs", "0.05 V/s"
_SCAN = re.compile(r"(\d+\.?\d*)\s*(m?V)\s*[/ ]?\s*s(?:\s*[-−]?1|ec)?", re.I)
# electrolyte: "0.1 M HClO4", "1 M NaOH", "0.5 M H2SO4"
_ELECTROLYTE = re.compile(r"(\d*\.?\d+)\s*M\s+([A-Z][A-Za-z0-9]*(?:\s?[A-Z][A-Za-z0-9]*)?)")
_REF = re.compile(r"(?:vs\.?\s*)?\b(RHE|Ag\s*/\s*AgCl|SCE|SHE|NHE|Hg\s*/\s*HgO)\b")
_TEMP = re.compile(r"(\d{2,3})\s*°?\s*C\b|room[- ]temperature|(\d{2,3})\s*K\b", re.I)
_CAPTION = re.compile(r"\b(?:Figure|Fig\.?|FIG\.?|Scheme)\s*(\d+)\b", re.I)


@dataclass
class FigureMetadata:
    scan_rate: str = ""
    electrolytes: list[str] = field(default_factory=list)
    reference_electrode: str = ""
    temperature: str = ""
    caption: str = ""
    source: str = "auto-extracted from PDF text (verify)"

    def as_dict(self) -> dict:
        d = {"note": self.source}
        if self.scan_rate:
            d["scanRate"] = self.scan_rate
        if self.electrolytes:
            d["electrolytes"] = self.electrolytes
        if self.reference_electrode:
            d["referenceElectrode"] = self.reference_electrode
        if self.temperature:
            d["temperature"] = self.temperature
        if self.caption:
            d["caption"] = self.caption
        return d

    @property
    def is_empty(self) -> bool:
        return not (self.scan_rate or self.electrolytes or self.reference_electrode
                    or self.temperature or self.caption)


def _normalize_scan_rate(value: str, unit: str) -> str:
    return f"{value} {unit.lower().replace('v', 'V')}/s"


def _extract_caption(text: str) -> str:
    m = _CAPTION.search(text)
    if not m:
        return ""
    start = m.start()
    # take up to the first sentence-ish boundary or ~300 chars
    chunk = _clean(text[start:start + 400]).strip()
    # cut at a double space / newline-run that separates caption from body
    cut = re.search(r"(?<=[.\)])\s{2,}", chunk)
    if cut:
        chunk = chunk[:cut.start() + 1]
    return chunk[:300]


def extract_figure_metadata(pdf_path: str, page_number: int,
                            axis_titles: tuple[str, str] | None = None
                            ) -> FigureMetadata:
    """Best-effort metadata from a page's text layer.

    ``axis_titles`` (x, y), if given (e.g. from :mod:`cvdigitize.autocalib`),
    are also scanned for a reference-electrode mention, since "E vs RHE" often
    lives only in the axis label.
    """
    doc = fitz.open(pdf_path)
    text = _clean(doc[page_number].get_text("text"))
    doc.close()

    meta = FigureMetadata()

    m = _SCAN.search(text)
    if m:
        meta.scan_rate = _normalize_scan_rate(m.group(1), m.group(2))

    seen = set()
    for conc, salt in _ELECTROLYTE.findall(text):
        entry = f"{conc} M {salt.strip()}"
        if entry.lower() not in seen:
            seen.add(entry.lower())
            meta.electrolytes.append(entry)
    meta.electrolytes = meta.electrolytes[:6]

    ref_hay = text
    if axis_titles:
        ref_hay = f"{axis_titles[0]} {axis_titles[1]} {text}"
    rm = _REF.search(ref_hay)
    if rm:
        meta.reference_electrode = re.sub(r"\s+", "", rm.group(1))

    tm = _TEMP.search(text)
    if tm:
        if tm.group(1):
            meta.temperature = f"{tm.group(1)} °C"
        elif tm.group(2):
            meta.temperature = f"{tm.group(2)} K"
        else:
            meta.temperature = "room temperature"

    meta.caption = _extract_caption(text)
    return meta
