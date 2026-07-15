"""Tests for the optional OCR module's safe-when-unavailable contract.

The active OCR path needs the Tesseract binary and is exercised on the user's
machine after install; here we only guarantee that, without it, nothing raises
and nothing is (mis)read — so OCR can never silently corrupt a calibration.
"""
import numpy as np

from cvdigitize import ocr


def test_available_returns_bool():
    assert isinstance(ocr.available(), bool)


def test_read_number_safe_when_unavailable(monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    crop = np.full((20, 40, 3), 255, dtype=np.uint8)
    value, conf = ocr.read_number(crop)
    assert value is None
    assert conf == 0.0


def test_read_axis_values_none_without_engine(monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    crops = {k: np.full((20, 40, 3), 255, np.uint8) for k in ("x_lo", "x_hi", "y_lo", "y_hi")}
    x_pair, y_pair = ocr.read_axis_values(crops)
    assert x_pair is None and y_pair is None


def test_read_axis_values_requires_both_labels(monkeypatch):
    # even if single reads "succeed", a missing partner axis yields None
    monkeypatch.setattr(ocr, "read_number", lambda c: (1.0, 0.9))
    assert ocr.read_axis_values({"x_lo": np.zeros((2, 2, 3), np.uint8)}) == (None, None)
