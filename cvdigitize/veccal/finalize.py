"""Turn a confirmed panel calibration into echemdb datapackages on disk.

:mod:`cvdigitize.veccal.scan` leaves each curve as an ordered loop in PDF
points. This is the other end: given the calibration a human just confirmed,
apply it, resample, and write one CSV + frictionless JSON + YAML per curve —
the same format :mod:`cvdigitize.package` produces for the rest of the tool, so
these files are interchangeable with the main CLI's output.

Writing happens per panel, the moment the user presses save, so a session that
is interrupted keeps everything already confirmed.
"""
from __future__ import annotations

import os

import numpy as np

from ..calibrate import Calibration
from ..package import CurveMeta, write_datapackage
from ..postprocess import loop_metrics, resample_arclength, resample_uniform_potential


def calibration_from_payload(payload: dict) -> Calibration:
    """Build a :class:`Calibration` from the browser's calibration form.

    Anchors arrive in PDF points (the browser converts back from crop pixels),
    which is the space the stored curve loops use, so no rescaling happens here.
    """
    return Calibration(
        x1=float(payload["x1"]), ex1=float(payload["ex1"]),
        x2=float(payload["x2"]), ex2=float(payload["ex2"]),
        y1=float(payload["y1"]), jy1=float(payload["jy1"]),
        y2=float(payload["y2"]), jy2=float(payload["jy2"]),
        x_unit=(payload.get("x_unit") or "V").strip(),
        y_unit=(payload.get("y_unit") or "uA/cm2").strip(),
        x_label=(payload.get("x_label") or "E").strip(),
        y_label=(payload.get("y_label") or "j").strip(),
    )


def validate_calibration(cal: Calibration) -> list[str]:
    """Human-readable reasons this calibration cannot be applied, if any."""
    problems = []
    if cal.x1 == cal.x2:
        problems.append("the two E anchors sit at the same position")
    if cal.y1 == cal.y2:
        problems.append("the two j anchors sit at the same position")
    if cal.ex1 == cal.ex2:
        problems.append("both E values are the same number")
    if cal.jy1 == cal.jy2:
        problems.append("both j values are the same number")
    for label, value in (("E", (cal.ex1, cal.ex2)), ("j", (cal.jy1, cal.jy2))):
        if any(not np.isfinite(v) for v in value):
            problems.append(f"a {label} value is not a finite number")
    return problems


def _resample(data: np.ndarray, n: int, mode: str) -> np.ndarray:
    if n <= 0:
        return data
    if mode == "uniform-E":
        return resample_uniform_potential(data, n_per_branch=max(2, n // 2))
    return resample_arclength(data, n=n)


def save_panel(unit: dict, calib_payload: dict, out_dir: str, *,
               resample: int = 1000, resample_mode: str = "arclength",
               scan_rate: str = "", yaml: bool = True) -> dict:
    """Write every curve of one panel as a calibrated datapackage.

    Returns ``{"out_dir", "curves": [...], "warnings": [...]}``. Raises
    ``ValueError`` when the calibration is unusable, so the browser can show the
    reason instead of silently writing mis-scaled data.
    """
    cal = calibration_from_payload(calib_payload)
    problems = validate_calibration(cal)
    if problems:
        raise ValueError("Calibration is not usable: " + "; ".join(problems) + ".")

    os.makedirs(out_dir, exist_ok=True)
    paper = unit.get("paper_meta") or {}
    figure_meta = dict(unit.get("figure_meta") or {})
    rate = (scan_rate or figure_meta.get("scanRate") or "").strip()

    written, warnings = [], []
    used_names: set[str] = set()
    for curve in unit.get("curves", []):
        loop = np.asarray(curve.get("loop_pdf") or [], dtype=float)
        if loop.ndim != 2 or len(loop) < 2:
            warnings.append(f"{curve.get('name', '?')}: no usable geometry, skipped")
            continue

        data = _resample(cal.apply(loop), resample, resample_mode)

        # The user may have edited names into a collision; keep both files.
        name = (curve.get("name") or "curve").strip() or "curve"
        base, n = name, 2
        while name in used_names:
            name, n = f"{base}-{n}", n + 1
        used_names.add(name)

        sample = (curve.get("sample") or "").strip()
        colour = curve.get("color_label") or curve.get("color") or ""
        meta = CurveMeta(
            name=name,
            figure=unit.get("figure") or (f"panel {unit['panel']}" if unit.get("panel") else ""),
            curve=f"{colour}: {sample}" if sample else colour,
            scan_rate=rate,
            x_label=cal.x_label, x_unit=cal.x_unit,
            y_label=cal.y_label, y_unit=cal.y_unit,
            source_pdf=unit.get("pdf", ""),
            method="digitized",
            comment=("extracted from the PDF's vector geometry; axis calibration "
                     "confirmed by a human in cvdigitize vector-calibrate"),
            extracted={**figure_meta, **({"paper": paper} if paper else {})},
        )
        paths = write_datapackage(out_dir, data, meta, yaml=yaml)
        written.append({
            "name": name, "color": curve.get("color"), "color_label": colour,
            "sample": sample, "n_points": int(len(data)),
            "loopiness": round(float(loop_metrics(data)["loopiness"]), 3),
            "files": {k: os.path.relpath(v, out_dir) for k, v in paths.items()},
        })

    if not written:
        raise ValueError("No curve in this panel had usable geometry.")
    return {"out_dir": out_dir, "curves": written, "warnings": warnings}
