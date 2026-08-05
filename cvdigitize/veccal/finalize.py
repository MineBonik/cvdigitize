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


def _tick_line(ticks: list[dict], *, min_ticks: int = 3, max_ticks: int = 12,
               fit_tol_frac: float = 0.02):
    """The straight line a figure's own tick labels imply, or ``None``.

    Returns ``(slope, intercept, value_span)`` mapping position -> value, but
    only when the ticks are trustworthy enough to judge a human's calibration
    against. They are rejected when:

    * there are too few to define a line, or so many that they cannot be read
      axis labels — a real axis carries a handful, and a run of dozens is the
      tick *detector* enumerating positions while OCR numbers them ``1, 2,
      3 …``. One corpus panel produced 60 such "ticks" whose values were
      simply their own index; trusting those would reject a calibration that
      matches the printed axis exactly.
    * their values are not monotonic in position, or do not themselves fall on
      a line. Ticks that disagree with *each other* cannot arbitrate.
    """
    pts = [(float(t["pos"]), float(t["value"])) for t in ticks
           if t.get("value") is not None and t.get("pos") is not None]
    if not (min_ticks <= len(pts) <= max_ticks):
        return None
    pts.sort()
    pos = np.array([p for p, _ in pts], float)
    val = np.array([v for _, v in pts], float)
    if len(np.unique(pos)) != len(pos):
        return None
    d = np.diff(val)
    if not (np.all(d > 0) or np.all(d < 0)):        # not monotonic -> unusable
        return None
    span = float(val.max() - val.min())
    if span <= 0:
        return None
    slope, intercept = np.polyfit(pos, val, 1)
    if np.max(np.abs((slope * pos + intercept) - val)) > fit_tol_frac * span:
        return None                                  # ticks disagree with themselves
    return float(slope), float(intercept), span


def tick_disagreement(cal: Calibration, unit: dict, *,
                      tol_frac: float = 0.02) -> list[str]:
    """Ways this calibration contradicts the figure's own printed tick labels.

    The overlay a human confirms cannot catch this: curves are drawn *through*
    the calibration being checked, so they land on the ink whether it is right
    or wrong. Only the ticks are independent evidence. Three real corpus
    errors were invisible until this ran — a panel calibrated against the
    right-hand axis (every current exactly 2x too large), one whose anchor was
    clicked 12 pt away from the tick it meant (every potential shifted by
    0.075 V), and a stacked sub-panel only ~96 pt wide whose "ticks" were
    detected 130+ pt outside its own frame — bled in from a neighbouring
    plot's axis, so the values were current readings masquerading as
    potential. That last case is why ticks are filtered to the panel's own
    ``frame_pdf`` below: evidence from a different axis is not evidence.

    Silence is not proof: when the ticks are unreadable this returns nothing,
    because "I cannot check" must not masquerade as "I checked and it is fine".
    """
    problems = []
    frame = unit.get("frame_pdf")
    axes = (
        ("E", unit.get("x_ticks") or [], cal.x1, cal.ex1, cal.x2, cal.ex2,
         (frame[0], frame[2]) if frame else None),
        ("j", unit.get("y_ticks") or [], cal.y1, cal.jy1, cal.y2, cal.jy2,
         (frame[1], frame[3]) if frame else None),
    )
    for name, ticks, p1, v1, p2, v2, bounds in axes:
        if bounds is not None:
            lo, hi = bounds
            margin = 0.05 * abs(hi - lo) + 3.0     # a few pt slack for tick marks
            ticks = [t for t in ticks if t.get("pos") is None
                    or lo - margin <= t["pos"] <= hi + margin]
        line = _tick_line(ticks)
        if line is None or p1 == p2:
            continue
        slope, intercept, span = line
        cal_slope = (v2 - v1) / (p2 - p1)
        positions = np.array([float(t["pos"]) for t in ticks
                              if t.get("value") is not None], float)
        predicted = v1 + (positions - p1) * cal_slope
        expected = slope * positions + intercept
        err = float(np.max(np.abs(predicted - expected)))
        if err > tol_frac * span:
            ratio = cal_slope / slope if slope else float("inf")
            hint = (f" (that is {ratio:.3g}x the printed scale — check you used "
                    f"the correct axis)" if abs(ratio - 1) > 0.25 else
                    " (check the anchor snapped to the tick you meant)")
            problems.append(
                f"the {name} calibration disagrees with the figure's own tick "
                f"labels by up to {err:.4g} ({100 * err / span:.0f}% of the "
                f"{span:.4g} they span){hint}")
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

    # NOTE: `tick_disagreement` deliberately does NOT gate saving. It compares
    # the human's anchors against detected ticks, but the anchors were being
    # snapped onto those same ticks by the UI — so it largely checked the
    # detector against itself, and fired on panels where the detector, not the
    # human, was wrong. It stays available for offline auditing of a finished
    # run (where a person reviews the flags with the figure in hand); it is not
    # evidence strong enough to refuse a chemist's reading of their own plot.

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
