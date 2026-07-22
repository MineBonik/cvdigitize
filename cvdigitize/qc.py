"""Eyeball QC for an extracted curve: transparent superimpose + one-click handoff.

Automatic extraction is usually right, but the honest way to trust a curve is to
*see* it lying on the original ink. A baked overlay PNG can't be judged well —
you can't tell a perfect trace from a near-miss because you can't take the line
away. So for each extracted panel we write:

  * ``panel.png``          — the bare figure render (also what trace_assist loads),
  * ``curve_overlay.png``  — a TRANSPARENT-background PNG, same pixel size, holding
                             only the extracted curve(s); drop it over the figure
                             in any image editor / slide to check the fit,
  * ``check.html``         — a self-contained viewer: the figure with the curve
                             superimposed and an **opacity slider** to fade the
                             trace in and out over the real ink (the by-eye test),
                             plus an **Open in trace_assist** button that loads this
                             exact panel into ``tools/trace_assist.html`` so a bad
                             auto-trace can be re-drawn by hand in seconds.

Rough in, precise out — and now, *verifiable* in one glance.
"""
from __future__ import annotations

import base64
import os

import numpy as np


# QC overlays deliberately DON'T use the curve's own colour — you're checking a
# trace against ink of the *same* colour, so a matching overlay is invisible when
# faded (a black trace over black ink tells you nothing). Use a vivid, rarely-in-
# figures palette, one distinct hue per curve, so every trace stands out over any
# ink and multiple curves stay distinguishable.
_QC_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
               "#e6009a", "#008080", "#9a6324", "#808000", "#000075"]


def _qc_hex(i: int) -> str:
    return _QC_PALETTE[i % len(_QC_PALETTE)]


def _png_data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def _save_panel_png(path: str, panel_rgb: np.ndarray) -> None:
    import cv2
    cv2.imwrite(path, cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR))


def _save_transparent_overlay(path: str, size_wh, curves, lw: float = 2.0) -> None:
    """Draw curves on a fully transparent canvas at native pixel size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, h = size_wh
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(h, 0)   # image coords: y down
    ax.axis("off")
    for i, cu in enumerate(curves):
        xy = np.asarray(cu["xy"], float)
        if len(xy) >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=_qc_hex(i), lw=lw,
                    solid_capstyle="round")
    fig.savefig(path, dpi=dpi, transparent=True)
    plt.close(fig)
    # matplotlib rounds figsize*dpi, so the PNG can be ±1 px off the panel;
    # force the exact size so curve_overlay.png registers pixel-for-pixel.
    import cv2
    out = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if out is not None and out.shape[:2] != (h, w):
        cv2.imwrite(path, cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST))


def _svg_polylines(curves) -> str:
    out = []
    for i, cu in enumerate(curves):
        xy = np.asarray(cu["xy"], float)
        if len(xy) < 2:
            continue
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        out.append(f'<polyline points="{pts}" fill="none" '
                   f'stroke="{_qc_hex(i)}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
    return "\n".join(out)


_GRADE_COLOR = {"good": "#2ecc71", "fair": "#f1c40f", "poor": "#e74c3c"}


def _curve_ink_mask(panel_rgb: np.ndarray, rgb):
    """The ink a curve of colour ``rgb`` was traced from — its own colour only.

    Scoring against the curve's OWN colour (not all panel ink) is what makes the
    check bite on a multi-panel figure: a chord that leaves the curve's colour —
    even if it happens to cross the schematic or a neighbouring panel — reads as
    off-ink. Dark/near-black curves use the brightness mask; colour curves use a
    hue band around their mean colour."""
    from .raster_extract import mask_dark_curve
    import cv2
    if rgb is None or max(rgb) < 0.35:                 # dark / black curve
        return mask_dark_curve(panel_rgb)
    target = np.uint8([[[int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]]])
    th = float(cv2.cvtColor(target, cv2.COLOR_RGB2HSV)[0, 0, 0])
    hsv = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    dh = np.abs((hue - th + 90.0) % 180.0 - 90.0)      # OpenCV hue is 0..180
    return (dh <= 12.0) & (sat >= 0.25) & (val >= 0.15)


def _fidelity_for_curves(panel_rgb: np.ndarray, curves: list[dict]) -> list[dict]:
    """Reference-free fidelity per curve, each scored against its OWN-colour ink.

    Returns one dict per curve: {score, grade, n_defects, off_ink_spans}."""
    from .fidelity import ink_fidelity, grade as _grade
    out = []
    for cu in curves:
        try:
            ink = _curve_ink_mask(panel_rgb, cu.get("rgb"))
        except Exception:
            ink = None
        if ink is None or not ink.any():
            out.append({"score": None, "grade": "n/a", "n_defects": 0, "off_ink_spans": []})
            continue
        r = ink_fidelity(np.asarray(cu["xy"], float), ink)
        out.append({"score": r["score"], "grade": _grade(r["score"]),
                    "n_defects": r["n_defects"], "off_ink_spans": r["off_ink_spans"]})
    return out


def _svg_off_ink_spans(fids: list[dict]) -> str:
    """Red dashed markers over the chord (off-ink) spans that failed the check."""
    out = []
    for f in fids:
        for sp in f.get("off_ink_spans", []):
            (x0, y0), (x1, y1) = sp["start"], sp["end"]
            out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                       f'stroke="#e74c3c" stroke-width="3" stroke-dasharray="7 5"/>')
    return "\n".join(out)


def _badges_html(curves: list[dict], fids: list[dict]) -> str:
    rows = []
    for i, (cu, f) in enumerate(zip(curves, fids)):
        col = _qc_hex(i)
        s = f["score"]
        badge = "n/a" if s is None else f"{s:.0f}"
        gcol = _GRADE_COLOR.get(f["grade"], "#888")
        flag = " &#9888; re-trace" if f["grade"] == "poor" else ""
        rows.append(
            f'<div class="crow"><span class="dot" style="background:{col}"></span>'
            f'<span class="cname">{cu.get("name", "curve")}</span>'
            f'<span class="score" style="background:{gcol}">{badge}</span>'
            f'<span class="flag">{flag}</span></div>')
    return "\n".join(rows)


_CHECK_TMPL = """<!doctype html>
<meta charset="utf-8">
<title>QC — {title}</title>
<style>
  body{{font:14px/1.5 system-ui,sans-serif;margin:0;background:#1a1a1a;color:#eee}}
  header{{padding:10px 16px;background:#111;position:sticky;top:0;display:flex;
    gap:18px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333}}
  header b{{color:#fff}}
  .ctl{{display:flex;gap:8px;align-items:center}}
  input[type=range]{{width:160px}}
  a.fix{{background:#2ecc71;color:#012;font-weight:600;text-decoration:none;
    padding:7px 14px;border-radius:6px}}
  a.fix:hover{{background:#5fe0a0}}
  .body{{display:flex;gap:0;align-items:flex-start}}
  .stage{{padding:16px;flex:1;display:flex;justify-content:center;min-width:0}}
  aside{{width:230px;padding:14px;background:#141414;border-left:1px solid #333;
    align-self:stretch}}
  aside h3{{margin:0 0 8px;font-size:12px;text-transform:uppercase;color:#999}}
  .crow{{display:flex;align-items:center;gap:7px;padding:3px 0;font-size:13px}}
  .dot{{width:12px;height:12px;border-radius:3px;flex:none}}
  .cname{{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .score{{color:#012;font-weight:700;border-radius:4px;padding:1px 7px;font-size:12px}}
  .flag{{color:#e74c3c;font-size:11px;white-space:nowrap}}
  .wrap{{position:relative;display:inline-block;max-width:100%;
    background:#fff;background-image:
      linear-gradient(45deg,#ddd 25%,transparent 25%,transparent 75%,#ddd 75%),
      linear-gradient(45deg,#ddd 25%,transparent 25%,transparent 75%,#ddd 75%);
    background-size:18px 18px;background-position:0 0,9px 9px}}
  img.bg{{display:block;max-width:100%;height:auto}}
  svg.ov{{position:absolute;inset:0;width:100%;height:100%}}
  .hint{{color:#999;font-size:12px;max-width:340px}}
  .worst{{font-weight:700}}
</style>
<header>
  <b>{title}</b>
  <span class="worst" style="color:{worst_color}">fidelity {worst_txt}</span>
  <div class="ctl"><label>curve <input id="op" type="range" min="0" max="100"
    value="100"></label></div>
  <div class="ctl"><label>figure <input id="bg" type="range" min="0" max="100"
    value="100"></label></div>
  <label class="ctl"><input type="checkbox" id="sp" checked> flag chords</label>
  <a class="fix" href="{trace_href}">Open in trace_assist &#9654;</a>
  <span class="hint">fade the curve over the ink: a good trace sits on the ink;
    red dashes mark chords across empty space — re-trace those in trace_assist</span>
</header>
<div class="body">
  <div class="stage"><div class="wrap">
    <img class="bg" id="bgimg" src="{panel_data}">
    <svg class="ov" id="ov" viewBox="0 0 {w} {h}" preserveAspectRatio="none">
{polylines}
    </svg>
    <svg class="ov" id="spans" viewBox="0 0 {w} {h}" preserveAspectRatio="none">
{spans}
    </svg>
  </div></div>
  <aside><h3>Curves &middot; ink-fidelity</h3>{badges}</aside>
</div>
<script>
  const ov=document.getElementById('ov'),op=document.getElementById('op'),
        bg=document.getElementById('bgimg'),bgs=document.getElementById('bg'),
        spans=document.getElementById('spans'),sp=document.getElementById('sp');
  op.addEventListener('input',()=>ov.style.opacity=op.value/100);
  bgs.addEventListener('input',()=>bg.style.opacity=bgs.value/100);
  sp.addEventListener('change',()=>spans.style.display=sp.checked?'block':'none');
</script>
"""


def write_qc(out_dir: str, panel_rgb: np.ndarray, curves: list[dict], *,
             tools_dir: str | None = None, panel_stem: str = "panel",
             title: str = "") -> dict:
    """Write panel.png, curve_overlay.png and check.html into ``out_dir``.

    ``curves`` is ``[{"name": str, "xy": (N,2) px, "rgb": (r,g,b)|None}, ...]`` in
    the panel's pixel coordinates. ``tools_dir`` locates ``trace_assist.html`` for
    the handoff link (defaults to the repo's ``tools/``).

    Also scores each curve's reference-free ink-fidelity and surfaces it: a
    green/amber/red badge per curve, red dashes over any chord (off-ink) spans,
    and a worst-curve headline. Returns the written paths plus ``fidelity`` (the
    per-curve score list) so the caller can record it in report.json / triage.
    """
    os.makedirs(out_dir, exist_ok=True)
    h, w = panel_rgb.shape[:2]

    panel_png = os.path.join(out_dir, f"{panel_stem}.png")
    overlay_png = os.path.join(out_dir, "curve_overlay.png")
    check_html = os.path.join(out_dir, "check.html")

    _save_panel_png(panel_png, panel_rgb)
    _save_transparent_overlay(overlay_png, (w, h), curves)
    fids = _fidelity_for_curves(panel_rgb, curves)

    if tools_dir is None:
        tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "tools")
    trace_html = os.path.join(tools_dir, "trace_assist.html")
    # href is resolved by the browser relative to check.html's own location;
    # the ?panel= value is resolved relative to trace_assist.html's location.
    href_to_tool = os.path.relpath(trace_html, out_dir).replace(os.sep, "/")
    panel_from_tool = os.path.relpath(panel_png, tools_dir).replace(os.sep, "/")
    label = title or panel_stem
    trace_href = (f"{href_to_tool}?panel={panel_from_tool}"
                  f"&name={label.replace(' ', '_')}")

    scored = [f["score"] for f in fids if f["score"] is not None]
    worst = min(scored) if scored else None
    worst_txt = "n/a" if worst is None else f"worst {worst:.0f}/100"
    from .fidelity import grade as _grade
    worst_color = "#999" if worst is None else _GRADE_COLOR[_grade(worst)]

    html = _CHECK_TMPL.format(
        title=label, w=w, h=h, polylines=_svg_polylines(curves),
        spans=_svg_off_ink_spans(fids), badges=_badges_html(curves, fids),
        worst_txt=worst_txt, worst_color=worst_color,
        panel_data=_png_data_url(panel_png), trace_href=trace_href)
    with open(check_html, "w", encoding="utf-8") as f:
        f.write(html)

    return {"panel_png": panel_png, "overlay_png": overlay_png,
            "check_html": check_html, "fidelity": fids}
