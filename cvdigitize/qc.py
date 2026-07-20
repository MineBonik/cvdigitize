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


_CHECK_TMPL = """<!doctype html>
<meta charset="utf-8">
<title>QC — {title}</title>
<style>
  body{{font:14px/1.5 system-ui,sans-serif;margin:0;background:#1a1a1a;color:#eee}}
  header{{padding:10px 16px;background:#111;position:sticky;top:0;display:flex;
    gap:18px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333}}
  header b{{color:#fff}}
  .ctl{{display:flex;gap:8px;align-items:center}}
  input[type=range]{{width:180px}}
  a.fix{{background:#2d7;color:#012;font-weight:600;text-decoration:none;
    padding:7px 14px;border-radius:6px}}
  a.fix:hover{{background:#5fb}}
  .stage{{padding:16px;display:flex;justify-content:center}}
  .wrap{{position:relative;display:inline-block;max-width:100%;
    background:#fff;background-image:
      linear-gradient(45deg,#ddd 25%,transparent 25%,transparent 75%,#ddd 75%),
      linear-gradient(45deg,#ddd 25%,transparent 25%,transparent 75%,#ddd 75%);
    background-size:18px 18px;background-position:0 0,9px 9px}}
  img.bg{{display:block;max-width:100%;height:auto}}
  svg.ov{{position:absolute;inset:0;width:100%;height:100%}}
  .hint{{color:#999;font-size:12px}}
</style>
<header>
  <b>{title}</b>
  <div class="ctl"><label>curve <input id="op" type="range" min="0" max="100"
    value="100"></label></div>
  <div class="ctl"><label>figure <input id="bg" type="range" min="0" max="100"
    value="100"></label></div>
  <a class="fix" href="{trace_href}">Open in trace_assist &#9654;</a>
  <span class="hint">fade the curve over the ink to judge the fit; if it's off,
    click Open in trace_assist and re-draw it</span>
</header>
<div class="stage"><div class="wrap">
  <img class="bg" id="bgimg" src="{panel_data}">
  <svg class="ov" id="ov" viewBox="0 0 {w} {h}" preserveAspectRatio="none">
{polylines}
  </svg>
</div></div>
<script>
  const ov=document.getElementById('ov'),op=document.getElementById('op'),
        bg=document.getElementById('bgimg'),bgs=document.getElementById('bg');
  op.addEventListener('input',()=>ov.style.opacity=op.value/100);
  bgs.addEventListener('input',()=>bg.style.opacity=bgs.value/100);
</script>
"""


def write_qc(out_dir: str, panel_rgb: np.ndarray, curves: list[dict], *,
             tools_dir: str | None = None, panel_stem: str = "panel",
             title: str = "") -> dict:
    """Write panel.png, curve_overlay.png and check.html into ``out_dir``.

    ``curves`` is ``[{"name": str, "xy": (N,2) px, "rgb": (r,g,b)|None}, ...]`` in
    the panel's pixel coordinates. ``tools_dir`` locates ``trace_assist.html`` for
    the handoff link (defaults to the repo's ``tools/``). Returns written paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    h, w = panel_rgb.shape[:2]

    panel_png = os.path.join(out_dir, f"{panel_stem}.png")
    overlay_png = os.path.join(out_dir, "curve_overlay.png")
    check_html = os.path.join(out_dir, "check.html")

    _save_panel_png(panel_png, panel_rgb)
    _save_transparent_overlay(overlay_png, (w, h), curves)

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

    html = _CHECK_TMPL.format(
        title=label, w=w, h=h, polylines=_svg_polylines(curves),
        panel_data=_png_data_url(panel_png), trace_href=trace_href)
    with open(check_html, "w", encoding="utf-8") as f:
        f.write(html)

    return {"panel_png": panel_png, "overlay_png": overlay_png, "check_html": check_html}
