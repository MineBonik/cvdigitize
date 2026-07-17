"""Render the echemdb benchmark as a visual HTML report.

For every reference curve in each paper it overlays the echemdb hand-digitized
curve (black) against cvdigitize's best extraction (coloured by grade), so the
shape agreement is visible at a glance rather than reduced to one number. The
output is a single self-contained HTML file (images inlined as data URIs) that
opens locally and doubles as the body of a shareable Artifact.

Usage:
    .venv\\Scripts\\python.exe scripts\\benchmark_report.py <pdf_dir> <echemdb_root> [out.html]
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import (all_paper_curves, map_pdf_to_entry, chamfer_oriented,
                       _norm, _flip_y, chamfer, cv_plausible)
import glob
from cvdigitize.echemdb_ref import load_entry_references

GRADES = [(0.01, "excellent", "#16a34a"), (0.03, "good", "#2563eb"),
          (0.08, "fair", "#d97706"), (1e9, "poor", "#dc2626")]


def _grade(ch):
    for thr, name, color in GRADES:
        if ch < thr:
            return name, color
    return "poor", "#dc2626"


def _oriented(ref_norm, curve):
    a, b = _norm(curve), _norm(_flip_y(curve))
    return a if chamfer(ref_norm, a) <= chamfer(ref_norm, b) else b


def _paper_figure(refs, pool) -> tuple[str, list[dict]]:
    """A composite PNG (one subplot per reference) as a data URI, plus rows."""
    n = len(refs)
    ncol = min(5, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.5 * nrow),
                             squeeze=False)
    fig.patch.set_alpha(0.0)
    rows = []
    for k, r in enumerate(refs):
        ax = axes[k // ncol][k % ncol]
        rn = _norm(r.xy)
        # reference (ground-truth) as a soft grey band underneath ...
        ax.plot(rn[:, 0], rn[:, 1], "-", color="#94a3b8", lw=2.6, alpha=0.9,
                solid_capstyle="round")
        if pool:
            i = min(range(len(pool)), key=lambda j: chamfer_oriented(rn, pool[j]["xy"]))
            ch = chamfer_oriented(rn, pool[i]["xy"])
            use = _oriented(rn, pool[i]["xy"])
            name, color = _grade(ch)
            # ... the automatic extraction drawn on top, coloured by grade
            ax.plot(use[:, 0], use[:, 1], "-", color=color, lw=1.0, alpha=0.95)
            rows.append({"curve": r.curve_label, "figure": r.figure,
                         "chamfer": ch, "grade": name,
                         "page": pool[i]["page"], "branch": pool[i]["branch"]})
        else:
            ch, name, color = float("nan"), "none", "#999"
            rows.append({"curve": r.curve_label, "figure": r.figure,
                         "chamfer": None, "grade": "no match",
                         "page": None, "branch": None})
        ax.set_title(f"{r.figure} · {r.curve_label}\n{name}"
                     + (f"  {ch:.3f}" if pool else ""),
                     fontsize=8, color=color, fontweight=500)
        ax.set_xticks([]); ax.set_yticks([])
        ax.patch.set_alpha(0.0)
        for s in ax.spines.values():
            s.set_edgecolor("#cbd5e1")
            s.set_linewidth(0.6)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), rows


def _cached_pool(pdf: str):
    """all_paper_curves(pdf) cached to disk keyed by path+mtime, so re-rendering
    the report (design tweaks) is instant rather than a fresh 2.5-min extraction."""
    cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".report_cache")
    os.makedirs(cdir, exist_ok=True)
    key = hashlib.md5(f"{os.path.abspath(pdf)}:{os.path.getmtime(pdf)}".encode()).hexdigest()
    cpath = os.path.join(cdir, key + ".pkl")
    if os.path.isfile(cpath):
        with open(cpath, "rb") as f:
            return pickle.load(f)
    pool = all_paper_curves(pdf)
    with open(cpath, "wb") as f:
        pickle.dump(pool, f)
    return pool


def build(pdf_dir: str, echemdb_root: str, out_html: str):
    papers = []
    all_ch = []
    for pdf in sorted(glob.glob(os.path.join(pdf_dir, "*.pdf"))):
        entry = map_pdf_to_entry(pdf, echemdb_root)
        if entry is None:
            continue
        refs = load_entry_references(entry)
        if not refs:
            continue
        print(f"rendering {os.path.basename(entry)} ({len(refs)} curves)...")
        # same honesty gate as the benchmark: only CV-shaped candidates eligible
        pool = [c for c in _cached_pool(pdf) if cv_plausible(c["xy"])]
        img, rows = _paper_figure(refs, pool)
        chs = [r["chamfer"] for r in rows if r["chamfer"] is not None]
        all_ch += chs
        papers.append({
            "name": os.path.basename(entry),
            "pdf": os.path.basename(pdf),
            "img": img, "rows": rows, "n_pool": len(pool),
            "mean": float(np.mean(chs)) if chs else None,
        })

    papers.sort(key=lambda p: (p["mean"] is None, p["mean"] if p["mean"] is not None else 9))
    n = len(all_ch)
    mean = float(np.mean(all_ch)) if all_ch else 0.0
    pct_good = 100 * np.mean([c < 0.03 for c in all_ch]) if all_ch else 0
    pct_exc = 100 * np.mean([c < 0.01 for c in all_ch]) if all_ch else 0

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(_html(papers, n, len(papers), mean, pct_good, pct_exc))
    print(f"\nwrote {out_html}  ({n} curves, mean {mean:.4f})")


def _html(papers, n_curves, n_papers, mean, pct_good, pct_exc) -> str:
    css = """
<style>
  :root{
    --bg:#f7f8fa;--fg:#161a20;--muted:#5a6472;--card:#ffffff;--line:#e3e7ee;
    --accent:#0e8f86;--accent-soft:#0e8f8618;--ref:#94a3b8;
    --mono:"SFMono-Regular",ui-monospace,"Cascadia Mono",Consolas,monospace}
  @media (prefers-color-scheme:dark){:root{
    --bg:#0f1216;--fg:#e7ebf0;--muted:#96a0af;--card:#171b21;--line:#252b34;
    --accent:#2dd4bf;--accent-soft:#2dd4bf1f}}
  :root[data-theme=dark]{--bg:#0f1216;--fg:#e7ebf0;--muted:#96a0af;--card:#171b21;
    --line:#252b34;--accent:#2dd4bf;--accent-soft:#2dd4bf1f}
  :root[data-theme=light]{--bg:#f7f8fa;--fg:#161a20;--muted:#5a6472;--card:#ffffff;
    --line:#e3e7ee;--accent:#0e8f86;--accent-soft:#0e8f8618}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:40px 22px 72px}
  .eyebrow{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
           color:var(--accent);font-weight:600;margin:0 0 10px}
  h1{font-size:27px;line-height:1.2;margin:0 0 6px;text-wrap:balance;letter-spacing:-.01em}
  .sub{color:var(--muted);margin:0 0 26px;max-width:68ch}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
        gap:12px;margin:0 0 26px}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px 18px}
  .kpi b{display:block;font-size:27px;line-height:1.1;font-variant-numeric:tabular-nums;
         letter-spacing:-.02em}
  .kpi.hl b{color:var(--accent)}
  .kpi span{color:var(--muted);font-size:12px;display:block;margin-top:3px}
  .legend{display:flex;gap:18px;flex-wrap:wrap;align-items:center;
          color:var(--muted);font-size:12.5px;margin:0 0 26px;padding:12px 16px;
          background:var(--accent-soft);border-radius:10px}
  .dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:middle}
  .bar{display:inline-block;width:16px;height:4px;border-radius:2px;margin-right:6px;
       vertical-align:middle;background:var(--ref)}
  .paper{background:var(--card);border:1px solid var(--line);border-radius:14px;
         padding:18px 18px 8px;margin:0 0 18px}
  .phead{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
  .phead h2{font-size:16px;margin:0;letter-spacing:-.01em}
  .phead .meta{color:var(--muted);font-size:12.5px;font-family:var(--mono);
               font-variant-numeric:tabular-nums}
  .imgwrap{overflow-x:auto;margin:10px 0 4px}
  .imgwrap img{max-width:100%;height:auto;display:block}
  .note{color:var(--muted);font-size:13px;max-width:74ch;margin-top:26px}
  code{background:var(--accent-soft);color:var(--fg);padding:1.5px 6px;border-radius:5px;
       font-size:12.5px;font-family:var(--mono)}
</style>"""
    kpis = f"""
  <div class="kpis">
    <div class="kpi"><b>{n_curves}</b><span>reference curves matched</span></div>
    <div class="kpi"><b>{n_papers}</b><span>real Pt papers</span></div>
    <div class="kpi hl"><b>{mean:.3f}</b><span>mean shape error (Chamfer)</span></div>
    <div class="kpi hl"><b>{pct_good:.0f}%</b><span>good-or-better &lt;0.03</span></div>
    <div class="kpi hl"><b>{pct_exc:.0f}%</b><span>excellent &lt;0.01</span></div>
  </div>"""
    legend = """
  <div class="legend">
    <span><span class="bar"></span>echemdb reference (hand-digitized)</span>
    <span><span class="dot" style="background:#16a34a"></span>excellent &lt;0.01</span>
    <span><span class="dot" style="background:#2563eb"></span>good &lt;0.03</span>
    <span><span class="dot" style="background:#d97706"></span>fair &lt;0.08</span>
    <span><span class="dot" style="background:#dc2626"></span>poor</span>
  </div>"""
    body = []
    for p in papers:
        meanstr = f"{p['mean']:.3f}" if p["mean"] is not None else "—"
        body.append(f"""
  <div class="paper">
    <div class="phead">
      <h2>{p['name']}</h2>
      <div class="meta">mean {meanstr} · {len(p['rows'])} curves · pool of {p['n_pool']}</div>
    </div>
    <div class="imgwrap"><img src="data:image/png;base64,{p['img']}" alt="{p['name']} overlays"></div>
  </div>""")
    return f"""{css}
<div class="wrap">
  <p class="eyebrow">Cyclic voltammetry · automated digitization</p>
  <h1>cvdigitize vs echemdb — visual benchmark</h1>
  <p class="sub">Each panel overlays echemdb's hand-digitized curve (grey) with cvdigitize's
     automatic extraction straight from the publisher PDF (coloured by shape-agreement grade).
     Closer overlap is better; error is calibration-independent normalised Chamfer distance.</p>
  {kpis}
  {legend}
  {''.join(body)}
  <p class="note">Method: each reference is parsed straight from the echemdb svgdigitizer SVG and
     matched to the best-fitting extracted curve pooled across the whole paper (vector + raster),
     by flip-invariant normalised-shape Chamfer. Reproduce with
     <code>scripts/benchmark.py --echemdb &lt;pdf_dir&gt; &lt;echemdb_root&gt;</code>.</p>
</div>"""


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[3] if len(sys.argv) > 3 else "data/out/_benchmark/index.html"
    build(sys.argv[1], sys.argv[2], out)
