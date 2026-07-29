# cvdigitize

Automated digitization of Cyclic Voltammetry (CV) curves from scientific PDFs
into `echemdb`-compatible data — replacing the manual svgdigitizer workflow.

Give it a paper PDF; it finds the CV figure (vector **or** scanned/rasterized),
detects each plot panel, separates the colour-coded curves, reconstructs each
closed loop, calibrates to real units, resamples to a clean even trace, pulls
experimental metadata (scan rate, electrolyte, reference electrode) from the
caption, and writes CSV + frictionless JSON + YAML datapackages.

> Key design choice: for vector figures we parse the PDF's **vector geometry**
> directly, so we already have the true curve points — no `svgdigitizer`/
> Inkscape round-trip needed. For rasterized figures (the whole plot flattened
> to a bitmap), a colour/brightness-based tracer recovers the curve from
> pixels instead. Either way the tool is self-contained (PyMuPDF + numpy +
> OpenCV + scikit-image) — no external services, no manual SVG editing.

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# just point it at a PDF — no subcommand needed
.\cvdigitize.bat mypaper.pdf
```

That auto-picks a page, extracts whatever curves it can, and writes a
normalised CSV plus an HTML report (`data\out\mypaper\index.html`) you can open
right away. Add calibration once you know what's in the figure (see Usage).

`cvdigitize.bat` / `cvdigitize.ps1` are thin launchers so you don't have to
type the venv path every time — `cvdigitize.bat info mypaper.pdf` instead of
`.venv\Scripts\python.exe -m cvdigitize info mypaper.pdf`. Both do the same
thing; use whichever your shell prefers.

### Working through a folder of vector papers — the recommended route

If your papers are native-vector PDFs (the figure is real path geometry, not a
picture), this is the fastest path from a folder to citable data:

```powershell
.\cvdigitize.bat vector-calibrate --in data\literature_Vladislav
```

It scans every PDF for vector CV panels, extracts each curve mathematically,
then opens a browser page that walks the panels **one at a time**. For each
panel you see the original figure with the extracted curves drawn on top, the
calibration already filled in wherever the tool could read the axis numbers
(real text tick labels, else OCR of the detected ticks), and one editable name
per curve. Press **Save & next** and that panel's CSV + JSON + YAML are written
immediately, so stopping half way loses nothing — re-run the same command to
resume.

Calibration is **per panel, not per curve**: every curve inside a panel shares
its axes, so you confirm the numbers once and all of its curves come out in real
units. Where the tool could not read the axis, click a tick on each axis (clicks
snap to detected ticks) and type its value.

Deliberately narrower than `batch`: it only ever touches vector figures, where
extraction is exact and needs no hand-tracing. For scanned/raster figures use
`studio` instead.

The first run scans the folder, which is the only slow part; after that the
browser reads one panel at a time. Papers are scanned in parallel across cores
(`--workers 1` to serialise, e.g. when profiling).

## Status

| Milestone | What it does | State |
|-----------|--------------|-------|
| **M0** | Vector PDF → auto-split each CV curve by stroke colour, with **automatic panel detection** (each plot's axes frame; handles grids and several figures per page) | done & verified |
| **M1** | Loop ordering, axis calibration, arc-length/uniform-E resampling, echemdb packaging, CLI | done & verified |
| **M2** | Auto vector/raster classify + colour/brightness-trace for rasterized figures, incl. multi-panel auto-detection, tiling-strip merging, closed **and L-shaped (despined)** axes, automatic tick-mark finding, and **multi-colour curve splitting by hue** (crossing-tolerant) | done & verified |
| **M3** | Automatic calibration | done for PDFs with live figure text; assisted (4 typed numbers) everywhere else |
| **V** | `vector-calibrate`: vector-only folder run — auto-detect every vector CV panel, name its curves, confirm each panel's axes in the browser, write echemdb datapackages as you go (incl. DOI/title/journal in the YAML) | done & verified |

### Calibration — three tiers, tried in order

1. **Auto (zero input).** Many vector figures keep their axis tick labels as
   real, selectable text (matplotlib output does by default). Each label is
   centred on its tick, so a linear fit through (label position, label value)
   *is* the calibration — detected, fitted, and applied automatically,
   including the axis-title text as the unit hint. Verified on ground truth:
   auto-calibrated output matches the known curves at Chamfer 0.004–0.007
   with no flags at all.
2. **Assisted (type 4 numbers).** Figures with outlined/vectorized text (ACS,
   Elsevier production PDFs...) have no text layer. The tool then detects the
   axes frame and tick-mark pixel positions itself (handling inward-pointing
   ticks and filtering unlabeled minor ticks by stroke length), crops zoomed
   images of the four outermost tick labels into `calib_helper.png`, and you
   rerun with `--x-ticks=A,B --y-ticks=C,D`. Verified on the rizo paper: all
   7 curves within Chamfer 0.006 of the manual reference in real units.
3. **Manual JSON** (`--calibration file.json`) — full control fallback; also
   what you'd use for exotic layouts. `--no-autocalib` disables tier 1.

Where Tesseract is installed, tier 2's four numbers can also be pre-filled by
OCR — strictly, and only as a suggestion a human confirms. See "OCR of tick
labels" below for why it is never treated as authoritative.

### Metadata (the thread's "longest part")

When the PDF has a text layer, the caption and axis titles are parsed for
**scan rate, electrolyte(s), reference electrode, temperature** and the caption
itself. These pre-fill the datapackage (scan rate goes straight into
`scanRate`; the rest into an `autoExtracted` block) so a curator confirms
rather than types. Everything is flagged auto-extracted and never overrides a
value you pass with `--scan-rate` etc. On the rizo paper this recovers
`50 mV/s`, `0.1 M HClO4` / `0.1 M NaOH`, and `RHE` with no input.

Each curve is also named by **what its colour is**, mapped either from a
caption legend ("black line: 0.10 M HClO4, red line: …", scoped per panel) or
an in-plot text-layer legend (colour swatch + adjacent label) — so the echemdb
`curve` field says `black: 0.10 M HClO4`, not just `black`.

### Picking the CV out of a busy figure

Every curve gets a **loop score** (a closed CV loop encloses area; a
schematic / micrograph / Nyquist arc / spectrum does not). `--cv-only` keeps
just the CV-like panels of a multi-panel figure, and `batch` chooses each
paper's most CV-like figure page automatically. This is a triage signal, not a
perfect classifier — a genuinely peak-shaped CV can score low — so it never
deletes data silently, only selects/orders.

### OCR of tick labels — optional, and never authoritative

A self-contained template matcher was prototyped and rejected: it isn't reliable
enough even on matplotlib's own font, and a mis-read tick would silently corrupt
the calibration. So the baseline everywhere is safe by construction — the tool
auto-detects tick *positions* and crops zoomed label images, and you read two
numbers per axis.

On top of that, `cvdigitize/ocr.py` will use **Tesseract if it is installed**
(`ocr.available()` reports whether it is; nothing else changes if it isn't). It
is deliberately strict: an axis is only read when *both* of its outer labels
come back above a confidence floor, so it either reads an axis fully or not at
all — it never half-calibrates.

`vector-calibrate` uses it for **pre-fill only**. The numbers land in the form
next to a picture of the curve on its own axes, and a human confirms them before
anything is written, which is what makes an OCR mistake visible rather than
silent. On the reference corpus most figures outline their axis text, so this is
the difference between "type four numbers" and "check four numbers".

## Verified results

**echemdb corpus benchmark — 10 real Pt papers, 48 reference curves.** The tool
is run on the publisher PDFs and scored against the echemdb curators' own
hand-digitized figures (parsed straight from their svgdigitizer SVGs — see
[`cvdigitize/echemdb_ref.py`](cvdigitize/echemdb_ref.py)). Each reference is
matched to the best-fitting extracted curve pooled across the whole paper
(vector + raster), by calibration-independent normalised-shape Chamfer. Run it
yourself:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --echemdb <pdf_dir> `
    data\echemdb\electrochemistry-data\literature\svgdigitizer
```

Candidate curves pass a **CV-plausibility gate** (arc-efficiency + self-crossing
bounds) before matching, so a reference whose real figure extracted badly
reports "no acceptable curve" rather than flattering itself against a logo /
watermark / rule / text-blob — the aggregate below is the *honest* one.

Aggregate over all 48 curves: **mean normalised Chamfer 0.026 · 58 %
good-or-better (<0.03) · 33 % excellent (<0.01)**, every reference matched.

| Paper (echemdb key) | figure kind | mean Chamfer | best curve |
|---|---|---|---|
| briega-martos 2021 (15 curves) | vector, multi-panel | **0.010** | 0.0025 excellent |
| clavilier 1980 (preparation) | scanned, figure-in-text | **0.009** | excellent |
| abe 2019 (3×3 panel grid) | raster, colour | **0.015** | 0.0085 excellent |
| garcia-araez 2011 | raster, colour | **0.022** | 0.0030 excellent |
| gomez-marin 2012 | raster, colour | **0.033** | 0.0124 good |
| sandbeck 2019 | raster, solid+dashed | **0.035** | 0.0159 good |
| markovic 1996 | scanned, scale-bar cal. | **0.043** | — |
| schnaidt 2017 | raster | **0.043** | — |
| briega-martos 2020 (7 curves) | raster, colour | 0.045 | — |
| clavilier 1980 (role) | scanned, classic axes | 0.061 | — |

Reads: cleanly-drawn vector figures are near-perfect; colour-coded raster
figures extract well. The big lever is **junction-aware strand decomposition**
([`cvdigitize/strands.py`](cvdigitize/strands.py)): a skeleton of overlapping
strokes is a *graph*, and the naive tracer dies at the first junction where a
solid curve is touched by its dashed sibling, crossed by another curve, or
crosses a plot axis. We collapse the skeleton to nodes+edges and pair branches
by straight-through tangent continuity, so each stroke is followed intact
across crossings (sharp CV peaks, which are not junctions, are preserved) —
axis-shaped strands are then classified and dropped, rather than erasing axis
pixels first and re-stitching the fragments (which shortcuts across the gap).
Applied everywhere in the raster pipeline — colour/dark masks, full-page-scan
figures, and bare-axes classic figures alike — this is what makes sandbeck's
solid+dashed panels extractable at all (0.071 → 0.035, they had been matching
junk) and takes clavilier-preparation, a curve that starts the session buried
in a full page of body text, from **0.23 → 0.009**, now the single
best-scoring paper in the benchmark. (Measured and rejected as non-improvements:
tangent-continuity at the *stitching* level instead of the junction level —
net-neutral; higher render zoom — net-negative.)

**Real paper — `rizo_2025_analysis_351` (ACS Electrochem 2025), vector figure.**
All 7 panel-(a) curves auto-extracted from the raw PDF match Vladislav's
hand-digitized reference:
- shape match (normalised Chamfer): **0.0007–0.0068**
- with one global calibration, cleaned curves overlay the reference in real
  units (V vs RHE, µA cm⁻²)
- point spacing **~10× more uniform** (CoV 2.4–8.4 → 0.26–0.69) with **zero**
  duplicate-potential points — the "data is not raw" issue Albert flagged, fixed.
- `--resample-mode uniform-E` goes further: it resamples each scan branch on a
  uniform **potential** grid (CoV → 0, potentiostat-like), exactly the "equal
  spacing of points" Albert described for raw data, while keeping the loop
  two-valued (anodic + cathodic).

**Synthetic vector PDF (known ground truth).** 3 curves, different layout/axes:
mean normalised Chamfer **0.005** vs truth.

**Real paper — arXiv:2305.06810, page 6, Figure 1c, rasterized composite figure.**
This whole 5-panel figure is a single flattened bitmap (confirmed: 0 vector
paths, 1 embedded image) — no tool that only reads vector geometry can touch
it. The raster pipeline: auto-detects both plot panels on the page (dropping
false-positive rectangles from curve-thickness and shared/collinear panel
borders), auto-finds axis tick pixel positions, isolates the curve from its
rainbow gradient fill by brightness (not colour — anti-aliasing blends the two
in HSV saturation but not value), skeletonizes, and orders the shuffled pixels
into a single closed loop. Calibrated result: **E ∈ [-0.80, 0.29] V, j ∈
[-52.6, 25.6] µA** — reads off the original printed axes essentially exactly.

Handled real-data quirks: the same logical curve drawn in slightly different
RGB across panels (panel-first grouping); shuffled/unordered path segments
(greedy stitching); legend colour swatches and stray specks (connected-
component filtering, both in vector paths and raster pixel blobs); light grid
lines (saturation filter); collinear borders of stacked panels coincidentally
passing as one big rectangle (containment filtering); tick marks bleeding into
frame-corner detection (corner-margin exclusion).

**Real paper — Gámez et al., Electrochimica Acta 512 (2025), vector figure with
a different PDF encoding.** This PDF draws curves as thin *filled* ribbon
polygons (`color=None`, only `fill` set) instead of stroked paths, and — more
subtly — renders body text with *stroke* colour, and axis borders/ticks as thin
filled rectangles, both black just like the real curve. All three land in the
same colour bucket as legitimate data unless explicitly excluded. Fixed by:
grouping by fill colour only when a colour has zero stroke data on the page
(so glyph shapes never contaminate an already-good stroked curve); recognising
axis-aligned thin rectangles (frame borders, via either the `re` opcode or 4
`l` ops) and narrow-but-tall shapes (stroked text columns) by their geometry
rather than by distance heuristics during stitching; excluding tiny (≤3-point,
≤12pt) isolated segments (tick marks) that skew a colour group's bounding box
without visibly deforming the trace. Result: all 5 curves across the page's two
stacked sub-figures extract cleanly with zero cross-contamination — verify via
`overlay.png` (no independent reference digitization exists for this paper).

*Design lesson worth keeping in mind*: an early attempt fixed the same bug
class by having the loop-stitcher (`order_curve`) refuse to bridge
"implausibly large" gaps. It seemed to work, but a single distance threshold
could not reliably separate a legitimate large gap inside one real curve from
a truly unrelated shape — it silently truncated good rizo curves down to 5-10%
of their points. Reverted in favour of the shape/colour filters above, which
fix contamination at its source instead of guessing distances after the fact.

## Setup (Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

```powershell
# 0) simplest possible: auto page, auto panels, auto calibration
cvdigitize mypaper.pdf
#   Panels are detected automatically (each plot's axes frame) — a 4-panel
#   figure or two stacked figures on one page come out as separate panel_*
#   folders with no flags. If the figure keeps its tick labels as text, every
#   panel is calibrated to real units automatically.

# 1) inspect: which pages hold vector CV figures vs rasterized ones?
cvdigitize info "data\in\paper.pdf"

# 2) VECTOR figure, all panels auto-detected (normalised unless text-calibratable)
cvdigitize extract "data\in\paper.pdf"

# 3) VECTOR, one panel + real units from typed tick values (outlined-text PDFs).
#    --x-ticks/--y-ticks describe ONE panel, so pick it with --panel:
cvdigitize extract "data\in\rizo_2025_analysis_351.pdf" --panel a `
    --x-ticks="0.2,0.8" --y-ticks="150,-150" --x-unit "V vs RHE" --y-unit "uA/cm2" `
    --figure 1a --scan-rate "50 mV/s"
#    (read the four numbers off panel_a/calib_helper.png; or use --panels 2x2
#    for a fixed grid, or --calibration file.json for full manual control)

# 4) build a VECTOR calibration: render a page with a pixel grid to read axis anchors
cvdigitize grid "data\in\paper.pdf" --page 1

# 5) RASTER/scanned figure: list the plot panels cvdigitize can see
cvdigitize extract "data\in\paper.pdf" --page 5
#   -> prints how many panels were found, a numbered preview PNG, and — for each
#      panel — calib_helper_panelN.png: zoomed crops of the four outermost tick
#      LABELS so you can just read off the numbers. Tick pixel positions are
#      auto-detected; you only supply the two outermost tick VALUES per axis.

# 6) RASTER, calibrated (note the `=`, needed so the shell doesn't treat
#    a leading "-0.8" as another flag). Numbers come straight off the helper crops.
cvdigitize extract "data\in\paper.pdf" --page 5 --raster-panel 1 `
    --x-ticks="-0.8,0.2" --y-ticks="25,-50" --x-unit "V vs Ag/AgCl" --y-unit "uA"

# 7) keep only the CV panels of a multi-panel figure (skip SEM / spectra /
#    Nyquist / schematics by their low loop score)
cvdigitize extract "data\in\paper.pdf" --cv-only

# 8) SURVEY a whole folder of PDFs at once -> gallery sorted by CV-likeness
#    (each paper's most CV-like figure page is chosen automatically)
cvdigitize batch "data\corpus"
#   -> data\out\_batch\index.html : one card per paper (figure type, candidate
#      curves, loop-score confidence). Great for triaging many papers fast.
#      Recreate the sample corpus with: python scripts\fetch_corpus.py

# 9) VECTOR-ONLY folder run: every vector CV panel, calibrated by hand in the
#    browser, one panel at a time (see "Working through a folder" above).
#    Writes each panel's datapackages on save; re-run to resume where you left off.
cvdigitize vector-calibrate --in "data\literature_Vladislav"
#   --rescan       re-detect panels, keeping the status and names you entered
#   --out DIR      where curves land (default <work>\curves)
#   --workers N    parallel scan workers (default CPU count - 1; 1 = sequential)
```

(`cvdigitize` above = `cvdigitize.bat` / `.\cvdigitize.ps1`, or
`.venv\Scripts\python.exe -m cvdigitize` if you'd rather call Python directly.)

### Crop &amp; triage a messy paper first — recommended for real-world batches

A real literature folder is full of figures that **aren't** clean single-plot
CVs: multi-panel composites (a crystal-structure schematic next to 3 separate
voltammograms, each with its own axes), SEM images, schemes. Handing these
straight to frame auto-detection fails in two different ways — measured on a
real paper: the schematic's ball-and-stick shapes get misread as **176
spurious plot frames**, or the whole composite collapses into **one panel**
with one calibration wrongly shared across sub-plots that don't have the same
axes. Fix: a human crops each real plot by hand first, so detection only ever
sees one clean figure at a time.

```powershell
# 1) render whole pages — no panel auto-detection at all
.venv\Scripts\python.exe scripts\trace_guided.py pages data\papers -o data\out\pages

# 2) open tools\trace_assist.html, load that folder. In Crop mode, drag a box
#    around ONE real plot (skip the schematic / other panels entirely — crop
#    them separately). Alt+drag inside it to blank out a legend/text box that
#    sits on the axes (a real, measured contamination source — see below).
#    Classify each crop: Normal CV / Strange CV / Not a CV. Export triage.json.

# 3) normal CVs auto-extract; strange ones you already traced are finalized;
#    ones you only cropped get their panel.png saved for tracing later
.venv\Scripts\python.exe scripts\trace_guided.py triage triage.json -o data\out\triaged
#    -> per panel: CSV(s) + check.html (fade-by-eye QC, same as below)
```

**Calibration is per-curve, not per-paper.** Each curve traced in trace_assist
carries its own calibration (with a one-click "copy from" when several curves
in the same plot do share axes) — a curve accidentally traced from a
different sub-plot than intended never silently inherits the wrong axes;
old single-calibration `guides.json` files still load and apply to every
curve unchanged.

**Measured win from masking a legend box** (chen_2024_deconvolution_4958,
Fig. 3b — a black + red pair with an in-plot legend): automatic extraction
without cropping scored Chamfer 0.106 (poor) against the echemdb ground truth,
because the trace absorbed the legend's black text. Cropping to the panel and
masking the legend rectangle alone brought it to **0.034** — a 3× improvement,
with zero code changes, just excluding ink that was never part of the curve.
(The red curve in the same panel needed hand-tracing instead — its hue mask
only held the sharp peak where red is least blended with the overlapping
black curve, elsewhere it lost too much of its own ink to color-overlap. Ink
fidelity actually scored that partial trace 100/100 — perfectly on its own
ink, just not enough of it — which is exactly why the fade-by-eye check
matters even when the automatic self-score looks perfect.)

### Guided digitization — when a figure is too tangled to auto-trace

For the genuinely hard cases (a bundle of 8 near-identical curves, a solid you
want separated from its dashed twin, a curve lost in a noisy scan), a person
resolves the *ambiguity* in a few seconds by scribbling roughly along the curve
they mean — and the tool does the precise part. This is a **separate** tool, not
wired into the main pipeline:

```powershell
# 1) export the figure panel to a PNG
.venv\Scripts\python.exe scripts\trace_guided.py panel paper.pdf 4 -o panel.png

# 2) open tools\trace_assist.html in a browser (offline, nothing uploaded),
#    load panel.png, add a named curve, and drag roughly along it. Repeat per
#    curve. Download guides.json.

# 3) turn the rough guides into pixel-accurate curves
.venv\Scripts\python.exe scripts\trace_guided.py run panel.png guides.json -o out\
#    -> one <curve>.csv per guide + overlay.png + check.html (fade-by-eye QC)
```

The scribble only says *which* ink is which curve and the sweep direction; the
trace snaps to the real ink ([`cvdigitize/guided.py`](cvdigitize/guided.py):
`extract_near_guide`). On synthetic crossing curves a guide jittered by ±6 px
still recovers the intended curve to **sub-pixel** accuracy while ignoring the
one it crosses.

`extract` writes, under `data/out/<pdf>/[panel_x/]`:
`<curve>.csv` (+ `.json`/`.yaml` when calibrated), `overlay.png`, `curves.png`,
a top-level `report.json`, and an `index.html` you can open to see everything
at a glance.

### Check every trace by eye — and fix a bad one in one click

Every extracted panel (automatic *and* guided) also gets a **`check.html`** and a
transparent **`curve_overlay.png`**. Open `check.html`: the original figure is
shown with the extracted curve laid on top in a vivid contrasting colour, and a
slider **fades the trace in and out over the real ink** — the honest way to judge
a fit is to watch whether the line sits exactly on the ink or drifts off it (a
baked overlay you can't remove hides near-misses). A second slider fades the
figure so you can inspect the bare trace. `curve_overlay.png` is the same trace on
a transparent background at the panel's exact pixel size, to drop over the figure
in any image editor or slide.

If a curve looks wrong, click **“Open in trace_assist ▶”** in `check.html`: it
opens `tools/trace_assist.html` with *that exact panel already loaded*, so you
scribble a rough guide, export `guides.json`, and re-run `trace_guided.py run` to
replace the bad auto-trace with a pixel-accurate hand-guided one (§ above). Raster
figures are where this matters; vector figures almost always pass the eyeball
check untouched.

### Ink-fidelity: the tool grades its own traces (no ground truth needed)

You don't have to eyeball blindly. Every trace is scored by **ink-fidelity** — how
well it lies on the ink it was traced from ([`cvdigitize/fidelity.py`](cvdigitize/fidelity.py)):
walk the extracted curve, measure the distance to the nearest ink of its own
colour, and penalise only the runs that cross **long** stretches of empty space
(a chord). Short off-ink hops are *not* penalised, so a correctly-traced **dashed**
curve, which bridges its dash gaps, still scores high. No reference curve is
needed, so it works on every paper.

The extractor prints a one-line triage (`Ink-fidelity: worst 79/100 … eyeball
orange_dash(79)…`), `check.html` shows a green/amber/red badge per curve and draws
**red dashes over the chord spans** so your eye lands on the exact bad segment, and
`report.json` records each curve's score. In `cvdigitize batch`, papers are sorted
CV-like-first then **worst-fidelity-first**, so the traces most in need of a look
are at the top of the gallery. In practice: vector curves score ~100, solid raster
curves score high, and same-colour dashed *loops* (whose dashes interleave between
the two sweep branches — genuinely ambiguous to auto-trace) are the ones flagged
for a trace_assist re-draw.

### Calibration JSON (vector figures)

Two linear axis maps (svgdigitizer's reference-point model). Fields: two x
anchors `(x1,ex1),(x2,ex2)` and two y anchors `(y1,jy1),(y2,jy2)` in PDF-point
pixels → data values, plus units. See
`configs/rizo_2025_analysis_351_fig1a.calib.json`. Read the pixel anchors off
the `grid` command's output image.

### Calibration for raster figures — no JSON needed

Tick-mark *pixel positions* are found automatically (short dark marks just
outside the axes box). You only supply the two data values the first and last
detected tick correspond to (`--x-ticks LO,HI --y-ticks LO,HI`) — reading them
straight off the printed axis labels. No pixel-hunting required.

## Reproduce the demos

```powershell
.\.venv\Scripts\python.exe scripts\m0_demo.py              # extract + overlay + separate
.\.venv\Scripts\python.exe scripts\m0_compare_reference.py # vs manual reference (scores)
.\.venv\Scripts\python.exe scripts\m1_pipeline.py          # full vector pipeline, real units
.\.venv\Scripts\python.exe scripts\m2_pipeline.py          # full raster pipeline, real units
.\.venv\Scripts\python.exe scripts\make_synthetic_pdf.py   # controlled ground-truth PDF
.\.venv\Scripts\python.exe scripts\test_synthetic.py       # vector accuracy vs ground truth
.\.venv\Scripts\python.exe -m pytest -q                    # unit tests (42)
```

## Layout

```
cvdigitize/
  ingest.py           # page classification (vector-curves / raster / sparse)
  vector_extract.py   # colour grouping + panel-first localisation (M0 core)
  raster_extract.py   # frame/panel detection, tick-finding, colour-trace, skeleton ordering (M2 core)
  postprocess.py      # stray-swatch filter, loop stitching, branch split, resample
  calibrate.py        # linear axis calibration (anchors / bbox / reference-fit)
  package.py          # CSV + frictionless JSON + YAML echemdb datapackage writer
  cli.py, __main__.py
scripts/               # demos + generators + accuracy tests
tests/                 # pytest unit tests (geometry, calibration, packaging, raster)
configs/                # calibration JSONs
data/ in|reference|synthetic_truth|out
cvdigitize.bat, cvdigitize.ps1   # convenience launchers (no venv path typing)
```

## How it works

### Vector branch (native PDF geometry)
1. **Ingest/classify** — split pages; per page decide vector-curves vs raster.
2. **Extract** — `page.get_drawings()` → group stroked sub-paths by colour;
   for multi-panel figures, localise to a grid cell first, then group by colour.
3. **Post-process** — drop stray components (legend swatches); greedily stitch
   shuffled sub-paths into one ordered loop; split anodic/cathodic branches.
4. **Calibrate** — map PDF pixels → (E, j) with two linear axis maps.
5. **Resample** — even arc-length trace (smooth) or uniform-E per branch
   (potentiostat-like), removing uneven spacing and duplicate potentials.
6. **Package** — CSV + frictionless JSON (+ YAML), echemdb schema.

### Raster branch (colour/brightness trace, for flattened/scanned figures)
1. **Locate** embedded images on the page; render the region at high zoom.
2. **Find every plot panel** (`detect_all_frames`) — long dark horizontal/
   vertical pixel runs are candidate border positions; candidates are merged
   across the few rows/columns anti-aliasing spreads a border over (matched by
   span overlap, not just proximity, so unrelated nearby features never merge);
   rectangles are validated directly against the pixel mask; any rectangle that
   contains another valid one is dropped (that only happens when two adjacent
   panels' collinear borders coincidentally also form a rectangle).
3. **Find axis ticks** (`detect_axis_ticks`) — short dark marks just outside
   the frame, on the bottom and left edges, excluding corner positions (which
   are usually border bleed, not real ticks).
4. **Isolate the curve** — mask by HSV *brightness*, not saturation (anti-
   aliasing blends a black line into a coloured fill in a way that raises
   apparent saturation but not brightness); keep the largest connected
   component (drops tick marks, arrows, legend text).
5. **Skeletonize + order** — thin to 1px, prune short spurs (tick-mark stubs
   touching the curve), walk the pixel graph into one ordered polyline.
6. **Calibrate + package** — same linear axis maps and echemdb packaging as
   the vector branch, fed by the auto-detected ticks instead of a JSON file.
