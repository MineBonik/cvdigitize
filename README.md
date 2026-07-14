# cvdigitize

Automated digitization of Cyclic Voltammetry (CV) curves from scientific PDFs
into `echemdb`-compatible data — replacing the manual svgdigitizer workflow.

Give it a paper PDF; it finds the CV figure (vector **or** scanned/rasterized),
separates the colour-coded curves, reconstructs each closed loop, calibrates to
real units, resamples to a clean even trace, and writes CSV + frictionless
JSON + YAML datapackages.

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

## Status

| Milestone | What it does | State |
|-----------|--------------|-------|
| **M0** | Vector PDF → auto-split each CV curve by stroke colour, localised per panel | done & verified |
| **M1** | Loop ordering, axis calibration, arc-length/uniform-E resampling, echemdb packaging, CLI | done & verified |
| **M2** | Auto vector/raster classify + colour/brightness-trace for rasterized figures, incl. multi-panel auto-detection, tiling-strip merging, closed **and L-shaped (despined)** axes, automatic tick-mark finding, and **multi-colour curve splitting by hue** (crossing-tolerant) | done & verified |
| **M3** | Automatic calibration | done for PDFs with live figure text; assisted (4 typed numbers) everywhere else |

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

The only remaining fully-manual case is reading tick label *text* on raster
figures without typing them (true OCR) — no OCR engine is available in this
environment, and the crops make the typing trivial.

## Verified results

**Real paper — `rizo_2025_analysis_351` (ACS Electrochem 2025), vector figure.**
All 7 panel-(a) curves auto-extracted from the raw PDF match Vladislav's
hand-digitized reference:
- shape match (normalised Chamfer): **0.0007–0.0068**
- with one global calibration, cleaned curves overlay the reference in real
  units (V vs RHE, µA cm⁻²)
- point spacing **~10× more uniform** (CoV 2.4–8.4 → 0.26–0.69) with **zero**
  duplicate-potential points — the "data is not raw" issue Albert flagged, fixed.

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
# 0) simplest possible: auto page, auto everything, normalised output
cvdigitize mypaper.pdf

# 1) inspect: which pages hold vector CV figures vs rasterized ones?
cvdigitize info "data\in\paper.pdf"

# 2) VECTOR figure, whole page, uncalibrated (normalised CSV + overlay)
cvdigitize extract "data\in\paper.pdf"

# 3) VECTOR, multi-panel + real units: split 2x2, take panel (a), calibrate
cvdigitize extract "data\in\rizo_2025_analysis_351.pdf" `
    --panels 2x2 --panel a `
    --calibration "configs\rizo_2025_analysis_351_fig1a.calib.json" `
    --figure 1a --scan-rate "50 mV/s"

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

# 7) SURVEY a whole folder of PDFs at once -> gallery sorted by CV-likeness
cvdigitize batch "data\corpus"
#   -> data\out\_batch\index.html : one card per paper (figure type, candidate
#      curves, loop-score confidence). Great for triaging many papers fast.
#      Recreate the sample corpus with: python scripts\fetch_corpus.py
```

(`cvdigitize` above = `cvdigitize.bat` / `.\cvdigitize.ps1`, or
`.venv\Scripts\python.exe -m cvdigitize` if you'd rather call Python directly.)

`extract` writes, under `data/out/<pdf>/[panel_x/]`:
`<curve>.csv` (+ `.json`/`.yaml` when calibrated), `overlay.png`, `curves.png`,
a top-level `report.json`, and an `index.html` you can open to see everything
at a glance.

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
