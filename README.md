# cvdigitize

Automated digitization of Cyclic Voltammetry (CV) curves from scientific PDFs
into `echemdb`-compatible data — replacing the manual svgdigitizer workflow.

Give it a paper PDF; it finds the CV figure, separates the colour-coded curves,
reconstructs each closed loop, calibrates to real units, resamples to a clean
even trace, and writes CSV + frictionless JSON + YAML datapackages.

> Key design choice: because we parse the PDF's **vector geometry** directly, we
> already have the true curve points — so no `svgdigitizer`/Inkscape round-trip
> is needed for vector figures. The tool is self-contained (PyMuPDF + numpy).

## Status

| Milestone | What it does | State |
|-----------|--------------|-------|
| **M0** | Vector PDF → auto-split each CV curve by stroke colour, localised per panel | ✅ done & verified |
| **M1** | Loop ordering, axis calibration, arc-length/uniform-E resampling, echemdb packaging, CLI | ✅ done & verified |
| M2 | Auto vector/raster classify (done) + OpenCV colour-trace for raster/scanned figures | ⏳ classifier done; tracer next |
| M3 | OCR axis auto-calibration | ⏳ |

## Verified results

**Real paper — `rizo_2025_analysis_351` (ACS Electrochem 2025).** All 7 panel-(a)
curves auto-extracted from the raw PDF match Vladislav's hand-digitized
reference:

- shape match (normalised Chamfer): **0.0007–0.0068**
- with one global calibration, the cleaned curves overlay the reference in real
  units (V vs RHE, µA cm⁻²)
- point spacing **~10× more uniform** (CoV 2.4–8.4 → 0.26–0.69) with **zero**
  duplicate-potential points — the "data is not raw" issue Albert flagged, fixed.

**Synthetic vector PDF (known ground truth).** 3 curves, different layout/axes:
mean normalised Chamfer **0.005** vs truth.

**Real raster paper (arXiv).** Correctly classified as raster-only ("no vector
figure candidates") — honestly routed to the not-yet-built M2 branch.

Handled real-data quirks: the same logical curve drawn in slightly different RGB
across panels (panel-first grouping); shuffled/again-ordered path segments
(greedy stitching); legend colour swatches (connected-component filtering);
light grid lines (saturation filter).

## Setup (Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

```powershell
# 1) inspect: which pages hold vector CV figures? (vector vs raster per page)
python -m cvdigitize info "data\in\paper.pdf"

# 2) extract (whole figure, auto page). UNCALIBRATED => normalised CSV + overlay
python -m cvdigitize extract "data\in\paper.pdf"

# 3) multi-panel + real units: split 2x2, take panel (a), apply a calibration
python -m cvdigitize extract "data\in\rizo_2025_analysis_351.pdf" \
    --panels 2x2 --panel a \
    --calibration "configs\rizo_2025_analysis_351_fig1a.calib.json" \
    --figure 1a --scan-rate "50 mV/s"

# 4) build a calibration: render a page with a pixel grid to read axis anchors
python -m cvdigitize grid "data\in\paper.pdf" --page 1
```

`extract` writes, under `data/out/<pdf>/[panel_x/]`:
`<curve>.csv` (+ `.json`/`.yaml` when calibrated), `overlay.png`, and a
top-level `report.json`.

### Calibration JSON

Two linear axis maps (svgdigitizer's reference-point model). Fields: two x
anchors `(x1,ex1),(x2,ex2)` and two y anchors `(y1,jy1),(y2,jy2)` in PDF-point
pixels → data values, plus units. See
`configs/rizo_2025_analysis_351_fig1a.calib.json`.

## Reproduce the demos

```powershell
.\.venv\Scripts\python.exe scripts\m0_demo.py              # extract + overlay + separate
.\.venv\Scripts\python.exe scripts\m0_compare_reference.py # vs manual reference (scores)
.\.venv\Scripts\python.exe scripts\m1_pipeline.py          # full pipeline, real units
.\.venv\Scripts\python.exe scripts\make_synthetic_pdf.py   # controlled ground-truth PDF
.\.venv\Scripts\python.exe scripts\test_synthetic.py       # accuracy vs ground truth
.\.venv\Scripts\python.exe -m pytest -q                    # unit tests (16)
```

## Layout

```
cvdigitize/
  ingest.py          # page classification (vector-curves / raster / sparse)
  vector_extract.py  # colour grouping + panel-first localisation (M0 core)
  postprocess.py     # stray-swatch filter, loop stitching, branch split, resample
  calibrate.py       # linear axis calibration (anchors / bbox / reference-fit)
  package.py         # CSV + frictionless JSON + YAML echemdb datapackage writer
  cli.py, __main__.py
scripts/             # demos + generators + accuracy tests
tests/               # pytest unit tests (geometry, calibration, packaging)
configs/             # calibration JSONs
data/ in|reference|synthetic_truth|out
```

## How it works (vector branch)

1. **Ingest/classify** — split pages; per page decide vector-curves vs raster.
2. **Extract** — `page.get_drawings()` → group stroked sub-paths by colour;
   for multi-panel figures, localise to a grid cell first, then group by colour.
3. **Post-process** — drop stray components (legend swatches); greedily stitch
   shuffled sub-paths into one ordered loop; split anodic/cathodic branches.
4. **Calibrate** — map PDF pixels → (E, j) with two linear axis maps.
5. **Resample** — even arc-length trace (smooth) or uniform-E per branch
   (potentiostat-like), removing uneven spacing and duplicate potentials.
6. **Package** — CSV + frictionless JSON (+ YAML), echemdb schema.
```
