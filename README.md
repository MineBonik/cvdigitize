# cvdigitize

Automated digitization of Cyclic Voltammetry (CV) curves from scientific PDFs
into `echemdb`-compatible data — replacing the manual svgdigitizer workflow.

See the full design in the approved plan (`i-have-the-email-quiet-ladybug.md`).

## Status

| Milestone | What it does | State |
|-----------|--------------|-------|
| **M0** | Vector PDF → auto-split each CV curve by stroke colour, localised per panel | ✅ working & verified |
| M1 | Add axis calibration + svgdigitizer → clean CSV; loop-order & resample post-process | ⏳ next |
| M2 | Auto-detect vector vs raster; OpenCV colour-trace for scanned figures | ⏳ |
| M3 | OCR axis auto-calibration; unified CLI | ⏳ |

## M0 result (proven on `rizo_2025_analysis_351.pdf`)

Directly from the raw PDF, the tool finds the CV figure page, separates all 7
overlaid curves by stroke colour, and localises them to the correct sub-panel
(a/b/c/d). Extracted panel-(a) curves match Vladislav's hand-digitized
reference CSVs with a normalised Chamfer distance of **0.0007–0.0068** (all
well under the 0.02 "excellent overlay" threshold).

Key real-data quirk handled: panel (a)'s Pt(331) curve is drawn in a slightly
different purple `(0.5,0,1.0)` than the other panels' violet `(0.5,0,0.5)`. The
pipeline localises panels **first**, then groups by colour **within** a panel,
so this inconsistency does not break the split.

## Setup (Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the M0 demo

```powershell
# 1) extract + overlay curves on the original, and separate each colour
.\.venv\Scripts\python.exe scripts\m0_demo.py

# 2) verify the extraction against the manual reference CSVs (prints scores)
.\.venv\Scripts\python.exe scripts\m0_compare_reference.py
```

Outputs land in `data/out/m0/`:
- `overlay.png` — extracted curves drawn over the original figure
- `separated_grid.png` — each colour on its own axes (shows the 4-panel copies)
- `compare_reference.png` — auto vs manual, per curve, with match scores

## Layout

```
cvdigitize/
  vector_extract.py   # M0: colour grouping + panel localisation (core library)
scripts/
  m0_demo.py                 # extraction + overlay artifacts
  m0_compare_reference.py    # verification vs manual reference
data/
  in/            # source PDFs
  reference/     # Vladislav's hand-made svgdigitizer output (ground truth)
  intermediate/  # per-curve SVGs (later milestones)
  out/           # generated artifacts
```
