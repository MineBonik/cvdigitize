# cvdigitize

Digitize Cyclic Voltammetry curves out of **native-vector** scientific PDFs into
`echemdb`-compatible datapackages — replacing the manual svgdigitizer workflow.

Give it a folder of papers. It finds every CV figure whose curves are stored as
real vector geometry, recovers each curve mathematically from the PDF's own path
data, separates the colour-coded curves, reconstructs each closed loop, and then
walks you through the panels in a browser so you confirm the **axis
calibration** — the one thing that cannot be read reliably from the file. Output
is CSV + frictionless JSON + echemdb YAML, written per panel as you go.

> **Vector only, by design.** For a vector figure the curve points are exact:
> we parse the PDF's path geometry directly, so there is no tracing, no
> resolution limit, and no guesswork — after a human validates a curve it is
> either right or it isn't a curve at all. Scanned/rasterized figures need
> pixel tracing, which is a fundamentally different (and fuzzier) problem; that
> pipeline was removed from this branch. Run `cvdigitize info` on a paper to
> find out which kind you have.

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# the whole workflow
.\cvdigitize.bat vector-calibrate --in data\papers
```

That scans the folder once, then opens a browser page that walks the CV panels
one at a time. `cvdigitize.bat` / `cvdigitize.ps1` are thin launchers so you
don't have to type the venv path; both do the same thing.

A bare path does the obvious thing:

```powershell
.\cvdigitize.bat data\papers        # a folder -> vector-calibrate
.\cvdigitize.bat paper.pdf          # a PDF    -> info
```

## The workflow

**1 — Scan (automatic, once).** Every PDF in the folder is checked for pages
carrying enough vector geometry to be a figure. On those, each plot's axes box
is detected, curves are assigned to the panel that contains them, and panels
that aren't cyclic voltammograms are dropped by their loop score. Papers are
scanned in parallel; ~3 minutes for 52 papers on 4 cores.

**2 — Confirm the axes (you, per panel).** Each panel is shown as the original
figure with the extracted curves drawn on top. Where the tool could read the
axis numbers — real text tick labels, or OCR of detected tick marks — the
calibration arrives pre-filled for you to check. Where it couldn't, nothing is
placed or typed for you: click a tick on each axis and type its value.

Your click lands exactly where you put it. Detected ticks are drawn as blue
reference lines with their values, but they never move an anchor and never fill
a box — tick detection is wrong often enough (latching onto a neighbouring
plot's axis, or numbering positions `1, 2, 3…` instead of reading them) that
letting it act would quietly produce a complete, plausible, wrong calibration.

Calibration is **per panel, not per curve**: every curve inside a panel shares
its axes, so a seven-curve figure is one calibration, not seven.

**3 — Name and save (you, per panel).** Each curve gets a suggested filename —
from the figure's legend text where it exists, otherwise a colour word — which
you can edit. Untick anything that isn't a real CV. **Save & next** writes that
panel's datapackages immediately, so stopping half-way loses nothing; re-run the
same command to resume.

### Units are never guessed

The unit boxes are filled from the figure's own axis title (`j (mA cm-2)`,
`E / V (vs. RHE)`) or left **blank**, and saving is refused until they are
filled. This is deliberate and was learned the hard way: two panels were once
saved as `uA/cm2` when their papers said `mA cm-2` — correct numbers, label
wrong by 1000×, and nothing on screen looked suspicious. A blank field a human
must fill is safe; a plausible default they have no reason to doubt is not.

The same rule runs throughout: **every automatic step may say "I don't know" and
hand the decision to a human, but none may guess plausibly.**

## What you get

```
data\out\vector_curated\
├─ index.json              # panel list + your progress (~300 KB)
├─ panels\<uid>.png        # the figure image you calibrate against
├─ geometry\<uid>.json     # curve points, loaded one panel at a time
└─ curves\<uid>\
   ├─ <name>.csv           # E, j — ordered, deduplicated, resampled
   ├─ <name>.json          # frictionless datapackage descriptor
   └─ <name>.yaml          # echemdb metadata: scan rate, electrolyte, DOI
```

The YAML carries the paper's **DOI, title, journal and year**, so a curve is
citable on its own — found for 49 of the 52 papers in the reference corpus, with
supplementary PDFs inheriting their parent's DOI.

## Commands

```powershell
# the tool
cvdigitize vector-calibrate --in "data\papers"
#   --rescan       re-detect panels, keeping the status/names/calibrations you entered
#   --out DIR      where curves land (default <work>\curves)
#   --workers N    parallel scan workers (default CPU count - 1; 1 = sequential)
#   --resample-mode uniform-E   even steps in potential per branch (see below)
#   --cv-threshold F            how loop-like a panel must be to count (0..1)

# pre-flight: what's in this paper, and can this tool use it?
cvdigitize info "paper.pdf"
cvdigitize info "paper.pdf" --quick     # page table only, skip panel detection
```

`info` runs the *same* detection the scan uses, so what it reports is exactly
what a scan would find — including telling you a paper's figures are raster and
this tool cannot help with them.

## Verified results

**Real paper — `rizo_2025_analysis_351` (ACS Electrochem 2025), vector figure.**
All 7 panel-(a) curves auto-extracted from the raw PDF match a hand-digitized
reference:
- shape match (normalised Chamfer): **0.0007–0.0068**
- with one calibration, curves overlay the reference in real units
  (V vs RHE, µA cm⁻²)
- point spacing **~10× more uniform** (CoV 2.4–8.4 → 0.26–0.69) with **zero**
  duplicate-potential points — the "data is not raw" critique, fixed
- `--resample-mode uniform-E` goes further: each scan branch is resampled on a
  uniform **potential** grid (CoV → 0, potentiostat-like), while keeping the
  loop two-valued (anodic + cathodic)

**Synthetic vector PDF (known ground truth).** 3 curves, different layout/axes:
mean normalised Chamfer **0.005** vs truth.

**Reference corpus (52 papers, 443 pages).** 84 CV panels across 20 papers,
243 curves, scanned in 3.1 minutes. 13 panels arrive with a pre-filled
calibration; the rest are two clicks and two numbers each.

**Real paper — Gámez et al., Electrochimica Acta 512 (2025), unusual encoding.**
This PDF draws curves as thin *filled* ribbon polygons (`color=None`, only
`fill` set), renders body text with *stroke* colour, and draws axis borders and
ticks as thin filled rectangles — all black, all landing in the same colour
bucket as the real curve. Fixed by grouping on fill colour only when a colour
has no stroke data on the page (so glyph shapes never contaminate an
already-good stroked curve), recognising axis-aligned thin rectangles and
narrow-but-tall shapes by geometry, and excluding tiny isolated segments (tick
marks) that skew a group's bounding box. All 5 curves extract with zero
cross-contamination.

*Design lesson worth keeping.* An early attempt fixed that same bug class by
having the loop-stitcher refuse to bridge "implausibly large" gaps. It seemed to
work, but one distance threshold could not separate a legitimate large gap
*inside* a real curve from a genuinely unrelated shape — it silently truncated
good curves to 5–10 % of their points. Reverted in favour of the shape/colour
filters above, which fix contamination at its source instead of guessing
distances after the fact.

## How it works

**Extraction is exact, not traced.** `page.get_drawings()` gives every path the
PDF draws, with its stroke colour and control points. Curves are grouped by
colour, filtered to drop axis rectangles / tick marks / text columns by shape,
and the shuffled sub-paths of one curve are stitched end-to-end into a single
ordered polyline.

**A CV is not a function.** It's a closed hysteresis loop: at most potentials
there are *two* current values (forward and reverse sweep), so column-by-column
tracing fails outright. The PDF also stores the loop as many disconnected
segments in arbitrary order. Recovering the correct traversal order around the
loop is the actual hard problem, and got the most engineering time.

**Panel detection is image-based, even here.** PDF geometry doesn't say which
strokes are the axes, so the page is rendered and the axes box found in pixels
([`cvdigitize/plotframe.py`](cvdigitize/plotframe.py)). That's why OpenCV is
still a dependency of a "vector-only" tool.

**svgdigitizer is not needed.** Parsing the geometry directly gives the true
points, so there's no SVG round-trip and no Inkscape.

## Layout

```
cvdigitize/
  cli.py            # two commands: info, vector-calibrate
  ingest.py         # page classification (vector / raster / sparse), rendering
  vector_extract.py # parse path geometry, group by colour, detect panels
  plotframe.py      # axes-box + tick-mark detection on the rendered page
  calibrate.py      # two-points-per-axis linear map (svgdigitizer's model)
  autocalib.py      # calibration + units from the PDF text layer
  ocr.py            # optional Tesseract read of tick labels (pre-fill only)
  postprocess.py    # loop ordering, branch split, dedupe, resampling
  metadata.py       # scan rate, electrolyte, reference electrode, legend
  paper_meta.py     # DOI, title, journal, year (SI inherits its parent's DOI)
  package.py        # CSV + frictionless JSON + echemdb YAML
  veccal/
    scan.py         # folder -> one calibration work unit per panel
    server.py       # localhost API behind vector-calibrate
    finalize.py     # confirmed calibration -> datapackages on disk
tools/veccal.html   # the calibration page
tests/              # 160 tests
```

## Limits

- **Dashed curves are dropped.** A dashed or dash-dot curve is stored as
  hundreds of separate short paths and they are not currently regrouped into one
  curve. Concretely: one corpus figure loses its Pt(111) reference, another a
  blue dash-dot trace. This is the top open item.
- **Most calibration is manual.** Only 13 of 84 corpus panels arrive pre-filled,
  because most of these papers convert their figure text to outlines, leaving
  nothing machine-readable.
- **A second y-axis is not detected.** If a panel plots something else against a
  right-hand axis, that curve gets the left axis's calibration and comes out
  silently wrong. Untick those.
- **The CV filter is a triage signal, not a classifier.** A genuinely
  peak-shaped CV can score low. It only selects and orders — it never deletes.
- **Raster figures are out of scope on this branch.** `info` will tell you when
  a paper is raster-only.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
