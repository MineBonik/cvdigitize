# cvdigitize

Digitizes Cyclic Voltammetry (CV) curves out of scientific paper PDFs into
`echemdb`-compatible datapackages (CSV + frictionless JSON + YAML), replacing a
manual svgdigitizer/Inkscape workflow. Self-contained: no external services, no
SVG round-trip.

`README.md` is the user-facing manual and the record of *measured* results
(benchmark numbers, rejected approaches). Read it before changing extraction
behaviour — several plausible-looking fixes are documented there as measured
regressions. `STUDIO_PLAN.md` is the build spec for CV Studio.

## Environment

The repo has **no `.venv`**, so `cvdigitize.bat` / `cvdigitize.ps1` (which
hardcode `.venv\Scripts\python.exe`) do not work as-is. Use pixi:

```powershell
pixi install                 # once
pixi run test                # pytest (268 pass, 1 skip)
pixi run cvdigitize info data\in\rizo_2025_analysis_351.pdf
pixi run studio              # CV Studio browser UI
```

`pixi.toml` mirrors `requirements.txt`; keep both in sync when adding a
dependency. Do not create a venv or `pip install` (see the global instructions).

`pixi run test` passes `--basetemp=.pytest_tmp` on purpose: `check.html` links to
`tools/trace_assist.html` by relative path, so with pytest's tmp dir on `C:` and
the repo on `D:` there is no such path and the handoff-link tests skip
themselves. Keeping the tmp dir on the repo's drive exercises them (271 passed
vs 269 passed / 3 skipped).

## Two extraction branches — know which one you're in

- **Vector** (`vector_extract.py`): parse `page.get_drawings()` geometry
  directly, so the true curve points are already there. Group stroked subpaths
  by colour, panel-first. Near-exact (benchmark Chamfer ~0.01).
- **Raster** (`raster_extract.py`): the figure is a flattened bitmap. Render at
  zoom, detect panel frames and tick positions from pixel runs, mask the curve
  by HSV *brightness* (not saturation — anti-aliasing blends a dark line into a
  coloured fill in saturation but not value), skeletonize, order the pixel graph.

Both then share `postprocess.py` → `calibrate.py` → `package.py`.

`strands.py` is the highest-leverage piece of the raster path: a skeleton of
overlapping strokes is a *graph*, so it is collapsed to nodes+edges and branches
are paired by straight-through tangent continuity. That is what survives
crossings, dashed siblings and axis lines. Sharp CV peaks are not junctions and
must stay intact.

## Module map

| File | Role |
| --- | --- |
| `ingest.py` | page classification: vector-curves / raster / sparse |
| `vector_extract.py` | colour grouping, panel detection, figure-page finding |
| `raster_extract.py` | frame + tick detection, colour trace, skeleton ordering |
| `strands.py` | junction-aware strand decomposition |
| `postprocess.py` | stray-component filter, loop stitching, branch split, resample |
| `calibrate.py` / `autocalib.py` | linear axis maps; tier-1 fit from text tick labels |
| `ocr.py` | optional Tesseract read of tick labels — pre-fill only, never authoritative |
| `metadata.py` / `paper_meta.py` | scan rate / electrolyte / reference electrode from caption; DOI, title, journal |
| `fidelity.py` | reference-free ink-fidelity score (dash-aware) |
| `qc.py` | `check.html` fade-by-eye overlay, `curve_overlay.png` |
| `package.py` | CSV + frictionless JSON + YAML echemdb writer |
| `echemdb_ref.py` | parse echemdb curators' svgdigitizer SVGs as ground truth |
| `guided.py` | snap a human's rough scribble to the real ink |
| `veccal/` | `vector-calibrate`: folder scan → per-panel browser confirm → save |
| `studio/` | CV Studio: 5-step guided per-paper workflow |
| `cli.py` | subcommands `info` `extract` `grid` `batch` `studio` `vector-calibrate` (a bare PDF path = `extract`) |

Browser UIs are single static files in `tools/` (`veccal.html`,
`studio.html`, `trace_assist.html`) talking to a **stdlib-only**
`ThreadingHTTPServer` bound to 127.0.0.1. No Flask, no JS build step, no new
dependencies — the servers write files, so they must stay off the network.

## Conventions that matter here

- **Never silently corrupt data.** The repeated design rule: a heuristic may
  select, order or flag, but it must not delete or half-calibrate. OCR reads an
  axis fully or not at all; the loop score triages but never drops curves; the
  CV-plausibility gate makes the benchmark report "no acceptable curve" rather
  than flatter itself against a watermark.
- **Human confirmation over clever guessing**, and guesses land next to a
  picture so a mistake is visible rather than silent.
- **Paths in generated artifacts are relative to this repo**, never absolute and
  never `file://`. `report.json`, `index.html`, `check.html` and the Studio /
  veccal APIs all address files relatively, so an output folder stays valid when
  it is moved or shared. Write outputs under the repo (`data/out`,
  `data/workspace`); `qc.py`'s `_rel_posix` returns `None` rather than inventing
  an absolute path when no relative one exists, and callers degrade.
- **Save incrementally.** `vector-calibrate` and Studio write each panel's
  datapackages the moment the user presses save, and re-running resumes from
  `index.json` / `studio_state.json`. Don't batch writes to the end.
- **Distance thresholds are a trap.** Fix contamination at its source (shape,
  colour, geometry) rather than having the stitcher refuse "implausible" gaps —
  that was tried and truncated good curves (README, "Design lesson").
- **Measure claims.** Extraction changes are justified with a benchmark number
  (`scripts/benchmark.py --echemdb`) or ground truth, not by eye. Rejected
  experiments are recorded rather than deleted.
- Docstrings explain *why* a non-obvious approach was chosen, often naming the
  real paper that forced it. Match that: a new heuristic should say what broke
  without it.
- Commit subjects state the user-visible effect in the imperative, with numbers
  when there are numbers: "Speed up the folder scan 4.8x: 15 min -> 3.1 min",
  "Read axis units from the figure instead of defaulting them".

## Data layout

`data/in/` test PDFs · `data/reference/` hand-digitized comparison curves ·
`data/synthetic_truth/` generated ground truth · `data/out/` generated output
(gitignored) · `data/workspace/` Studio per-paper state (gitignored).
`data/echemdb/`, `data/corpus/` and `data/literature_Vladislav/` are
external/copyrighted and deliberately not committed.
