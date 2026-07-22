# CV Studio — step-by-step guided digitization pipeline (implementation spec)

**Audience:** the implementer (Sonnet 5 ultracode). This is a build spec, not a
narrative. Everything here is grounded in the existing codebase — reuse the
functions named in the "Reuse map"; do not rebuild them.

## 1. Goal

Replace the current à-la-carte tools (`pages`, `gallery`, `trace_assist`,
`triage`, `run`, `extract`) with **one linear, guided, per-paper workflow** that
pings between a browser UI (human steps: crop, classify, validate, hand-trace)
and Python (compute steps: analyze PDF, extract vector, auto-calibrate,
auto-extract) live, with a validation gate at calibration and at extraction, and
echemdb-style output naming. Organized as one workspace folder per paper.

The existing tools keep working; Studio is an additive orchestration layer that
reuses their internals.

## 2. Decisions already locked (do not re-litigate)

- **Architecture:** a **local companion server**, Python **stdlib only**
  (`http.server.ThreadingHTTPServer` + `BaseHTTPRequestHandler`) — no Flask, no
  new dependencies. Launch `python -m cvdigitize studio`; it opens the browser
  at `http://localhost:<port>`. Localhost only; nothing uploaded.
- **Figure-type gradation (5-tier):** `single_cv` · `multipanel_cv` · `strange_cv`
  · `other_graph` · `not_a_graph`. The type is a *starting hint*, not a hard
  gate — **hand-tracing is always reachable from the extraction step for any
  type** when the by-eye overlay check fails (explicit user amendment).
- **Auto-calibration:** positions auto, human types 2 values per axis. Detect
  the axis frame + tick *positions* and show zoomed crops of the end numbers; the
  human reads and types them. No OCR dependency, no silent mis-reads.

## 3. Architecture

New package `cvdigitize/studio/`:
- `server.py` — the stdlib HTTP server + JSON API (§6). Entry point wired into
  `cvdigitize/__main__.py` as a `studio` subcommand:
  `python -m cvdigitize studio [--port 8799] [--workspace data/workspace] [--no-open]`.
  Uses `webbrowser.open` unless `--no-open`.
- `workspace.py` — workspace CRUD (§5): create/load per-paper folders, read/write
  `studio_state.json`, `crops/*.json`, regenerate `index.html`.
- `pipeline.py` — thin adapters that call existing `cvdigitize` functions and
  shape their I/O for the API (keeps `server.py` request-handling-only).
- `tools/studio.html` — the single-page client. **Fork it from the current
  `tools/trace_assist.html`** (reuse its canvas transforms, crop mode, trace
  mode, calibration markers, zoom/pan). The client is thin: it renders state and
  calls the API; all CV compute is server-side.

Images cross the wire as data URLs for previews (overlays, tick-label crops) and
as workspace-relative file paths for things already on disk (the server also
statically serves the workspace dir read-only).

## 4. UI flow (studio.html) — the linear stepper

Left rail = steps 1→5 (click to revisit). Center = canvas. Right = context panel.
The active crop is the unit of work through steps 2–5.

- **Step 1 — Paper & sources.**
  Pick a PDF (path field; a file picker may `POST` the bytes, but path is
  primary since papers live in `data/literature_Vladislav/`). Server analyzes and
  returns a page filmstrip, each page tagged `vector` / `raster` / `sparse` and
  "N embedded images". Clicking a source loads it on the canvas.
  **Crop:** pick the **type first** (5 buttons), then drag the box; `Alt`+drag =
  mask-out rectangle (paint white); `Enter` = save → the crop appears in a
  gallery with a **delete ✕**. Crops are **re-croppable** (load a saved crop as a
  source and crop again to tighten — this is the `crop1→crop2→crop3` iterative
  case the user hit on `hoshi_2008_surface_6070` p2, where crop3 @445×479 was
  best). `multipanel_cv` prompts, after the outer crop, to sub-crop each panel;
  each child becomes a `single_cv` crop. If the crop's source region is vector,
  show **"Try vector extract"** (calls `/extract_vector`); if the returned
  polylines look right, jump straight to Step 5.
- **Step 2 — Calibrate.** For the selected crop: **Auto-calibrate** → 4 proposed
  markers drop on the overlay + zoomed end-number crops appear; you type 2
  values/axis; drag any marker to fix. Overlay stays on so you validate by eye.
  Manual-from-scratch calibration (today's calibrate mode) also available.
- **Step 3 — Measure (automatic, shown).** On calibrate-accept the server
  measures **line width** and **axis width**, sets the default brush ≈ 1.8× line
  width, and computes the **axis-exclusion band** (the loci through the
  calibration points, ± axis width, only between the points) — drawn as a faint
  red band so you see exactly what extraction will ignore.
- **Step 4 — Extract & check.** **Auto-extract** → superimposed overlay with each
  curve emphasized (bright, thick) + per-curve fidelity badges + off-ink chords
  flagged red. Choice window: **Accept** (→ Step 5) or **Hand-trace** (→ trace
  mode, brush pre-set from the measured line width). Hand-trace is available here
  for **any** type when the eye check fails.
- **Step 5 — Label & save.** Per curve: a name field (e.g. "Pt(111) 0.1 M
  HClO₄"), a figure/panel tag (default `crop{N}`, editable to echemdb `f2a`
  style). **Save** → `{base}_{figtag}_{label}.csv/.json/.yaml` in `curves/`.
  Per-paper `index.html` refreshes.

## 5. Workspace layout (one folder per paper)

```
data/workspace/{paper_stem}/
  analysis.json          # page classification + embedded-image regions (from /open_paper)
  sources/               # croppable source images
    p2_img0.png          #   embedded images, extracted natively when present
    p3_full.png          #   else full-page renders
  crops/
    {stem}_crop1.png     # baked crop (mask applied)
    {stem}_crop1.json    # {type, source, bbox, excludeRects, calibration, curves[],
                         #  lineWidth, axisWidth, parentCrop?}
  curves/                # final echemdb-format outputs
    {stem}_f2a_black.csv (+ .json + .yaml)
  studio_state.json      # whole session for resume + a decisions log (audit trail)
  index.html             # per-paper dashboard (auto-regenerated on save)
```

`{paper_stem}` = the PDF filename stem. The `literature_Vladislav` filenames are
already echemdb-style `{surname}_{year}_{firstword}_{firstpage}`
(e.g. `hoshi_2008_surface_6070`), so the base name comes straight from the file.
If a chosen PDF is **not** in that pattern, prompt the human once for the base.

## 6. API endpoints (server.py) — request → response contracts

All POST, JSON in/out. `paper` = paper_stem. Errors return `{error}` + 4xx.

| Endpoint | In | Out | Backing calls |
|---|---|---|---|
| `/api/open_paper` | `{pdf_path}` | `{paper, pages:[{n,kind,n_embedded,sources:[png],vector:bool}]}` | `ingest.classify_pdf`; `raster_extract.find_image_regions`+`render_region` (embedded, higher-res) else `ingest.render_page`; write `analysis.json` |
| `/api/extract_vector` | `{paper, source, bbox}` | `{polylines:[{name,rgb,xy}]}` or 400 if not vector | `vector_extract.*` restricted to bbox |
| `/api/save_crop` | `{paper, type, source, bbox, excludeRects, image}` | `{crop, png}` | `workspace.save_crop` (bake mask, write png+json) |
| `/api/delete_crop` | `{paper, crop}` | `{ok}` | `workspace.delete_crop` (png+json+any curves) |
| `/api/autocalibrate` | `{paper, crop}` | `{points:{E1,E2,j1,j2:[x,y]}, labelCrops:{E1..:dataURL}}` (positions only) | `raster_extract.detect_frame_bbox`+`detect_axis_ticks`; `autocalib.detect_ticks_for_bbox`; `raster_extract.crop_tick_labels` |
| `/api/measure` | `{paper, crop, calibration}` | `{lineWidth, axisWidth, suggestedBrush, exclusionBand}` | new `raster_extract.measure_line_and_axis_width` (§8) |
| `/api/autoextract` | `{paper, crop, calibration}` | `{curves:[{name,rgb,xy_px,xy_real}], overlay:dataURL, fidelity:[...]}` | `raster_extract.extract_from_cropped_image` with axis-masked input (§8); `qc`/`fidelity` |
| `/api/trace` | `{paper, crop, guides, calibration}` | same shape as `/autoextract` | `guided.extract_near_guide` (per-stroke radius, §7) |
| `/api/save_curves` | `{paper, crop, curves:[{name,figtag,label,xy_real}], calibration, type}` | `{written:[paths]}` | `package.write_datapackage`; regenerate `index.html` |

## 7. Brush fixes the user explicitly called out

**7a. Changing the brush mid-trace corrupts the whole curve.**
Root cause: radius is stored per-**curve** (`p.curves[active].radius = brush` at
each `pointerdown`) and `redraw()` uses `curve.radius` for *all* its strokes, so
changing the slider then drawing one more stroke rewrites the whole curve's
width. **Fix: store radius per-STROKE.** Change the stroke shape to
`{radius, pts:[[x,y]…]}`. Thread per-stroke radius through:
- `tools/studio.html` (+ `trace_assist.html`) draw/export/import,
- `cvdigitize/guided.py` `extract_near_guide` / `extract_guides` — snap each
  stroke with **its own** radius (the per-stroke loop already exists in
  `_snap_stroke`; pass the stroke's radius instead of the guide's),
- the `/api/trace` adapter.
Back-compat: an old flat `strokes:[[...]]` + one `radius` still loads — wrap each
stroke as `{radius: guideRadius, pts}` on import.

**7b. The drawn band under-represents what actually gets captured.**
Root cause: `redraw()` draws the corridor at `radius*zoom*0.5`, but the real snap
corridor is `2*radius` (a diameter) — **4× wider than shown**. **Fix: draw the
band at the true captured width** = `2*radius*zoom` screen px, semi-transparent,
with the thin centre line on top. What you see is then exactly what will be
snapped. (This is the "add a much bigger coefficient" the user asked for; it is
specifically `0.5 → 2.0` on the diameter.)

## 8. Line/axis width + axis exclusion ("don't confuse line with axis")

New `raster_extract.measure_line_and_axis_width(crop_rgb, calibration) -> dict`:
- The axes are the loci **through the calibration points**: x-axis = the
  horizontal line at E1/E2's y-pixel; y-axis = the vertical line at j1/j2's
  x-pixel. Measure each axis's thickness from the dark-run length crossing it.
- `line_width` = median stroke width of the **non-axis** dark/colour ink, reusing
  the existing `ink_area / skeleton_length` measure (see
  `raster_extract._mask_to_curves`, `max_stroke_width`).
- Returns `{line_width, axis_width, suggestedBrush≈1.8*line_width, exclusionBand}`
  where `exclusionBand` is the set of rows/cols within ±axis_width of the two
  axis loci, **only between the calibration points**.

Extraction (`/autoextract`) masks the `exclusionBand` out of the ink **before**
tracing, so the solid-black axis is never traced as data. This is cleaner and
more exact than the current strand `_is_axis_strand` heuristic because
calibration pins the axis position precisely. `line_width` also drives adaptive
params: brush default ≈1.8×lw; `max_stroke_width` gate ≈ `max(9, 3*lw)`; fidelity
"on-ink" tolerance ≈ lw.

## 9. Output naming

- Base = `{paper_stem}` (already echemdb-style; §5).
- Curve file = `{base}_{figtag}_{label}` — `figtag` defaults to `crop{N}`,
  editable to `f2a`; `label` = sanitized UI curve name.
  e.g. `hoshi_2008_surface_6070_f2a_black`.
- Write CSV + frictionless JSON + YAML via existing `package.write_datapackage`,
  carrying the human's label, the calibration, and the source bbox in metadata
  (so provenance is recoverable). Default curve labels can be pre-filled from
  `metadata.parse_curve_legend` when the caption names colours.

## 10. Reuse map (existing functions — do NOT rebuild)

`ingest.classify_pdf`, `ingest.render_page`; `raster_extract.find_image_regions`,
`render_region`, `extract_from_cropped_image`, `detect_frame_bbox`,
`detect_axis_ticks`, `split_color_curves`, `mask_dark_curve`, `crop_tick_labels`;
`autocalib.detect_ticks_for_bbox`, `assisted_tick_calibration`; `strands.*`;
`guided.extract_near_guide`, `split_strokes`; `qc.write_qc`,
`fidelity.ink_fidelity`; `postprocess.*`; `package.write_datapackage`;
`metadata.extract_figure_metadata`, `parse_curve_legend`;
`vector_extract.*` for the vector-crop path.

## 11. Phasing (suggested build order)

- **G1 — skeleton + Step 1.** `studio` subcommand + server + `/open_paper`
  (embedded-first, page fallback) + workspace layout + crop/type/delete + gallery.
  Ships the organized per-paper folders and typed crops.
- **G2 — brush fixes (§7).** Self-contained UI change, testable immediately;
  also back-port to `trace_assist.html`.
- **G3 — calibrate (Step 2).** `/autocalibrate` (positions) + overlay-validate +
  marker drag + type-2-values.
- **G4 — measure + extract (Steps 3–4).** `measure_line_and_axis_width` +
  axis-exclusion masking + `/autoextract` with emphasized superimposition +
  fidelity gate + Accept/Hand-trace choice.
- **G5 — vector-crop path.** `/extract_vector` + Step-1 "Try vector extract".
- **G6 — label & save + resume (Step 5).** echemdb-named save + per-paper
  `index.html` + `studio_state.json` resume + decisions log.

## 12. Verification

- **Unit:** `measure_line_and_axis_width` on a synthetic crop with known line vs
  axis thickness; axis-exclusion mask removes the axis rows/cols and nothing
  else; per-stroke radius round-trips through export→import; an old flat
  `guides.json` still loads (back-compat).
- **Server:** hit each endpoint with a saved fixture request; assert response
  shape + that files land in the workspace. Use `--no-open` + a fixed port in
  tests; drive endpoints with `urllib`, no browser needed.
- **Integration on the real cases the user gave:**
  - `hoshi_2008_surface_6070` p2 (3 embedded images) → `/open_paper` extracts
    them to `sources/`; crop to the best plot (the crop3 case) → auto-calibrate →
    auto-extract → overlay check.
  - `chen_2024_deconvolution_4958` Fig 3b → crop panel b + mask legend →
    auto-extract → black curve Chamfer ≤ 0.04 vs echemdb (already shown
    achievable); red → hand-trace fallback (fidelity gate flags it).
- **Regression:** existing 138 tests pass; the 3 ground-truth papers
  (hara 0.016 / garcia 0.022 / chen 0.023) unchanged.

## 13. Risks / notes for the implementer

- **stdlib server is single-threaded per handler by default** — use
  `ThreadingHTTPServer` so a slow extract doesn't freeze the UI; guard shared
  workspace writes with a lock.
- **Big data URLs** (baked crops, overlays) can exceed practical limits — prefer
  serving workspace files by path for anything already on disk; reserve data URLs
  for transient previews (tick-label crops, extraction overlays).
- **Vector-crop extraction** needs the bbox mapped from rendered-image pixels
  back to PDF points (render zoom factor) before calling `vector_extract` — reuse
  the zoom bookkeeping in `raster_extract.extract_raster_curve`.
- Keep Studio **additive**: don't remove `pages`/`gallery`/`triage`/`run`; Studio
  calls the same internals, so both stay valid entry points.
