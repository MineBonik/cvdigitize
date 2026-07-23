"""Per-paper workspace CRUD: ``data/workspace/{paper_stem}/...`` on disk.

One folder per paper (see STUDIO_PLAN.md §5):

    {paper_stem}/
      analysis.json      page classification + sources, from /api/open_paper
      sources/           croppable source images (embedded-first, else full page)
      crops/
        {stem}_crop{n}.png    baked crop (mask already applied)
        {stem}_crop{n}.json   {name, type, source, bbox, excludeRects,
                               calibration, curves, lineWidth, axisWidth, parentCrop}
      curves/            final echemdb-format outputs (later phases)
      studio_state.json  session info, for resume
      index.html         per-paper dashboard (later phases)

All writes are guarded by a process-wide lock — the server is threaded, and
two requests touching the same paper's JSON files must not interleave.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading

import cv2
import numpy as np

_LOCK = threading.Lock()
_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def paper_stem(pdf_path: str) -> str:
    return os.path.splitext(os.path.basename(pdf_path))[0]


def safe_name(name: str) -> str:
    return _SAFE.sub("_", name).strip("_") or "item"


def paper_dir(workspace: str, stem: str) -> str:
    return os.path.join(workspace, stem)


def ensure_paper_dir(workspace: str, stem: str) -> str:
    d = paper_dir(workspace, stem)
    os.makedirs(os.path.join(d, "sources"), exist_ok=True)
    os.makedirs(os.path.join(d, "crops"), exist_ok=True)
    os.makedirs(os.path.join(d, "curves"), exist_ok=True)
    return d


def list_papers(workspace: str) -> list[str]:
    if not os.path.isdir(workspace):
        return []
    return sorted(n for n in os.listdir(workspace)
                 if os.path.isdir(os.path.join(workspace, n)))


# --------------------------------------------------------------------------- #
# analysis.json
# --------------------------------------------------------------------------- #
def write_analysis(workspace: str, stem: str, analysis: dict) -> str:
    d = ensure_paper_dir(workspace, stem)
    path = os.path.join(d, "analysis.json")
    with _LOCK, open(path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    return path


def read_analysis(workspace: str, stem: str) -> dict | None:
    path = os.path.join(paper_dir(workspace, stem), "analysis.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# sources/
# --------------------------------------------------------------------------- #
def save_source_png(workspace: str, stem: str, name: str, rgb: np.ndarray) -> str:
    d = ensure_paper_dir(workspace, stem)
    path = os.path.join(d, "sources", name)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def load_source_image(workspace: str, stem: str, name: str) -> np.ndarray | None:
    path = os.path.join(paper_dir(workspace, stem), "sources", name)
    if not os.path.exists(path):
        return None
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# crops/
# --------------------------------------------------------------------------- #
FIGURE_TYPES = ("single_cv", "multipanel_cv", "strange_cv", "other_graph", "not_a_graph")


def _crop_json_path(workspace: str, stem: str, crop: str) -> str:
    return os.path.join(paper_dir(workspace, stem), "crops", crop + ".json")


def _crop_png_path(workspace: str, stem: str, crop: str) -> str:
    return os.path.join(paper_dir(workspace, stem), "crops", crop + ".png")


def next_crop_name(workspace: str, stem: str) -> str:
    d = ensure_paper_dir(workspace, stem)
    existing = {f[:-5] for f in os.listdir(os.path.join(d, "crops")) if f.endswith(".json")}
    n = 1
    while f"{stem}_crop{n}" in existing:
        n += 1
    return f"{stem}_crop{n}"


def _decode_data_url_png(data_url: str, path: str) -> None:
    _, b64 = data_url.split(",", 1)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def save_crop(workspace: str, stem: str, *, type_: str, source: str, bbox,
             exclude_rects, image_data_url: str, parent_crop: str | None = None) -> dict:
    """Write a baked crop (mask already applied client-side) as png + json."""
    if type_ not in FIGURE_TYPES:
        raise ValueError(f"unknown crop type: {type_!r}")
    d = ensure_paper_dir(workspace, stem)
    name = next_crop_name(workspace, stem)
    png_path = os.path.join(d, "crops", name + ".png")
    with _LOCK:
        _decode_data_url_png(image_data_url, png_path)
        meta = {
            "name": name, "type": type_, "source": source, "bbox": list(bbox),
            "excludeRects": exclude_rects or [], "parentCrop": parent_crop,
            "calibration": None, "curves": [], "lineWidth": None, "axisWidth": None,
        }
        with open(_crop_json_path(workspace, stem, name), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return meta


def list_crops(workspace: str, stem: str) -> list[dict]:
    d = paper_dir(workspace, stem)
    crops_dir = os.path.join(d, "crops")
    if not os.path.isdir(crops_dir):
        return []
    out = []
    for fn in sorted(os.listdir(crops_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(crops_dir, fn), encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def load_crop_meta(workspace: str, stem: str, crop: str) -> dict | None:
    path = _crop_json_path(workspace, stem, crop)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_crop_meta(workspace: str, stem: str, crop: str, meta: dict) -> None:
    with _LOCK, open(_crop_json_path(workspace, stem, crop), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_crop_image(workspace: str, stem: str, crop: str) -> np.ndarray | None:
    path = _crop_png_path(workspace, stem, crop)
    if not os.path.exists(path):
        return None
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def delete_crop(workspace: str, stem: str, crop: str) -> bool:
    """Remove a crop's png+json.

    Curve files saved from this crop (§9 naming, added in a later phase) are
    not yet tracked back to their source crop, so cleaning those up is left to
    the save/label step that will introduce that bookkeeping.
    """
    found = False
    with _LOCK:
        for p in (_crop_png_path(workspace, stem, crop), _crop_json_path(workspace, stem, crop)):
            if os.path.exists(p):
                os.remove(p)
                found = True
    return found
