"""Write a digitized curve as an echemdb-compatible datapackage (M1).

Mirrors the structure of the rizo reference output: a ``<name>.csv`` with an
``E,j`` table plus a ``<name>.json`` frictionless Data Package descriptor
carrying an ``echemdb`` metadata block. Optionally also emits a ``<name>.yaml``
for human-friendly metadata review (the step Albert does with an LLM).
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CurveMeta:
    """Minimal provenance for one digitized curve."""

    name: str
    figure: str = ""
    curve: str = ""           # colour or sample label
    scan_rate: str = ""       # e.g. "50 mV/s"
    x_label: str = "E"
    x_unit: str = "V vs RHE"
    y_label: str = "j"
    y_unit: str = "uA/cm2"
    source_pdf: str = ""
    method: str = "digitized"
    tags: list[str] = field(default_factory=list)
    comment: str = ""
    extracted: dict = field(default_factory=dict)  # auto-extracted text metadata


def write_csv(path: str, data: np.ndarray, meta: CurveMeta) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([meta.x_label, meta.y_label])
        for row in data:
            w.writerow([repr(float(row[0])), repr(float(row[1]))])


def build_descriptor(csv_name: str, meta: CurveMeta) -> dict:
    """Frictionless Data Package descriptor with an echemdb metadata block."""
    return {
        "resources": [
            {
                "name": meta.name,
                "type": "table",
                "path": os.path.basename(csv_name),
                "scheme": "file",
                "format": "csv",
                "mediatype": "text/csv",
                "encoding": "utf-8",
                "schema": {
                    "fields": [
                        {"name": meta.x_label, "type": "number", "unit": meta.x_unit},
                        {"name": meta.y_label, "type": "number", "unit": meta.y_unit},
                    ]
                },
                "metadata": {
                    "echemdb": {
                        "experimental": {"tags": meta.tags},
                        "source": {"figure": meta.figure, "curve": meta.curve,
                                   "pdf": os.path.basename(meta.source_pdf)},
                        "figureDescription": {
                            "type": meta.method,
                            "simultaneousMeasurements": [],
                            "measurementType": "custom",
                            "scanRate": meta.scan_rate,
                            "fields": [
                                {"name": meta.x_label, "type": "number",
                                 "unit": meta.x_unit, "orientation": "horizontal"},
                                {"name": meta.y_label, "type": "number",
                                 "unit": meta.y_unit, "orientation": "vertical"},
                            ],
                            "comment": meta.comment,
                        },
                        # Metadata parsed from the figure caption / page text.
                        # Present so a curator can confirm rather than re-type;
                        # always flagged as auto-extracted, never authoritative.
                        **({"autoExtracted": meta.extracted} if meta.extracted else {}),
                    }
                },
            }
        ]
    }


def write_json(path: str, csv_name: str, meta: CurveMeta) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_descriptor(csv_name, meta), f, indent=4)


def _to_yaml(obj, indent: int = 0) -> str:
    """Tiny YAML emitter (dict/list/scalar) to avoid a PyYAML dependency."""
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return pad + "{}\n"
        out = ""
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out += f"{pad}{k}:\n{_to_yaml(v, indent + 1)}"
            else:
                out += f"{pad}{k}: {_scalar(v)}\n"
        return out
    if isinstance(obj, list):
        if not obj:
            return pad + "[]\n"
        out = ""
        for v in obj:
            if isinstance(v, (dict, list)) and v:
                out += f"{pad}-\n{_to_yaml(v, indent + 1)}"
            else:
                out += f"{pad}- {_scalar(v)}\n"
        return out
    return pad + _scalar(obj) + "\n"


#: Characters/patterns that stop a YAML plain (unquoted) scalar from being read
#: back as the string we wrote. ``": "`` is the dangerous one in practice: a
#: curve label like ``red: Pt(111)`` or a caption containing a newline used to
#: be emitted bare, and the file then failed to parse at all
#: ("mapping values are not allowed here") — the sidecar is meant to be
#: machine-read by echemdb, so that made the metadata worthless.
_YAML_UNSAFE = (": ", " #", "\n", "\r", "\t", '"', "'", "\\")
_YAML_UNSAFE_LEADING = "-?:,[]{}#&*!|>%@`\"' "
#: Plain scalars that YAML 1.1 readers coerce to a non-string type.
_YAML_RESERVED = {"true", "false", "yes", "no", "on", "off", "null", "~", ""}


def _needs_quotes(s: str) -> bool:
    if s.strip() != s or s.lower() in _YAML_RESERVED:
        return True
    if s[0] in _YAML_UNSAFE_LEADING or s.endswith(":"):
        return True
    if any(bad in s for bad in _YAML_UNSAFE):
        return True
    # A string that reads back as a number must be quoted to stay a string.
    try:
        float(s)
    except ValueError:
        return False
    return True


def _scalar(v) -> str:
    """One YAML scalar, quoted whenever a plain one would not round-trip.

    Quoting uses ``json.dumps`` because JSON is a subset of YAML 1.2, so its
    double-quoted form (with ``\\n`` escapes and \\u escapes) is already valid
    YAML — no separate escaping logic to get wrong.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False) if _needs_quotes(v) else v
    if v is None:
        return "null"
    return str(v)


def write_yaml(path: str, csv_name: str, meta: CurveMeta) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_to_yaml(build_descriptor(csv_name, meta)))


def write_datapackage(out_dir: str, data: np.ndarray, meta: CurveMeta,
                      *, yaml: bool = True) -> dict[str, str]:
    """Write CSV + JSON (+ optional YAML). Returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    base = meta.name
    csv_path = os.path.join(out_dir, base + ".csv")
    json_path = os.path.join(out_dir, base + ".json")
    write_csv(csv_path, data, meta)
    write_json(json_path, csv_path, meta)
    paths = {"csv": csv_path, "json": json_path}
    if yaml:
        yaml_path = os.path.join(out_dir, base + ".yaml")
        write_yaml(yaml_path, csv_path, meta)
        paths["yaml"] = yaml_path
    return paths
