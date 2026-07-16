"""Read an echemdb ``svgdigitizer`` SVG as a ground-truth reference curve.

Each echemdb figure is stored as an Inkscape SVG that carries three things we
need to reconstruct the digitized curve in real units, with no source PNG and
no svgdigitizer dependency:

  * **the traced curve** — the single ``<path>`` whose stroke is a bright colour
    (cyan/magenta/red/green ...); every other path is black axis/marker markup;
  * **four calibration markers** — short black paths, each tagged by a text
    label ``E1: <value> V vs <ref>`` / ``E2: ...`` / ``j1: <value> <unit>`` /
    ``j2: ...``. The *start point* of each marker path (after its group
    transform) is the pixel location of that axis value (svgdigitizer's
    convention);
  * **the source page** — ``<image xlink:href="..._p<N>.png">`` names the PDF
    page the figure was traced from.

From the two x-markers (E1, E2) and two y-markers (j1, j2) we build decoupled
linear pixel->data maps (axes are curator-aligned to the pixel grid) and apply
them to the flattened curve, yielding an ``(E, j)`` reference polyline. Units
and reference electrode are recorded but do not matter for a normalised-shape
benchmark; converting to real units is what fixes the SVG y-axis flip so the
reference has the same orientation as our extracted curve.

This module is deliberately self-contained (lxml + numpy): a small SVG path
flattener and affine-transform walker, so the benchmark has zero new deps.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from lxml import etree

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"

# a numeric token in an SVG path "d" string or a transform argument
_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_CMD = re.compile(r"[MmLlHhVvCcSsQqTtZzAa]")
# axis reference: E1/E2/j1/j2, also i1/i2/I1 (current) and x1/x2/y1/y2 aliases
_LABEL = re.compile(
    r"(?P<key>[EejJiIxXyY][12])\s*:\s*(?P<value>[-+]?[\d.]+)\s*(?P<unit>[^\s]+)?"
    r"(?:\s*(?:vs\.?)\s*(?P<ref>[^\s].*))?",
    re.IGNORECASE,
)
# scale-bar calibration: "I_scale_bar: 10 uA" / "j_scale_bar: 0.1 mA / cm2"
_SCALEBAR = re.compile(
    r"(?P<axis>[EejJiIxXyY])_scale_?bar\s*:\s*(?P<value>[-+]?[\d.]+)\s*(?P<unit>[^\s]+)?",
    re.IGNORECASE,
)


def _axis_of(key: str) -> str:
    """Normalise a reference key letter to axis 'e' (potential) or 'j' (current)."""
    c = key[0].lower()
    if c in ("e", "x"):
        return "e"
    return "j"  # j, i (current), y


@dataclass
class ReferenceCurve:
    """A ground-truth curve parsed from an echemdb SVG, in real (E, j) units."""
    xy: np.ndarray                 # (N, 2) columns E, j
    page: int | None               # source PDF page (1-based, as named in the PNG)
    figure: str                    # e.g. "2a1"
    curve_label: str               # e.g. "black", "blue"
    name: str                      # source SVG basename (no extension)
    E_unit: str = "V"
    E_ref: str = ""
    j_unit: str = ""
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# affine transforms
# ----------------------------------------------------------------------------
def _identity() -> np.ndarray:
    return np.eye(3)


def _parse_transform(text: str) -> np.ndarray:
    """Compose a 3x3 affine from an SVG ``transform`` attribute.

    Handles translate / scale / matrix / rotate applied left-to-right (SVG
    order), which covers everything the echemdb curators emit.
    """
    m = _identity()
    for name, args in re.findall(r"(translate|scale|matrix|rotate)\s*\(([^)]*)\)", text):
        v = [float(x) for x in _NUM.findall(args)]
        t = _identity()
        if name == "translate":
            t[0, 2] = v[0]
            t[1, 2] = v[1] if len(v) > 1 else 0.0
        elif name == "scale":
            sx = v[0]
            sy = v[1] if len(v) > 1 else sx
            t[0, 0], t[1, 1] = sx, sy
        elif name == "matrix" and len(v) == 6:
            a, b, c, d, e, f = v
            t = np.array([[a, c, e], [b, d, f], [0, 0, 1]], float)
        elif name == "rotate":
            ang = np.radians(v[0])
            cos, sin = np.cos(ang), np.sin(ang)
            t = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], float)
            if len(v) == 3:  # rotate around (cx, cy)
                cx, cy = v[1], v[2]
                pre = _identity(); pre[0, 2], pre[1, 2] = cx, cy
                post = _identity(); post[0, 2], post[1, 2] = -cx, -cy
                t = pre @ t @ post
        m = m @ t
    return m


def _apply(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine to an (N, 2) array of points."""
    pts = np.atleast_2d(pts)
    homo = np.column_stack([pts, np.ones(len(pts))])
    out = homo @ mat.T
    return out[:, :2]


# ----------------------------------------------------------------------------
# SVG path "d" flattening
# ----------------------------------------------------------------------------
def _tokenize_path(d: str):
    """Yield (command, [floats]) tuples from an SVG path data string."""
    i = 0
    n = len(d)
    while i < n:
        mcmd = _CMD.search(d, i)
        if not mcmd:
            break
        cmd = mcmd.group()
        j = mcmd.end()
        nxt = _CMD.search(d, j)
        seg = d[j:nxt.start()] if nxt else d[j:]
        nums = [float(x) for x in _NUM.findall(seg)]
        yield cmd, nums
        i = nxt.start() if nxt else n


def _cubic(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[1:, None]
    mt = 1 - t
    return (mt**3) * p0 + 3 * (mt**2) * t * p1 + 3 * mt * (t**2) * p2 + (t**3) * p3


def _quad(p0, p1, p2, n=16):
    t = np.linspace(0, 1, n)[1:, None]
    mt = 1 - t
    return (mt**2) * p0 + 2 * mt * t * p1 + (t**2) * p2


def flatten_path(d: str, bezier_samples: int = 16) -> np.ndarray:
    """Flatten an SVG path ``d`` string into an (N, 2) polyline of points.

    Supports M/L/H/V/C/S/Q/T/Z (absolute and relative). Beziers are sampled;
    arcs (A) are approximated by their endpoint (rare in echemdb curves).
    """
    pts: list[np.ndarray] = []
    cur = np.zeros(2)
    start = np.zeros(2)
    prev_ctrl = None      # for S/T smooth continuations
    prev_cmd = ""
    for cmd, nums in _tokenize_path(d):
        rel = cmd.islower()
        c = cmd.upper()
        k = 0
        if c == "M":
            # first pair is a moveto, subsequent pairs are implicit linetos
            first = True
            while k + 1 < len(nums):
                p = np.array(nums[k:k + 2], float)
                cur = cur + p if rel else p
                if first:
                    start = cur.copy()
                    first = False
                pts.append(cur.copy())
                k += 2
            prev_ctrl = None
        elif c == "L":
            while k + 1 < len(nums):
                p = np.array(nums[k:k + 2], float)
                cur = cur + p if rel else p
                pts.append(cur.copy())
                k += 2
            prev_ctrl = None
        elif c == "H":
            for x in nums:
                cur = np.array([cur[0] + x if rel else x, cur[1]])
                pts.append(cur.copy())
            prev_ctrl = None
        elif c == "V":
            for y in nums:
                cur = np.array([cur[0], cur[1] + y if rel else y])
                pts.append(cur.copy())
            prev_ctrl = None
        elif c == "C":
            while k + 5 < len(nums):
                c1 = cur + nums[k:k + 2] if rel else np.array(nums[k:k + 2])
                c2 = cur + nums[k + 2:k + 4] if rel else np.array(nums[k + 2:k + 4])
                end = cur + nums[k + 4:k + 6] if rel else np.array(nums[k + 4:k + 6])
                pts.extend(_cubic(cur, c1, c2, end, bezier_samples))
                cur, prev_ctrl = end, c2
                k += 6
        elif c == "S":
            while k + 3 < len(nums):
                c1 = 2 * cur - prev_ctrl if (prev_ctrl is not None and prev_cmd in "CS") else cur
                c2 = cur + nums[k:k + 2] if rel else np.array(nums[k:k + 2])
                end = cur + nums[k + 2:k + 4] if rel else np.array(nums[k + 2:k + 4])
                pts.extend(_cubic(cur, c1, c2, end, bezier_samples))
                cur, prev_ctrl = end, c2
                k += 4
        elif c == "Q":
            while k + 3 < len(nums):
                c1 = cur + nums[k:k + 2] if rel else np.array(nums[k:k + 2])
                end = cur + nums[k + 2:k + 4] if rel else np.array(nums[k + 2:k + 4])
                pts.extend(_quad(cur, c1, end, bezier_samples))
                cur, prev_ctrl = end, c1
                k += 4
        elif c == "T":
            while k + 1 < len(nums):
                c1 = 2 * cur - prev_ctrl if (prev_ctrl is not None and prev_cmd in "QT") else cur
                end = cur + nums[k:k + 2] if rel else np.array(nums[k:k + 2])
                pts.extend(_quad(cur, c1, end, bezier_samples))
                cur, prev_ctrl = end, c1
                k += 2
        elif c == "A":
            # arc: approximate by jumping to each endpoint
            while k + 6 < len(nums):
                end = cur + nums[k + 5:k + 7] if rel else np.array(nums[k + 5:k + 7])
                pts.append(end.copy())
                cur = end
                k += 7
            prev_ctrl = None
        elif c == "Z":
            pts.append(start.copy())
            cur = start.copy()
            prev_ctrl = None
        prev_cmd = c
    if not pts:
        return np.empty((0, 2))
    return np.asarray(pts, float)


# ----------------------------------------------------------------------------
# SVG walking
# ----------------------------------------------------------------------------
def _local(tag) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _stroke_of(el) -> str | None:
    style = el.get("style", "")
    m = re.search(r"stroke\s*:\s*(#[0-9a-fA-F]{3,6}|[a-zA-Z]+)", style)
    if m:
        return m.group(1).lower()
    return (el.get("stroke") or "").lower() or None


def _is_curve_stroke(stroke: str | None) -> bool:
    """True for a bright traced-curve stroke, False for black/white/none/gray."""
    if not stroke or stroke in ("none", "black", "white"):
        return False
    if stroke.startswith("#"):
        h = stroke[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) != 6:
            return False
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        mx, mn = max(r, g, b), min(r, g, b)
        if mx < 40:                      # ~black
            return False
        if mn > 200 and mx - mn < 30:    # ~white/light gray
            return False
        return (mx - mn) > 40 or mx > 120  # has colour, or a solid mid tone
    return True  # a named colour like "red"


def _text_content(el) -> str:
    return "".join(el.itertext()).strip()


def _walk(el, mat, curves, markers, images):
    """Depth-first walk accumulating transforms; collect paths/labels/images."""
    tf = el.get("transform")
    if tf:
        mat = mat @ _parse_transform(tf)
    tag = _local(el.tag)
    if tag == "path":
        d = el.get("d")
        if d:
            stroke = _stroke_of(el)
            pts = flatten_path(d)
            if len(pts):
                curves.append((stroke, _apply(mat, pts)))
    elif tag == "text":
        txt = _text_content(el)
        markers.append((txt, mat, el))
    elif tag == "image":
        href = el.get(f"{{{_XLINK_NS}}}href") or el.get("href")
        if href:
            images.append(href)
    for child in el:
        _walk(child, mat, curves, markers, images)


def _marker_point(label_el, label_mat) -> np.ndarray | None:
    """The reference pixel point for a calibration label: the start of the
    marker path that shares the label's parent <g>, after transforms.

    ``label_mat`` already includes the group transform (the walk applied it
    before descending into the group's children), and the sibling marker path
    lives in that same group, so it shares the transform. We resolve the
    sibling directly from the element tree rather than by object identity —
    lxml recreates element proxies, so ``id()`` is not stable across walks.
    """
    pts = _marker_path(label_el, label_mat)
    return pts[0] if pts is not None else None


def _marker_path(label_el, label_mat) -> np.ndarray | None:
    """Full transformed points of the marker path sharing the label's <g>."""
    parent = label_el.getparent()
    if parent is None:
        return None
    for child in parent:
        if _local(child.tag) == "path" and child.get("d"):
            pts = _apply(label_mat, flatten_path(child.get("d")))
            if len(pts):
                return pts
    return None


def _page_from_href(href: str) -> int | None:
    m = re.search(r"_p(\d+)\.png$", href, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _linear(px_lo, v_lo, px_hi, v_hi):
    if px_hi == px_lo:
        return None
    slope = (v_hi - v_lo) / (px_hi - px_lo)
    return lambda px: v_lo + (px - px_lo) * slope


def parse_reference_svg(path: str) -> ReferenceCurve | None:
    """Parse an echemdb SVG into a :class:`ReferenceCurve` in real (E, j) units.

    Returns None if the curve or a full x/y calibration cannot be recovered.
    """
    tree = etree.parse(path)
    root = tree.getroot()
    curves: list[tuple[str | None, np.ndarray]] = []
    markers: list[tuple[dict, np.ndarray, object]] = []
    images: list[str] = []
    _walk(root, _identity(), curves, markers, images)

    # ---- the traced curve ----------------------------------------------
    # Usually a bright-coloured stroke, but some curators trace in plain black
    # (same colour as the axis markers). Calibration markers are 2-point line
    # segments while the curve is hundreds of points, so: prefer the longest
    # bright-coloured path, else fall back to the longest path of any colour
    # (the point-count floor excludes the short markers either way).
    MIN_CURVE_PTS = 20
    colored = [(s, p) for s, p in curves
               if _is_curve_stroke(s) and len(p) >= MIN_CURVE_PTS]
    if colored:
        curve_px = max(colored, key=lambda sp: len(sp[1]))[1]
    else:
        longish = [p for _, p in curves if len(p) >= MIN_CURVE_PTS]
        if not longish:
            return None
        curve_px = max(longish, key=len)

    # ---- classify text labels into axis references + scale bars ----
    refs: dict[str, dict] = {}          # e1/e2/j1/j2 -> {pt, value, unit, ref}
    scalebars: dict[str, dict] = {}     # 'e'/'j' -> {length_px, value, unit}
    for txt, mat, el in markers:
        mb = _SCALEBAR.match(txt)
        if mb:
            pts = _marker_path(el, mat)
            if pts is not None and len(pts) >= 2:
                axis = _axis_of(mb.group("axis"))
                span = pts[:, 0] if axis == "e" else pts[:, 1]
                scalebars[axis] = {"length_px": float(abs(span.max() - span.min())),
                                   "value": float(mb.group("value")),
                                   "unit": mb.group("unit") or ""}
            continue
        m = _LABEL.match(txt)
        if not m or not m.group("key"):
            continue
        axis = _axis_of(m.group("key"))
        idx = m.group("key")[1]                 # '1' or '2'
        pt = _marker_point(el, mat)
        if pt is None:
            continue
        try:
            val = float(m.group("value"))
        except (TypeError, ValueError):
            continue
        refs[f"{axis}{idx}"] = {"pt": pt, "value": val,
                                "unit": m.group("unit") or "",
                                "ref": (m.group("ref") or "").strip()}

    # ---- x (potential) map: needs two references ----
    if not {"e1", "e2"} <= set(refs):
        return None
    fx = _linear(refs["e1"]["pt"][0], refs["e1"]["value"],
                 refs["e2"]["pt"][0], refs["e2"]["value"])
    if fx is None:
        return None

    # ---- y (current) map: two references, else scale-bar + zero, else flip ----
    j_unit = ""
    y_calibrated = True
    if {"j1", "j2"} <= set(refs):
        fy = _linear(refs["j1"]["pt"][1], refs["j1"]["value"],
                     refs["j2"]["pt"][1], refs["j2"]["value"])
        j_unit = refs["j1"]["unit"]
    elif "j" in scalebars and "j1" in refs and scalebars["j"]["length_px"] > 0:
        # scale bar gives px-per-unit; single reference gives the zero level.
        # screen y increases downward, so higher current = smaller py.
        per_px = scalebars["j"]["value"] / scalebars["j"]["length_px"]
        py0, v0 = refs["j1"]["pt"][1], refs["j1"]["value"]
        fy = lambda py: v0 + (py0 - py) * per_px
        j_unit = scalebars["j"]["unit"]
    else:
        fy, y_calibrated = (lambda py: -py), False   # shape-only orientation

    if fx is None or fy is None:
        return None

    E = fx(curve_px[:, 0])
    j = fy(curve_px[:, 1])
    xy = np.column_stack([E, j])

    base = os.path.splitext(os.path.basename(path))[0]
    fig, label = _parse_name(base)
    page = _page_from_href(images[0]) if images else None
    return ReferenceCurve(
        xy=xy, page=page, figure=fig, curve_label=label, name=base,
        E_unit="V", E_ref=refs["e1"]["ref"], j_unit=j_unit,
        meta={"n_points": len(xy), "y_calibrated": y_calibrated},
    )


def _parse_name(base: str) -> tuple[str, str]:
    """Split an echemdb basename ``<key>_f<fig>_<label>`` -> (fig, label)."""
    m = re.search(r"_f([0-9a-z]+)_([a-z0-9]+)$", base, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def load_entry_references(entry_dir: str) -> list[ReferenceCurve]:
    """Parse every ``*.svg`` in an echemdb entry directory."""
    import glob
    out = []
    for svg in sorted(glob.glob(os.path.join(entry_dir, "*.svg"))):
        try:
            ref = parse_reference_svg(svg)
        except Exception:
            ref = None
        if ref is not None and len(ref.xy) >= 10:
            out.append(ref)
    return out
