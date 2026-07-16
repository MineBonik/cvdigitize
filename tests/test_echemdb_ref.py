"""Tests for the echemdb svgdigitizer-SVG ground-truth parser.

All fixtures are synthetic SVG strings built in-memory, so these run without
the (gitignored) echemdb dataset. They pin the two things the benchmark relies
on: the pixel->(E, j) calibration recovered from the four axis markers, and the
robust selection of the traced curve among black axis/marker paths.
"""
import numpy as np

from cvdigitize import echemdb_ref as er


# --------------------------------------------------------------------------- #
# path flattening + transforms
# --------------------------------------------------------------------------- #
def test_flatten_absolute_and_relative_lineto():
    a = er.flatten_path("M 0,0 L 10,0 L 10,10")
    b = er.flatten_path("m 0,0 l 10,0 l 0,10")
    assert np.allclose(a[0], [0, 0]) and np.allclose(a[-1], [10, 10])
    assert np.allclose(b[-1], [10, 10])


def test_flatten_cubic_bezier_endpoints():
    pts = er.flatten_path("M 0,0 C 0,10 10,10 10,0")
    assert np.allclose(pts[0], [0, 0])
    assert np.allclose(pts[-1], [10, 0], atol=1e-6)
    assert len(pts) > 4  # sampled, not just endpoints


def test_implicit_lineto_after_moveto():
    # a second coordinate pair after M is an implicit L
    pts = er.flatten_path("M 0,0 5,5 10,0")
    assert len(pts) == 3
    assert np.allclose(pts[1], [5, 5])


def test_parse_transform_translate_and_scale():
    m = er._parse_transform("translate(10,20)")
    assert np.allclose(er._apply(m, np.array([[1.0, 2.0]])), [[11, 22]])
    m2 = er._parse_transform("scale(2,3)")
    assert np.allclose(er._apply(m2, np.array([[1.0, 1.0]])), [[2, 3]])


def test_parse_transform_composition_order():
    # translate then scale: point (1,1) -> scale*(1,1)+translate
    m = er._parse_transform("translate(5,0) scale(2)")
    assert np.allclose(er._apply(m, np.array([[1.0, 1.0]])), [[7, 2]])


# --------------------------------------------------------------------------- #
# curve-stroke classification
# --------------------------------------------------------------------------- #
def test_is_curve_stroke_rejects_black_white_none():
    assert not er._is_curve_stroke("#000000")
    assert not er._is_curve_stroke("black")
    assert not er._is_curve_stroke("none")
    assert not er._is_curve_stroke("#ffffff")


def test_is_curve_stroke_accepts_bright_colours():
    for c in ("#00ffff", "#ff00ff", "#fd0000", "#00ff00", "red"):
        assert er._is_curve_stroke(c)


# --------------------------------------------------------------------------- #
# full SVG parsing
# --------------------------------------------------------------------------- #
def _svg(curve_d, curve_stroke="#00ffff", *, page=4, extra_markers=""):
    """A minimal echemdb-style SVG: image ref + 4 calibration markers + curve."""
    def marker(gx, gy, px, py, label):
        return f'''<g transform="translate({gx},{gy})">
          <path style="fill:none;stroke:#000000" d="m {px},{py} 10,2"/>
          <text x="0" y="0"><tspan>{label}</tspan></text></g>'''
    return f'''<svg xmlns="http://www.w3.org/2000/svg"
        xmlns:xlink="http://www.w3.org/1999/xlink" width="200" height="200">
      <image xlink:href="paper_p{page}.png" x="0" y="0" width="200" height="200"/>
      {marker(0, 0, 20, 180, "E1: 0.0 V vs RHE")}
      {marker(0, 0, 120, 180, "E2: 1.0 V vs RHE")}
      {marker(0, 0, 20, 150, "j1: -50 uA / cm2")}
      {marker(0, 0, 20, 50, "j2: 50 uA / cm2")}
      {extra_markers}
      <g><text x="0" y="0"><tspan>curve: black</tspan></text>
        <path style="fill:none;stroke:{curve_stroke};stroke-width:3.8" d="{curve_d}"/>
      </g>
    </svg>'''


def _write(tmp_path, svg, name="paper_f2a_black.svg"):
    p = tmp_path / name
    p.write_text(svg, encoding="utf-8")
    return str(p)


def _line_d(x0, y0, x1, y1, n=40):
    """A many-point polyline 'd' from (x0,y0) to (x1,y1) (real curves aren't
    2-point straight segments, and the parser floors curves at 20 points)."""
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def test_parse_recovers_calibration_and_page(tmp_path):
    # curve spanning the marker box: x 20..120 -> E 0..1, y 50..150 -> j 50..-50
    svg = _svg(_line_d(20, 50, 120, 150))
    ref = er.parse_reference_svg(_write(tmp_path, svg))
    assert ref is not None
    assert ref.page == 4
    assert ref.figure == "2a" and ref.curve_label == "black"
    E, j = ref.xy[:, 0], ref.xy[:, 1]
    # E1 marker at px=20 -> 0 V, E2 at px=120 -> 1 V
    assert abs(E.min() - 0.0) < 1e-6 and abs(E.max() - 1.0) < 1e-6
    # j2 at py=50 -> +50, j1 at py=150 -> -50 (screen y down => higher current up)
    assert abs(j.max() - 50.0) < 1e-6 and abs(j.min() + 50.0) < 1e-6


def test_curve_selected_even_when_black(tmp_path):
    # a long black curve must win over the short black axis markers
    d = "M " + " ".join(f"{20+i},{100+int(30*np.sin(i/3))}" for i in range(80))
    svg = _svg(d, curve_stroke="#000000")
    ref = er.parse_reference_svg(_write(tmp_path, svg))
    assert ref is not None
    assert len(ref.xy) >= 60


def test_missing_axis_marker_returns_none(tmp_path):
    # drop the E2 marker -> no x calibration -> None
    svg = _svg(_line_d(20, 50, 120, 150)).replace("E2: 1.0 V vs RHE", "note: nothing")
    assert er.parse_reference_svg(_write(tmp_path, svg)) is None


def test_scalebar_current_calibration(tmp_path):
    # markovic-style: I1 zero reference + a vertical scale bar of known value
    extra = '''<g transform="translate(0,0)">
        <path style="fill:none;stroke:#000000" d="m 15,60 0,40"/>
        <text x="0" y="0"><tspan>I_scale_bar: 10 uA</tspan></text></g>'''
    # replace the two j markers with a single I1 zero reference
    svg = _svg(_line_d(20, 50, 120, 150), extra_markers=extra)
    svg = svg.replace("j1: -50 uA / cm2", "I1: 0 uA").replace("j2: 50 uA / cm2", "note: x")
    ref = er.parse_reference_svg(_write(tmp_path, svg))
    assert ref is not None
    # 40 px of scale bar == 10 uA => 0.25 uA/px; curve spans py 50..150 (100 px)
    j = ref.xy[:, 1]
    assert abs((j.max() - j.min()) - 25.0) < 1e-6
