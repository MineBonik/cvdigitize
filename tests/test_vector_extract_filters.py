"""Regression tests for vector_extract's frame-line/text-column exclusion
filters. Each covers a real bug found while digitizing a second real paper
whose PDF uses different conventions than the first reference paper:
axis borders drawn as thin filled rectangles (both via the ``re`` opcode and
as 4 ``l`` lineto ops), and body text rendered with *stroke* colour landing
in the same colour bucket as a real stroked curve.
"""
import numpy as np

from cvdigitize.vector_extract import (
    _is_axis_aligned_rect, _is_thin_line_polyline, _is_narrow_tall_column,
    _is_tick_mark, _items_to_polylines,
)


def _rect_points(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)


def test_axis_aligned_rect_detects_perfect_rectangle():
    assert _is_axis_aligned_rect(_rect_points(10, 20, 11, 220))


def test_axis_aligned_rect_rejects_curve_like_points():
    # points that vary continuously in both x and y, not snapping to 2 levels
    t = np.linspace(0, 1, 20)
    poly = np.column_stack([10 + t, 20 + 50 * np.sin(t * 3)])
    assert not _is_axis_aligned_rect(poly)


def test_thin_line_polyline_flags_frame_border():
    # a real frame border: thin (0.8pt) and long (210pt), 5-point rectangle
    poly = _rect_points(147.4, 474.0, 148.2, 684.1)
    assert _is_thin_line_polyline(poly)


def test_thin_line_polyline_ignores_small_marker():
    # a small data-point marker rect: both dimensions small
    poly = _rect_points(100, 100, 100.85, 104.46)
    assert not _is_thin_line_polyline(poly)


def test_thin_line_polyline_ignores_curve_shape_even_if_thin_bbox():
    # thin-and-long bbox, but not an axis-aligned rectangle (real curve shape)
    t = np.linspace(0, 1, 20)
    poly = np.column_stack([10 + 0.5 * np.sin(t * 10), 20 + 200 * t])
    assert not _is_thin_line_polyline(poly)


def test_narrow_tall_column_flags_text_like_shape():
    # many points, narrow width, tall height -> stroked body text / column rule
    ys = np.linspace(200, 500, 200)
    xs = 375 + 3 * np.sin(ys)  # wiggles within a narrow column, like glyphs
    poly = np.column_stack([xs, ys])
    assert _is_narrow_tall_column(poly)


def test_narrow_tall_column_spares_real_steep_peak():
    # a real steep CV peak: narrow-ish but well under the height safety margin
    poly = np.column_stack([np.full(20, 200.0), np.linspace(150, 190, 20)])
    assert not _is_narrow_tall_column(poly)


def test_narrow_tall_column_spares_wide_curve():
    poly = np.column_stack([np.linspace(0, 300, 20), np.full(20, 50.0)])
    assert not _is_narrow_tall_column(poly)


def test_tick_mark_flags_tiny_isolated_dash():
    # a real axis tick: one 2-point segment, ~3pt long
    poly = np.array([[158.35, 290.02], [158.35, 286.79]])
    assert _is_tick_mark(poly)


def test_tick_mark_spares_larger_short_segment():
    # short (3 points) but spans much more than a tick ever would
    poly = np.array([[0.0, 0.0], [20.0, 30.0], [40.0, 60.0]])
    assert not _is_tick_mark(poly)


def test_tick_mark_spares_long_curve_fragment():
    # many points -> a real (if coarse) curve fragment, not a tick
    poly = np.column_stack([np.linspace(0, 5, 10), np.linspace(0, 5, 10)])
    assert not _is_tick_mark(poly)


class _P:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_items_to_polylines_drops_thin_frame_rectangle_from_l_ops():
    """The exact real-world shape: a frame border drawn as 4 lineto ops."""
    items = [
        ("l", _P(147.42, 474.01), _P(147.42, 684.10)),
        ("l", _P(147.42, 684.10), _P(148.24, 684.10)),
        ("l", _P(148.24, 684.10), _P(148.24, 474.01)),
        ("l", _P(148.24, 474.01), _P(147.42, 474.01)),
    ]
    polylines = _items_to_polylines(items)
    assert polylines == [] or all(len(p) == 0 for p in polylines)


def test_items_to_polylines_keeps_real_curve_from_l_ops():
    t = np.linspace(0, 2 * np.pi, 30)
    pts = [(50 + 40 * np.cos(a), 50 + 40 * np.sin(a)) for a in t]
    items = [("l", _P(*pts[i]), _P(*pts[i + 1])) for i in range(len(pts) - 1)]
    polylines = _items_to_polylines(items)
    assert len(polylines) == 1
    assert len(polylines[0]) == len(pts)
