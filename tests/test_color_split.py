"""Tests for raster multi-colour curve splitting."""
import numpy as np

from cvdigitize.raster_extract import split_color_curves, _hue_name


def _draw_curve(img, ys, color, thickness=2):
    h, w, _ = img.shape
    for x in range(w):
        y = int(ys[x])
        img[max(0, y - thickness):min(h, y + thickness + 1), x] = color


def _canvas(h=200, w=400):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_two_colour_curves_split():
    img = _canvas()
    xs = np.arange(400)
    _draw_curve(img, 80 + 40 * np.sin(xs / 40), (220, 20, 20))    # red
    _draw_curve(img, 120 + 40 * np.sin(xs / 55), (20, 40, 220))   # blue
    curves = split_color_curves(img)
    names = sorted(c["name"] for c in curves)
    assert names == ["blue", "red"]
    for c in curves:
        assert np.ptp(c["polyline_px"][:, 0]) > 0.8 * 400   # spans the plot


def test_dark_curve_survives_gradient_fill():
    """A black curve over a saturated gradient fill must still come out as ONE
    dark curve, and the fill must not produce fake colour curves."""
    img = _canvas()
    xs = np.arange(400)
    ys = 100 + 60 * np.sin(xs / 50)
    # gradient fill below the curve (saturated, hue sweeping)
    for x in range(400):
        hue_rgb = (255, int(255 * x / 400), 60)
        img[int(ys[x]) + 3:180, x] = hue_rgb
    _draw_curve(img, ys, (0, 0, 0))
    curves = split_color_curves(img)
    assert len(curves) == 1
    assert curves[0]["name"] == "dark"
    assert np.ptp(curves[0]["polyline_px"][:, 0]) > 0.8 * 400


def test_fill_only_yields_nothing():
    img = _canvas()
    img[50:150, 100:300] = (230, 60, 60)    # a fat filled rectangle
    assert split_color_curves(img) == []


def test_hue_names():
    assert _hue_name(5) == "red"
    assert _hue_name(120) == "green"
    assert _hue_name(230) == "blue"
    assert _hue_name(350) == "red"
