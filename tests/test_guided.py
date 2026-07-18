"""Tests for guided extraction (rough human trace -> pixel-accurate curve)."""
import cv2
import numpy as np

from cvdigitize.guided import extract_near_guide, extract_guides, _corridor_mask


def _two_crossing_curves():
    img = np.full((300, 500, 3), 255, np.uint8)
    xs = np.arange(20, 480)
    yA = (150 - 60 * np.sin(xs / 50)).astype(int)
    yB = (150 + 60 * np.sin(xs / 50)).astype(int)   # mirror image, crosses A
    for x, y in zip(xs, yA):
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 0), -1)
    for x, y in zip(xs, yB):
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 0), -1)
    return img


def test_corridor_mask_covers_guide():
    m = _corridor_mask((100, 100, 3), np.array([[10, 50], [90, 50]]), radius=5)
    assert m[50, 50] and not m[10, 50 + 40]  # on the line yes, far away no


def test_guided_follows_intended_curve_not_crossing_one():
    img = _two_crossing_curves()
    gx = np.linspace(30, 470, 15)
    gy = 150 - 60 * np.sin(gx / 50) + np.random.default_rng(0).normal(0, 6, 15)
    out = extract_near_guide(img, np.column_stack([gx, gy]))
    assert len(out) > 100

    def yA(x):
        return 150 - 60 * np.sin(x / 50)

    def yB(x):
        return 150 + 60 * np.sin(x / 50)
    eA = np.mean([abs(p[1] - yA(p[0])) for p in out])
    eB = np.mean([abs(p[1] - yB(p[0])) for p in out])
    assert eA < 3.0            # snaps to the real ink (pixel-accurate)
    assert eA < eB / 5         # follows A, decisively not the crossing B


def test_guided_is_precise_from_a_rough_guide():
    # even a very coarse 6-point guide with big wobble recovers the curve
    img = _two_crossing_curves()
    gx = np.linspace(40, 460, 6)
    gy = 150 - 60 * np.sin(gx / 50) + np.array([12, -10, 8, -12, 9, -7])
    out = extract_near_guide(img, np.column_stack([gx, gy]))
    err = np.mean([abs(p[1] - (150 - 60 * np.sin(p[0] / 50))) for p in out])
    # a very coarse guide still recovers the curve to a few % of its amplitude
    # (120 px peak-to-peak); a denser guide gets sub-pixel (see the test above)
    assert err < 8.0


def test_extract_guides_named():
    img = _two_crossing_curves()
    gx = np.linspace(30, 470, 12)
    guides = [{"name": "upper", "points": np.column_stack(
        [gx, 150 - 60 * np.sin(gx / 50)]).tolist()}]
    res = extract_guides(img, guides)
    assert len(res) == 1 and res[0]["name"] == "upper"
    assert len(res[0]["polyline_px"]) > 100


def test_empty_guide_returns_empty():
    img = _two_crossing_curves()
    assert len(extract_near_guide(img, [[10, 10]])) == 0
    # a guide over blank space finds no ink
    assert len(extract_near_guide(np.full((100, 100, 3), 255, np.uint8),
                                  [[10, 10], [90, 90]])) == 0
