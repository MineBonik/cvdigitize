"""Tests for guided extraction (rough human trace -> pixel-accurate curve)."""
import cv2
import numpy as np

from cvdigitize.guided import (extract_near_guide, extract_guides, _corridor_mask,
                               split_strokes)


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


# --------------------------------------------------------------------------- #
# multi-stroke guides: a real hand trace is drawn as several separate
# mouse-drags (pen lifted between them), which must NOT be treated as one
# continuous path — a flattened join would draw a straight "teleport" through
# whatever ink happens to lie between the drags (a real bug caught by hand).
# --------------------------------------------------------------------------- #
def test_split_strokes_finds_the_pen_lift_gap():
    # two dense runs of points with one big jump between them
    a = np.column_stack([np.linspace(0, 100, 40), np.full(40, 10.0)])
    b = np.column_stack([np.linspace(300, 400, 40), np.full(40, 10.0)])  # far away
    flat = np.vstack([a, b])
    strokes = split_strokes(flat)
    assert len(strokes) == 2
    assert len(strokes[0]) == 40 and len(strokes[1]) == 40


def test_split_strokes_keeps_single_smooth_path_whole():
    smooth = np.column_stack([np.linspace(0, 200, 100), np.linspace(0, 50, 100)])
    assert len(split_strokes(smooth)) == 1


def test_flattened_multi_stroke_guide_does_not_teleport():
    # simulate exactly the bug: two real strokes along curve A, flattened with
    # no boundary marker (as if the person lifted the pen partway and resumed
    # drawing nearby — a small, realistic gap, not a semantic jump elsewhere)
    img = _two_crossing_curves()

    def yA(x):
        return 150 - 60 * np.sin(x / 50)
    left = np.column_stack([np.linspace(30, 240, 15), yA(np.linspace(30, 240, 15))])
    right = np.column_stack([np.linspace(255, 470, 15), yA(np.linspace(255, 470, 15))])
    flat_guide = np.vstack([left, right])   # small gap in the middle, unmarked

    out = extract_near_guide(img, flat_guide)
    eA = np.mean([abs(p[1] - yA(p[0])) for p in out])
    # must still snap to the real ink everywhere, not a straight line across the gap
    assert eA < 3.0


def test_explicit_strokes_param_matches_two_drags():
    img = _two_crossing_curves()

    def yA(x):
        return 150 - 60 * np.sin(x / 50)
    s1 = np.column_stack([np.linspace(30, 240, 15), yA(np.linspace(30, 240, 15))])
    s2 = np.column_stack([np.linspace(255, 470, 15), yA(np.linspace(255, 470, 15))])
    out = extract_near_guide(img, None, strokes=[s1, s2])
    assert len(out) > 50
    eA = np.mean([abs(p[1] - yA(p[0])) for p in out])
    assert eA < 3.0


def test_large_gap_between_strokes_still_routes_to_correct_curve():
    # a genuinely big pen-lift gap: the two pieces can't be joined with full
    # accuracy across the untraced span, but stitching must still pick curve A
    # (its own two pieces) over jumping to the unrelated crossing curve B
    img = _two_crossing_curves()

    def yA(x):
        return 150 - 60 * np.sin(x / 50)

    def yB(x):
        return 150 + 60 * np.sin(x / 50)
    left = np.column_stack([np.linspace(30, 150, 12), yA(np.linspace(30, 150, 12))])
    right = np.column_stack([np.linspace(350, 470, 12), yA(np.linspace(350, 470, 12))])
    out = extract_near_guide(img, np.vstack([left, right]))
    assert len(out) > 20
    eA = np.mean([abs(p[1] - yA(p[0])) for p in out])
    eB = np.mean([abs(p[1] - yB(p[0])) for p in out])
    assert eA < eB   # decisively curve A's own pieces, not a detour through B


def test_extract_guides_accepts_strokes_format():
    img = _two_crossing_curves()

    def yA(x):
        return 150 - 60 * np.sin(x / 50)
    s1 = np.column_stack([np.linspace(30, 200, 12), yA(np.linspace(30, 200, 12))]).tolist()
    s2 = np.column_stack([np.linspace(300, 470, 12), yA(np.linspace(300, 470, 12))]).tolist()
    res = extract_guides(img, [{"name": "A", "strokes": [s1, s2]}])
    assert len(res) == 1 and len(res[0]["polyline_px"]) > 50
