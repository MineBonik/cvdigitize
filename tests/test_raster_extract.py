"""Unit tests for the raster (M2) extraction pipeline, using small synthetic
images built with cv2 drawing primitives — no PDF/file I/O needed.
"""
import cv2
import numpy as np

from cvdigitize.raster_extract import (
    _long_runs, _merge_collinear_segments, _cluster_peaks,
    detect_frame_bbox, detect_all_frames, detect_axis_ticks,
    mask_dark_curve, largest_component, skeletonize_curve, skeleton_to_polyline,
    _erase_straight_lines, extract_frameless_curve,
)


def _draw_frame(img, box, color=(0, 0, 0), thickness=2):
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)


def _blank(h=300, w=400):
    return np.full((h, w, 3), 255, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def test_long_runs_finds_contiguous_true_spans():
    arr = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0], dtype=bool)
    runs = _long_runs(arr, minlen=3)
    assert runs == [(1, 4), (6, 11)]


def test_long_runs_respects_minlen():
    arr = np.array([1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    assert _long_runs(arr, minlen=4) == [(3, 8)]


def test_cluster_peaks_groups_nearby_and_separates_far():
    peaks = _cluster_peaks(np.array([10, 11, 12, 50, 51]))
    assert peaks == [11, 50]  # (10+11+12)/3=11, (50+51)/2=50.5->50


def test_merge_collinear_segments_merges_overlapping_span():
    # two rows of the same border, 1px apart, near-identical span -> one line
    segs = [(10, 5, 100), (11, 6, 99)]
    lines = _merge_collinear_segments(segs)
    assert len(lines) == 1
    pos, a0, a1 = lines[0]
    assert 10 <= pos <= 11


def test_merge_collinear_segments_keeps_nonoverlapping_apart():
    # same rows, but spans don't overlap -> different features, stay separate
    segs = [(10, 0, 20), (11, 200, 220)]
    lines = _merge_collinear_segments(segs)
    assert len(lines) == 2


# --------------------------------------------------------------------------- #
# frameless extraction (classic crossing-axis figures with no bounding box)
# --------------------------------------------------------------------------- #
def test_erase_straight_lines_removes_axes_keeps_curve():
    mask = np.zeros((200, 300), dtype=bool)
    mask[100, 10:290] = True          # long horizontal axis
    mask[10:190, 20] = True           # long vertical axis
    # a short, locally-curvy stroke: never a long single-row/col run
    for x in range(40, 260):
        mask[120 + int(20 * np.sin(x / 20)), x] = True
    cleaned = _erase_straight_lines(mask)
    # axis row and column gone
    assert cleaned[100, 150] == False
    assert cleaned[50, 20] == False
    # curve pixels survive
    assert cleaned[:, 40:260].any()


def test_extract_frameless_curve_traces_sine_without_frame():
    img = np.full((240, 360, 3), 255, dtype=np.uint8)
    # bare crossing axes (no box) + a wavy curve, all black
    cv2.line(img, (30, 120), (340, 120), (0, 0, 0), 2)   # potential axis
    cv2.line(img, (30, 20), (30, 220), (0, 0, 0), 2)     # current axis
    xs = np.arange(40, 330)
    ys = (120 - 60 * np.sin((xs - 40) / 45)).astype(int)
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 0), -1)
    poly = extract_frameless_curve(img)
    assert len(poly) >= 50
    # spans most of the curve's x-range
    assert np.ptp(poly[:, 0]) > 0.6 * (xs.max() - xs.min())


def test_extract_frameless_curve_empty_on_blank():
    img = np.full((120, 160, 3), 255, dtype=np.uint8)
    assert len(extract_frameless_curve(img)) == 0


# --------------------------------------------------------------------------- #
# frame detection
# --------------------------------------------------------------------------- #
def test_detect_frame_bbox_finds_single_box():
    img = _blank()
    box = (50, 40, 350, 260)
    _draw_frame(img, box)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = detect_frame_bbox(gray)
    assert found is not None
    for a, b in zip(found, box):
        assert abs(a - b) <= 3


def test_detect_all_frames_finds_two_stacked_panels():
    img = _blank(h=600, w=400)
    box_top = (50, 30, 350, 260)
    box_bottom = (50, 320, 350, 550)
    _draw_frame(img, box_top)
    _draw_frame(img, box_bottom)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frames = detect_all_frames(gray)
    assert len(frames) == 2
    frames_sorted = sorted(frames, key=lambda f: f[1])
    for found, expected in zip(frames_sorted, [box_top, box_bottom]):
        for a, b in zip(found, expected):
            assert abs(a - b) <= 3


def test_detect_all_frames_ignores_thick_curved_region():
    """A thick filled blob (like a wide curve peak) must not be mistaken for a frame."""
    img = _blank(h=300, w=400)
    box = (50, 40, 350, 260)
    _draw_frame(img, box)
    # a thick black blob spanning many rows with long horizontal extent
    cv2.rectangle(img, (100, 100), (300, 180), (0, 0, 0), -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frames = detect_all_frames(gray)
    assert len(frames) == 1
    for a, b in zip(frames[0], box):
        assert abs(a - b) <= 3


# --------------------------------------------------------------------------- #
# tick detection
# --------------------------------------------------------------------------- #
def test_detect_axis_ticks_finds_evenly_spaced_ticks():
    # Ticks inset from the frame corners, like a real axes (matplotlib always
    # leaves a margin) — a tick drawn exactly at the corner is indistinguishable
    # from border bleed and is deliberately excluded, see edge_margin.
    img = _blank(h=300, w=400)
    box = (60, 40, 340, 240)
    _draw_frame(img, box)
    x0, y0, x1, y1 = box
    xt_true = [x0 + int((x1 - x0) * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    for x in xt_true:
        cv2.line(img, (x, y1 + 3), (x, y1 + 8), (0, 0, 0), 1)
    yt_true = [y0 + int((y1 - y0) * f) for f in (0.15, 0.5, 0.85)]
    for y in yt_true:
        cv2.line(img, (x0 - 8, y), (x0 - 3, y), (0, 0, 0), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ticks = detect_axis_ticks(gray, box)
    assert len(ticks["x_ticks"]) == 5
    assert len(ticks["y_ticks"]) == 3
    for found, expected in zip(ticks["x_ticks"], xt_true):
        assert abs(found - expected) <= 3


# --------------------------------------------------------------------------- #
# curve isolation: mask -> component -> skeleton -> ordered polyline
# --------------------------------------------------------------------------- #
def test_mask_dark_curve_ignores_saturated_fill():
    img = _blank(h=100, w=100)
    img[:, :] = (0, 200, 255)  # a saturated orange fill (BGR), bright
    cv2.line(img, (10, 50), (90, 50), (0, 0, 0), 2)  # black curve across it
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = mask_dark_curve(rgb, value_thresh=0.55)
    # the black line pixels are dark; the orange fill is bright -> excluded
    assert mask[50, 50]
    assert not mask[10, 10]


def test_largest_component_drops_small_specks():
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 10:90] = True   # big blob (curve stand-in)
    mask[5:8, 5:8] = True       # small speck (tick mark stand-in)
    biggest = largest_component(mask)
    assert biggest.sum() == mask[40:60, 10:90].sum()
    assert not biggest[6, 6]


def test_skeleton_to_polyline_orders_a_simple_line():
    mask = np.zeros((50, 100), dtype=bool)
    mask[25, 10:90] = True  # a straight horizontal 1px line
    skel = skeletonize_curve(mask)
    poly = skeleton_to_polyline(skel)
    assert len(poly) >= 70
    # endpoints of the ordered path should be the two ends of the line
    xs = poly[:, 0]
    assert (xs[0] <= 12 and xs[-1] >= 88) or (xs[0] >= 88 and xs[-1] <= 12)


def test_skeleton_to_polyline_prunes_short_spur():
    mask = np.zeros((60, 100), dtype=bool)
    mask[25, 10:90] = True   # main line
    mask[25:33, 50] = True   # a short spur sticking down from the middle
    skel = skeletonize_curve(mask)
    poly = skeleton_to_polyline(skel, max_spur_len=15)
    # pruned result should stay close to the main line's y (spur removed)
    assert poly[:, 1].max() - poly[:, 1].min() <= 3
