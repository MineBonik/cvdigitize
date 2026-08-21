"""Unit tests for plot-frame and tick detection on the rendered page.

These cover the image-based half of a deliberately vector-only tool: PDF path
geometry says nothing about which strokes are the axes, so the page is rendered
and the axes box found in pixels. `detect_all_frames` is what decides which
curves belong to which panel, so a regression here silently mis-assigns data.

Six of these were the only unit tests for these functions and were lost when
raster_extract.py was deleted and the code moved here verbatim; they are
restored from 22b8a26^ against the new module path.
"""
import cv2
import numpy as np

from cvdigitize.plotframe import (
    _cluster_peaks, _long_runs, _merge_collinear_segments,
    detect_all_frames, detect_axis_ticks,
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
# panel frame detection - decides which curves belong to which panel
# --------------------------------------------------------------------------- #


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


def test_detect_axis_ticks_handles_a_frame_flush_against_the_left_edge():
    """left=0 leaves no room for the outward tick band; the inward fallback
    must still find the real ticks rather than erroring on the empty slice."""
    img = _blank(h=300, w=400)
    box = (0, 40, 340, 240)
    _draw_frame(img, box)
    x0, y0, x1, y1 = box
    yt_true = [y0 + int((y1 - y0) * f) for f in (0.15, 0.5, 0.85)]
    for y in yt_true:
        cv2.line(img, (x0 + 3, y), (x0 + 8, y), (0, 0, 0), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = detect_axis_ticks(gray, box)["y_ticks"]
    assert len(found) == 3, found
    for f, expected in zip(found, yt_true):
        assert abs(f - expected) <= 3
