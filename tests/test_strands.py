"""Tests for junction-aware strand decomposition.

Synthetic skeletons of overlapping strokes: the decomposition must follow each
physical stroke straight through crossings instead of dying at them, keep a
sharp peak (which is not a junction) intact, and isolate a solid curve from a
dashed sibling that overlaps it.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from cvdigitize.strands import skeleton_to_strands, strands_to_curves, _is_axis_strand


def _draw(*segments, h=90, w=90):
    img = np.zeros((h, w), np.uint8)
    for p0, p1 in segments:
        cv2.line(img, p0, p1, 1, 1)
    return skeletonize(img > 0)


def _arclen(xy):
    return float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum())


def test_x_crossing_gives_two_strands():
    skel = _draw(((5, 5), (85, 85)), ((5, 85), (85, 5)))
    s = skeleton_to_strands(skel)
    assert len(s) == 2
    assert all(_arclen(x) > 90 for x in s)   # each diagonal survives whole


def test_curve_passes_through_axis():
    skel = _draw(((5, 45), (85, 45)), ((45, 5), (45, 85)))  # a '+'
    s = skeleton_to_strands(skel)
    assert len(s) == 2


def test_t_junction_stub_is_separate():
    # horizontal line with a downward stub from the middle
    skel = _draw(((5, 45), (85, 45)), ((45, 45), (45, 85)))
    s = skeleton_to_strands(skel)
    lens = sorted(_arclen(x) for x in s)
    assert len(s) == 2
    assert lens[0] < lens[1]                 # stub shorter than the through-line


def test_sharp_peak_stays_one_strand():
    # a V (sharp turn) is inside one stroke, not a junction -> single strand
    skel = _draw(((5, 80), (45, 5)), ((45, 5), (85, 80)))
    s = skeleton_to_strands(skel)
    assert len(s) == 1


def test_solid_survives_dashed_overlay():
    # a long horizontal solid crossed by several short vertical dashes
    segs = [((5, 60), (115, 60))]
    for x in range(20, 110, 16):
        segs.append(((x, 50), (x, 70)))
    skel = _draw(*segs, h=120, w=120)
    s = skeleton_to_strands(skel)
    longest = max(s, key=_arclen)
    assert np.ptp(longest[:, 0]) > 90        # the solid spans the full width whole


def test_strands_to_curves_separates_solid_and_dashed():
    segs = [((5, 60), (115, 60))]
    for x in range(20, 110, 16):
        segs.append(((x, 50), (x, 70)))
    skel = _draw(*segs, h=120, w=120)
    curves = strands_to_curves(skeleton_to_strands(skel))
    assert len(curves) == 2                   # solid + assembled dashes
    assert np.ptp(curves[0][:, 0]) > 90       # solid first, spans the width


def test_is_axis_strand():
    horiz = np.column_stack([np.arange(80), np.full(80, 20.0)])
    assert _is_axis_strand(horiz)
    diag = np.column_stack([np.arange(80), np.arange(80.0)])
    assert not _is_axis_strand(diag)          # 45deg is not axis-aligned


def test_pure_loop_returns_one_strand():
    img = np.zeros((80, 80), np.uint8)
    cv2.circle(img, (40, 40), 25, 1, 1)
    s = skeleton_to_strands(skeletonize(img > 0))
    assert len(s) == 1
    assert len(s[0]) > 50
