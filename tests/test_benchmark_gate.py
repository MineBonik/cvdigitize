"""Tests for the benchmark's CV-plausibility gate (Phase 1 honesty gate).

Synthetic curves stand in for the real extraction pool: a clean voltammogram
loop must pass; a near-straight line, a zig-zag/text blob and a stack of rules
must be rejected so a failed extraction reports "no acceptable curve" instead
of matching junk at a fair-looking score.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from benchmark import cv_plausible, _arc_efficiency, _self_crossings  # noqa: E402


def _cv_loop(n=400):
    """A duck-shaped closed loop: two single-valued branches over the sweep."""
    t = np.linspace(0, 2 * np.pi, n)
    x = np.cos(t)
    y = np.sin(t) * (0.6 + 0.3 * np.cos(2 * t))  # asymmetric like a CV
    return np.column_stack([x, y])


def test_arc_efficiency_line_vs_loop():
    line = np.column_stack([np.linspace(0, 1, 100), np.linspace(0, 1, 100)])
    assert _arc_efficiency(line) < 1.2
    assert _arc_efficiency(_cv_loop()) > 2.0


def test_self_crossings_zero_for_simple_loop():
    assert _self_crossings(_cv_loop()) <= 2


def test_plausible_accepts_cv_loop():
    assert cv_plausible(_cv_loop()) is True


def test_rejects_near_straight_line():
    # arc-efficiency ~1: the text-scan / axis-fragment failure mode
    x = np.linspace(0, 1, 200)
    line = np.column_stack([x, 0.5 * x + 0.01 * np.sin(3 * x)])
    assert cv_plausible(line) is False


def test_rejects_zigzag_blob():
    # a jagged self-crossing scribble (letters / logo / multi-panel smear)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(120, 2))
    assert cv_plausible(pts) is False


def test_rejects_stack_of_rules():
    # parallel horizontal rules stitched into one path -> high arc-efficiency
    seg = []
    for k in range(8):
        y = k / 8
        seg += [[0, y], [1, y], [1, y + 0.001]]
    assert cv_plausible(np.asarray(seg, float)) is False


def test_rejects_too_short():
    assert cv_plausible(np.zeros((5, 2))) is False
