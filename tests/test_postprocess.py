"""Unit tests for the geometry post-processing (deterministic, no PDF needed)."""
import numpy as np
import pytest

from cvdigitize.postprocess import (
    order_curve, dedupe, split_branches, resample_arclength,
    resample_uniform_x, keep_main_components,
)


def make_loop(n=200):
    """A simple triangular-sweep CV loop: E up then down, j a smooth response.

    Has a current step at the switching potentials (like a real capacitive CV).
    """
    Ef = np.linspace(0, 1, n)
    jf = 0.5 + 0.3 * np.sin(np.pi * Ef)
    Er = np.linspace(1, 0, n)
    jr = -0.5 - 0.3 * np.sin(np.pi * Er)
    return np.column_stack([np.r_[Ef, Er], np.r_[jf, jr]])


def make_closed_loop(n=400):
    """A continuous closed loop (ellipse) with no discontinuities."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 0.5 + 0.5 * np.cos(t)
    y = 0.3 * np.sin(t)
    return np.column_stack([x, y])


def split_into_subpaths(xy, k=7, rng=None):
    """Cut a curve into k contiguous sub-paths and shuffle their order."""
    rng = rng or np.random.default_rng(0)
    idx = np.array_split(np.arange(len(xy)), k)
    parts = [xy[i] for i in idx]
    rng.shuffle(parts)
    # randomly flip some
    parts = [p[::-1] if rng.random() < 0.5 else p for p in parts]
    return parts


def test_order_curve_reconnects_shuffled_segments():
    loop = make_closed_loop()          # continuous => reordering must be gap-free
    parts = split_into_subpaths(loop, k=9)
    ordered = order_curve(parts)
    # same number of points (segments just reconnected)
    assert len(ordered) == len(loop)
    # consecutive gaps should be small (no teleporting across the plot)
    steps = np.hypot(np.diff(ordered[:, 0]), np.diff(ordered[:, 1]))
    diag = np.hypot(*(loop.max(0) - loop.min(0)))
    assert steps.max() < 0.05 * diag


def test_order_curve_single_and_empty():
    assert order_curve([]).shape == (0, 2)
    one = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(order_curve([one]), one)


def test_dedupe_removes_consecutive_duplicates():
    xy = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2]], float)
    out = dedupe(xy)
    assert len(out) == 3


def test_split_branches_direction():
    loop = make_loop()
    fwd, rev = split_branches(loop)
    # forward ends at the high-E vertex; reverse starts there
    assert fwd[0, 0] < fwd[-1, 0]         # E increases on the anodic scan
    assert rev[0, 0] > rev[-1, 0]         # E decreases on the cathodic scan
    assert abs(fwd[-1, 0] - rev[0, 0]) < 1e-9


def test_resample_arclength_even_spacing():
    loop = make_loop()
    out = resample_arclength(loop, n=500)
    assert len(out) == 500
    steps = np.hypot(np.diff(out[:, 0]), np.diff(out[:, 1]))
    # arc-length resampling => nearly constant step (low coefficient of variation)
    assert np.std(steps) / np.mean(steps) < 0.05


def test_resample_uniform_x_grid():
    branch = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 2, 50)])
    out = resample_uniform_x(branch, n=11)
    assert len(out) == 11
    dx = np.diff(out[:, 0])
    assert np.allclose(dx, dx[0])         # uniform in x
    assert out[-1, 1] == pytest.approx(2.0, abs=1e-6)


def test_keep_main_components_drops_stray_swatch():
    loop = make_loop()
    parts = split_into_subpaths(loop, k=6)
    # add a tiny isolated "legend swatch" far away
    swatch = np.array([[5.0, 5.0], [5.1, 5.0]])
    kept = keep_main_components(parts + [swatch])
    pts = np.vstack(kept)
    # the swatch (x≈5) must be gone; real curve (x in [0,1]) retained
    assert pts[:, 0].max() < 2.0
    assert len(pts) >= len(loop) - 5


def test_keep_main_components_keeps_both_branches():
    """Two large disconnected branches (open loop) must both survive."""
    b1 = np.column_stack([np.linspace(0, 1, 80), np.full(80, 0.5)])
    b2 = np.column_stack([np.linspace(0, 1, 80), np.full(80, -0.5)])  # gap 1.0 in y
    kept = keep_main_components([b1, b2])
    assert sum(len(p) for p in kept) == 160
