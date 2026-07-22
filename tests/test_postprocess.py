"""Unit tests for the geometry post-processing (deterministic, no PDF needed)."""
import numpy as np
import pytest

from cvdigitize.postprocess import (
    order_curve, dedupe, split_branches, resample_arclength,
    resample_uniform_x, resample_uniform_potential, keep_main_components,
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


def test_ink_aware_prefers_on_ink_continuation_over_nearer_chord():
    # ink runs along the row y=50; A ends at (50,50). B continues it on ink,
    # C is a stray whose endpoint is nearer in raw distance but off the ink.
    ink = np.zeros((100, 120), bool)
    ink[49:52, 10:96] = True                     # the real curve's ink band
    A = np.array([[10.0, 50], [50.0, 50]])
    B = np.array([[60.0, 50], [95.0, 50]])       # on-ink continuation (dist 10)
    C = np.array([[50.0, 42], [50.0, 20]])       # off-ink stray (endpoint dist 8)

    plain = order_curve([A, B, C])               # pure nearest -> jumps to C first
    inked = order_curve([A, B, C], ink_mask=ink) # ink-aware -> continues onto B
    # after A's tail (50,50): plain teleports up to C(50,42); ink-aware goes to B(60,50)
    def point_after_A(res):
        i = np.where((res[:, 0] == 50) & (res[:, 1] == 50))[0]
        return res[i[-1] + 1] if len(i) and i[-1] + 1 < len(res) else None
    assert point_after_A(plain)[1] < 50          # plain heads up toward C
    assert point_after_A(inked)[0] == 60         # ink-aware heads along the ink to B


def test_ink_aware_dash_gap_is_not_reported_as_chord():
    # a dashed curve: short on-tangent gaps between dashes must NOT be flagged.
    ink = np.zeros((100, 240), bool)
    parts = []
    for x0 in range(10, 220, 40):                # dashes with ~15px gaps
        ink[49:52, x0:x0 + 25] = True
        parts.append(np.array([[float(x0), 50], [float(x0 + 24), 50]]))
    _, gaps = order_curve(parts, ink_mask=ink, return_gaps=True)
    assert gaps == []                            # every gap is dash-sized, none a chord


def test_ink_aware_reports_unavoidable_long_chord_as_gap():
    ink = np.zeros((100, 400), bool)
    ink[49:52, 10:60] = True
    ink[49:52, 330:390] = True                   # a big empty span between two runs
    A = np.array([[10.0, 50], [58.0, 50]])
    B = np.array([[332.0, 50], [388.0, 50]])
    _, gaps = order_curve([A, B], ink_mask=ink, return_gaps=True)
    assert len(gaps) == 1                         # the only join is a long off-ink chord


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


def test_resample_uniform_potential_branches_are_uniform_in_E():
    loop = make_loop(300)
    out = resample_uniform_potential(loop, n_per_branch=200)
    assert len(out) == 400
    fwd, rev = out[:200], out[200:]
    # anodic branch rises in E, cathodic falls; each uniform in E
    assert fwd[0, 0] < fwd[-1, 0]
    assert rev[0, 0] > rev[-1, 0]
    for branch in (fwd, rev):
        dE = np.abs(np.diff(branch[:, 0]))
        assert np.std(dE) / np.mean(dE) < 1e-6      # perfectly even in potential
    # loop stays two-valued: at a mid potential both branches give a current
    mid = 0.5
    j_fwd = np.interp(mid, fwd[:, 0], fwd[:, 1])
    j_rev = np.interp(mid, rev[::-1, 0], rev[::-1, 1])
    assert abs(j_fwd - j_rev) > 0.1                 # hysteresis preserved


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
