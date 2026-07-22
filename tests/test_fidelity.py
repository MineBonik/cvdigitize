"""Tests for reference-free ink-fidelity scoring (with dash awareness)."""
import numpy as np

from cvdigitize.fidelity import ink_fidelity, grade


def _blank(h=400, w=600):
    return np.zeros((h, w), bool)


def _draw_solid_line(mask, y=200, x0=50, x1=550, half=1):
    for x in range(x0, x1):
        mask[y - half:y + half + 1, x] = True
    return np.column_stack([np.arange(x0, x1), np.full(x1 - x0, y, float)])


def test_perfect_trace_scores_high_no_defects():
    m = _blank()
    poly = _draw_solid_line(m)
    r = ink_fidelity(poly, m)
    assert r["score"] >= 95
    assert r["n_defects"] == 0
    assert grade(r["score"]) == "good"


def test_chord_across_empty_space_is_flagged():
    # trace runs along ink on the left, then chords straight across a blank gap
    m = _blank()
    _draw_solid_line(m, x0=50, x1=250)               # ink only on the left third
    poly = np.column_stack([np.arange(50, 550), np.full(500, 200, float)])  # keeps going
    r = ink_fidelity(poly, m)
    assert r["score"] < 70                            # long off-ink run tanks the score
    assert r["n_defects"] >= 1
    sp = r["off_ink_spans"][0]
    assert sp["start"][0] >= 240 and sp["len"] > 100  # localised to the empty stretch


def test_dashed_curve_is_not_penalised():
    # ink drawn as dashes; the trace is a continuous line bridging the gaps.
    m = _blank()
    y = 200
    for x in range(50, 550):
        if (x // 20) % 2 == 0:            # 20px dash, 20px gap
            m[y - 1:y + 2, x] = True
    poly = np.column_stack([np.arange(50, 550), np.full(500, y, float)])
    r = ink_fidelity(poly, m)
    # roughly half the samples sit over gaps, but every gap is short (dash-sized)
    assert r["frac_off_ink"] > 0.2
    assert r["n_defects"] == 0            # no gap is long enough to be a chord
    assert r["score"] >= 95              # dash bridging must not lower the score


def test_partial_coverage_detected():
    # ink spans the full width but the trace only covers the left half
    m = _blank()
    _draw_solid_line(m, x0=50, x1=550)
    poly = np.column_stack([np.arange(50, 300), np.full(250, 200, float)])
    r = ink_fidelity(poly, m)
    assert 0.35 < r["coverage"] < 0.65    # ~half the ink is covered
