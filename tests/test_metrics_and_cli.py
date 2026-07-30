"""Tests for loop_metrics (CV-likeness), the page classifier, and the CLI."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvdigitize.postprocess import loop_metrics


def test_loop_metrics_closed_loop_scores_high():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    ellipse = np.column_stack([np.cos(t), 0.6 * np.sin(t)])
    m = loop_metrics(ellipse)
    assert m["closure"] > 0.95          # start≈end
    assert m["area_frac"] > 0.6         # fills most of its bbox
    assert m["loopiness"] > 0.5


def test_loop_metrics_straight_line_scores_low():
    line = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)])
    m = loop_metrics(line)
    assert m["area_frac"] < 0.05
    assert m["loopiness"] < 0.05


def test_loop_metrics_open_arc_low_closure():
    t = np.linspace(0, np.pi, 100)       # half circle: endpoints far apart
    arc = np.column_stack([np.cos(t), np.sin(t)])
    m = loop_metrics(arc)
    assert m["closure"] < 0.6
    assert m["loopiness"] < 0.4


def test_loop_metrics_degenerate():
    assert loop_metrics(np.zeros((2, 2)))["loopiness"] == 0.0
    assert loop_metrics(np.empty((0, 2)))["loopiness"] == 0.0


def _make_pdf(path, n_images=0):
    fig, ax = plt.subplots(figsize=(4, 3))
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(0.5 + 0.4 * np.cos(t), 0.4 * np.sin(t), color="red")
    ax.set_xlabel("E / V"); ax.set_ylabel("j")
    fig.savefig(path)
    plt.close(fig)


def test_classifier_has_largest_image_frac(tmp_path):
    from cvdigitize.ingest import classify_pdf
    pdf = str(tmp_path / "p.pdf")
    _make_pdf(pdf)
    infos = classify_pdf(pdf)
    assert len(infos) == 1
    assert hasattr(infos[0], "largest_image_frac")
    assert 0.0 <= infos[0].largest_image_frac <= 1.0


def test_info_reports_the_figure_page(tmp_path, capsys):
    """`info` must agree with what a scan would find — it is the pre-flight check."""
    import argparse
    from cvdigitize.cli import cmd_info
    pdf = str(tmp_path / "p.pdf")
    _make_pdf(pdf)
    rc = cmd_info(argparse.Namespace(pdf=pdf, quick=False, cv_threshold=0.08))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Candidate figure pages: [0]" in out
    assert "curve(s), loop score" in out
    assert "vector-calibrate" in out          # points at the next step


def test_info_on_a_missing_file_is_an_error_not_a_crash(tmp_path):
    import argparse
    from cvdigitize.cli import cmd_info
    rc = cmd_info(argparse.Namespace(pdf=str(tmp_path / "nope.pdf"),
                                     quick=False, cv_threshold=0.08))
    assert rc == 2


def test_info_quick_skips_panel_detection(tmp_path, capsys):
    import argparse
    from cvdigitize.cli import cmd_info
    pdf = str(tmp_path / "p.pdf")
    _make_pdf(pdf)
    cmd_info(argparse.Namespace(pdf=pdf, quick=True, cv_threshold=0.08))
    out = capsys.readouterr().out
    assert "--quick" in out
    assert "loop score" not in out


def test_bare_path_dispatch(tmp_path):
    """A folder means 'scan it'; a file means 'inspect it'."""
    from cvdigitize.cli import _with_implicit_command
    d = tmp_path / "papers"
    d.mkdir()
    assert _with_implicit_command([str(d)]) == ["vector-calibrate", "--in", str(d)]
    assert _with_implicit_command(["paper.pdf"]) == ["info", "paper.pdf"]
    # explicit commands and flags pass through untouched
    assert _with_implicit_command(["info", "x.pdf"]) == ["info", "x.pdf"]
    assert _with_implicit_command(["--help"]) == ["--help"]
    assert _with_implicit_command([]) == []
