"""Axis calibration: map PDF/pixel coordinates to physical (E, j) values (M1).

Standard scientific plots are axis-aligned (no rotation), so calibration is two
independent linear maps, exactly like svgdigitizer's reference-point model:

    E = ex1 + (px - x1) * (ex2 - ex1) / (x2 - x1)
    j = jy1 + (py - y1) * (jy2 - jy1) / (y2 - y1)

The y map naturally encodes the pixel-y flip (pixel y grows downward while
current grows upward) as long as the two y anchors are given correctly.

Anchors come from one of:
  * a human reading four axis reference points (the robust default), or
  * ``fit_from_reference`` — least-squares fit to a known reference curve,
    used to calibrate the rizo demo without hand-clicking.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Calibration:
    """Two-point-per-axis linear calibration (axis-aligned plots)."""

    x1: float; ex1: float   # pixel x -> potential value at two x anchors
    x2: float; ex2: float
    y1: float; jy1: float   # pixel y -> current value at two y anchors
    y2: float; jy2: float
    x_unit: str = "V"
    y_unit: str = "uA/cm2"
    x_label: str = "E"
    y_label: str = "j"

    def to_data(self, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        E = self.ex1 + (px - self.x1) * (self.ex2 - self.ex1) / (self.x2 - self.x1)
        j = self.jy1 + (py - self.y1) * (self.jy2 - self.jy1) / (self.y2 - self.y1)
        return E, j

    def apply(self, xy: np.ndarray) -> np.ndarray:
        E, j = self.to_data(xy[:, 0], xy[:, 1])
        return np.column_stack([E, j])

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


def _fit_linear(px: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) for target ~ slope*px + intercept."""
    A = np.column_stack([px, np.ones_like(px)])
    slope, intercept = np.linalg.lstsq(A, target, rcond=None)[0]
    return float(slope), float(intercept)


def fit_from_reference(
    extracted_xy: np.ndarray,
    reference_Ej: np.ndarray,
    *,
    x_unit: str = "V",
    y_unit: str = "uA/cm2",
) -> Calibration:
    """Derive a Calibration by aligning an extracted (pixel) curve to a known
    reference curve in data units.

    Both curves describe the same CV. We match them by normalised arc-length
    (each is monotonic in cumulative length around the loop), then linearly fit
    pixel-x -> E and pixel-y -> j. Used to calibrate the rizo demo from the
    hand-digitized reference; not needed once real axis anchors are supplied.
    """
    from .postprocess import order_curve, resample_arclength

    ex = resample_arclength(order_curve([extracted_xy]) if extracted_xy.ndim == 2
                            else extracted_xy, n=len(reference_Ej))
    # Reference is already ordered; resample to same count along arc length.
    ref = resample_arclength(reference_Ej, n=len(reference_Ej))

    # Align orientation: the reference E should increase with pixel x. Compare
    # endpoints and flip the reference ordering if anti-correlated.
    if np.corrcoef(ex[:, 0], ref[:, 0])[0, 1] < 0:
        ref = ref[::-1]

    sx, ix = _fit_linear(ex[:, 0], ref[:, 0])
    sy, iy = _fit_linear(ex[:, 1], ref[:, 1])
    # Express as two anchors per axis at the pixel extremes.
    x1, x2 = float(ex[:, 0].min()), float(ex[:, 0].max())
    y1, y2 = float(ex[:, 1].min()), float(ex[:, 1].max())
    return Calibration(
        x1=x1, ex1=sx * x1 + ix, x2=x2, ex2=sx * x2 + ix,
        y1=y1, jy1=sy * y1 + iy, y2=y2, jy2=sy * y2 + iy,
        x_unit=x_unit, y_unit=y_unit,
    )


def calibration_from_bboxes(
    pixel_bbox: tuple[float, float, float, float],
    data_bbox: tuple[float, float, float, float],
    *,
    x_unit: str = "V",
    y_unit: str = "uA/cm2",
) -> Calibration:
    """Calibration mapping a pixel bounding box to a data bounding box.

    ``pixel_bbox`` = (px_min, py_min, px_max, py_max) of the extracted curve(s);
    ``data_bbox``  = (E_min, j_min, E_max, j_max) of the true data extent.
    Encodes the pixel-y flip: the top of the plot (py_min) is the largest
    current (j_max). Robust for a whole shared-axes panel when the pooled
    extent of several curves spans the axes.
    """
    px0, py0, px1, py1 = pixel_bbox
    e0, j0, e1, j1 = data_bbox
    return Calibration(
        x1=px0, ex1=e0, x2=px1, ex2=e1,
        y1=py0, jy1=j1, y2=py1, jy2=j0,   # py_min -> j_max, py_max -> j_min
        x_unit=x_unit, y_unit=y_unit,
    )


def calibration_from_anchors(
    x_anchor1: tuple[float, float], x_anchor2: tuple[float, float],
    y_anchor1: tuple[float, float], y_anchor2: tuple[float, float],
    **units,
) -> Calibration:
    """Build a Calibration from ``(pixel, value)`` anchors, one pair per axis."""
    return Calibration(
        x1=x_anchor1[0], ex1=x_anchor1[1], x2=x_anchor2[0], ex2=x_anchor2[1],
        y1=y_anchor1[0], jy1=y_anchor1[1], y2=y_anchor2[0], jy2=y_anchor2[1],
        **units,
    )
