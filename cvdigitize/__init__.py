"""cvdigitize — digitize Cyclic Voltammetry curves from native-vector PDFs.

Pipeline: ingest/classify -> extract curves from the PDF's own path geometry ->
post-process (loop ordering, dedupe, resample) -> human-confirmed axis
calibration -> package (CSV + frictionless JSON + echemdb YAML).

Vector only, by design: for a vector figure the curve points are exact, so
there is no tracing and no resolution limit. Raster figures need pixel
tracing, a fundamentally fuzzier problem, and that pipeline is not part of
this package — ``cvdigitize info`` reports when a paper's figures are raster.
"""

__version__ = "0.0.1"
