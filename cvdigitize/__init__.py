"""cvdigitize — automated digitization of Cyclic Voltammetry curves from PDFs.

Pipeline (see plan): ingest/classify -> extract (vector/raster) -> digitize ->
post-process (loop ordering, resample) -> package (CSV + frictionless JSON + YAML).

Currently implemented: M0 vector curve extraction & color separation.
"""

__version__ = "0.0.1"
