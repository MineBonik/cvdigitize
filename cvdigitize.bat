@echo off
REM Convenience launcher: run `cvdigitize <args>` from this folder instead of
REM typing the full venv path. Example: cvdigitize info "data\in\paper.pdf"
"%~dp0.venv\Scripts\python.exe" -m cvdigitize %*
