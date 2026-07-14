# Convenience launcher: run `.\cvdigitize.ps1 <args>` from this folder instead
# of typing the full venv path. Example: .\cvdigitize.ps1 info "data\in\paper.pdf"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m cvdigitize @args
