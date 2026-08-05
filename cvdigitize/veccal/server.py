"""Local server behind ``cvdigitize vector-calibrate``.

Serves one page (``tools/veccal.html``) plus a small JSON API that walks the
panels found by :mod:`cvdigitize.veccal.scan`. Each panel is shown as its cropped figure
with the extracted curves drawn over it and the calibration pre-filled; the user
confirms or fixes it, and saving writes that panel's datapackages immediately —
so closing the browser never loses confirmed work.

Binds to 127.0.0.1 only: this is a single-user local tool, and the API happily
writes files, so it must not be reachable from the network.
"""
from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .finalize import save_panel
from .scan import (index_path, load_index, migrate_index, needs_migration,
                   save_index, scan_folder, unit_with_geometry)

# The index is read on every request and written on every save. Re-reading it
# from disk each time was the tool's main slowness, so it is held in memory
# behind a lock and only the (small) file write hits the disk.
_LOCK = threading.Lock()
_CACHE: dict[str, dict] = {}


def _index(work_dir: str) -> dict:
    with _LOCK:
        if work_dir not in _CACHE:
            _CACHE[work_dir] = load_index(work_dir)
        return _CACHE[work_dir]


def _persist(work_dir: str, index: dict) -> None:
    with _LOCK:
        _CACHE[work_dir] = index
        save_index(work_dir, index)


def reset_cache() -> None:
    """Drop the in-memory index (tests run several servers in one process)."""
    with _LOCK:
        _CACHE.clear()


_UI = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tools", "veccal.html")

_MIME = {".html": "text/html; charset=utf-8", ".png": "image/png",
         ".json": "application/json", ".csv": "text/csv"}


class VecCalError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _safe_join(root: str, rel: str) -> str:
    """Join ``rel`` under ``root``, refusing anything that escapes it."""
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, rel.lstrip("/\\")))
    if target != root_abs and not target.startswith(root_abs + os.sep):
        raise VecCalError("path outside the work directory", 403)
    return target


class Handler(BaseHTTPRequestHandler):
    work_dir = "data/out/vector_curated"
    out_dir = "data/out/vector_curated/curves"
    resample = 1000
    resample_mode = "arclength"

    server_version = "cvdigitize-veccal"

    def log_message(self, fmt, *args):        # keep the console for progress
        pass

    # -- plumbing ----------------------------------------------------------
    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str):
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, 404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type",
                         _MIME.get(os.path.splitext(path)[1].lower(),
                                   "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            raise VecCalError(f"bad JSON body: {exc}") from exc

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        try:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_file(_UI)
            elif path == "/api/index":
                self._send_json(self._index_payload())
            elif path == "/api/panel":
                self._send_json(self._handle_panel(self.path))
            elif path.startswith("/panels/"):
                self._send_file(_safe_join(self.work_dir, path[1:]))
            else:
                self._send_json({"error": "not found"}, 404)
        except VecCalError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except Exception as exc:                       # never 500 silently
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self):
        try:
            path = self.path.split("?", 1)[0]
            body = self._body()
            if path == "/api/save":
                self._send_json(self._handle_save(body))
            elif path == "/api/skip":
                self._send_json(self._handle_mark(body, "skipped"))
            elif path == "/api/reopen":
                self._send_json(self._handle_mark(body, "pending"))
            elif path == "/api/edit":
                self._send_json(self._handle_edit(body))
            else:
                self._send_json({"error": "not found"}, 404)
        except VecCalError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except ValueError as exc:                      # calibration rejected
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # -- handlers ----------------------------------------------------------
    def _counts(self, units: list) -> dict:
        return {
            "total": len(units),
            "saved": sum(1 for u in units if u.get("status") == "saved"),
            "skipped": sum(1 for u in units if u.get("status") == "skipped"),
            "pending": sum(1 for u in units if u.get("status", "pending") == "pending"),
            "prefilled": sum(1 for u in units if u.get("calib_source") != "none"),
        }

    def _index_payload(self) -> dict:
        """The panel list, without point arrays — see ``scan._split_geometry``."""
        index = _index(self.work_dir)
        units = index.get("units", [])
        return {
            "source_folder": index.get("source_folder", ""),
            "out_dir": self.out_dir,
            "counts": self._counts(units),
            "units": units,
        }

    def _handle_panel(self, raw_path: str) -> dict:
        """One panel, with its geometry — fetched only for the panel on screen."""
        query = parse_qs(urlparse(raw_path).query)
        uid = (query.get("uid") or [""])[0]
        unit = self._find(_index(self.work_dir), uid)
        return {"unit": unit_with_geometry(self.work_dir, unit)}

    def _find(self, index: dict, uid: str) -> dict:
        for u in index.get("units", []):
            if u.get("uid") == uid:
                return u
        raise VecCalError(f"unknown panel {uid!r}", 404)

    def _handle_save(self, body: dict) -> dict:
        uid = body.get("uid") or ""
        index = _index(self.work_dir)
        unit = self._find(index, uid)

        # Curve names/samples the user edited in the form win over the guesses,
        # and any curve they unticked is left out of this panel entirely.
        for edited in body.get("curves") or []:
            for curve in unit.get("curves", []):
                if curve.get("color") == edited.get("color"):
                    if edited.get("name"):
                        curve["name"] = edited["name"].strip()
                    curve["sample"] = (edited.get("sample") or "").strip()
                    if "include" in edited:
                        curve["include"] = bool(edited["include"])

        payload = unit_with_geometry(self.work_dir, unit)
        payload["curves"] = [c for c in payload["curves"] if c.get("include") is not False]
        if not payload["curves"]:
            raise VecCalError("Every curve in this panel is unticked — "
                              "tick at least one, or skip the panel.")

        result = save_panel(
            payload,
            body.get("calibration") or {},
            os.path.join(self.out_dir, uid),
            resample=self.resample, resample_mode=self.resample_mode,
            scan_rate=(body.get("scan_rate") or "").strip(),
        )
        unit["status"] = "saved"
        unit["saved_calibration"] = body.get("calibration")
        unit["saved_curves"] = result["curves"]
        _persist(self.work_dir, index)
        return {"ok": True, **result, "counts": self._counts(index["units"])}

    def _handle_mark(self, body: dict, status: str) -> dict:
        index = _index(self.work_dir)
        unit = self._find(index, body.get("uid") or "")
        unit["status"] = status
        if status == "skipped":
            unit["skip_reason"] = (body.get("reason") or "").strip()
        _persist(self.work_dir, index)
        return {"ok": True, "status": status,
                "counts": self._counts(index["units"])}

    def _handle_edit(self, body: dict) -> dict:
        """Persist name/sample edits without saving files (so they survive a reload)."""
        index = _index(self.work_dir)
        unit = self._find(index, body.get("uid") or "")
        for edited in body.get("curves") or []:
            for curve in unit.get("curves", []):
                if curve.get("color") == edited.get("color"):
                    if edited.get("name"):
                        curve["name"] = edited["name"].strip()
                    curve["sample"] = (edited.get("sample") or "").strip()
        _persist(self.work_dir, index)
        return {"ok": True}


def make_server(work_dir: str, out_dir: str, *, port: int = 0,
                resample: int = 1000, resample_mode: str = "arclength"
                ) -> ThreadingHTTPServer:
    reset_cache()
    handler = type("BoundHandler", (Handler,), {
        "work_dir": work_dir, "out_dir": out_dir,
        "resample": resample, "resample_mode": resample_mode,
    })
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def run(source_folder: str, work_dir: str, out_dir: str, *, port: int = 8756,
        rescan: bool = False, cv_threshold: float = 0.08,
        open_browser: bool = True, resample: int = 1000,
        resample_mode: str = "arclength", workers: int | None = None) -> int:
    """Scan if needed, then serve the calibration UI until interrupted."""
    if rescan or not os.path.exists(index_path(work_dir)):
        if not source_folder:
            print("No scan found in the work directory and no --in folder given.")
            return 2
        if not os.path.isdir(source_folder):
            print(f"Not a folder: {source_folder}")
            return 2
        print(f"Scanning {source_folder} for vector CV panels ...")

        def progress(done, total, label):
            print(f"  [{done:>3}/{total}] {label}")

        index = scan_folder(source_folder, work_dir, cv_threshold=cv_threshold,
                            progress=progress, workers=workers)
        units = index["units"]
        pre = sum(1 for u in units if u.get("calib_source") != "none")
        papers = len({u["stem"] for u in units})
        curves = sum(len(u["curves"]) for u in units)
        print(f"\nFound {len(units)} CV panel(s) ({curves} curves) in {papers} paper(s).")
        print(f"{pre} panel(s) came with a pre-filled calibration to confirm.")
        if not units:
            print("Nothing to calibrate.")
            return 1
    else:
        index = load_index(work_dir)
        if needs_migration(index):
            # An index written before geometry moved into its own files; split it
            # once so this session gets the fast path without a full re-scan.
            print("Splitting curve geometry out of the index (one-off) ...")
            index = migrate_index(work_dir)
            print(f"  index.json is now "
                  f"{os.path.getsize(index_path(work_dir)) / 1e6:.1f} MB")
        done = sum(1 for u in index["units"] if u.get("status") == "saved")
        print(f"Resuming: {done}/{len(index['units'])} panel(s) already saved "
              f"(--rescan to re-detect).")

    httpd = make_server(work_dir, out_dir, port=port,
                        resample=resample, resample_mode=resample_mode)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"\nCV vector calibration -> {url}")
    print(f"Curves are written to {out_dir} as you save. Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
    return 0
