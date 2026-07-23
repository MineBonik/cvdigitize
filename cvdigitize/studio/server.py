"""CV Studio's local companion server — stdlib only (no Flask, no new deps).

``ThreadingHTTPServer`` so a slow extract doesn't freeze the UI; workspace
writes go through ``workspace.py``'s lock. Localhost only, nothing uploaded.
See STUDIO_PLAN.md §3/§6 for the architecture and API contract.
"""
from __future__ import annotations

import json
import mimetypes
import os
import posixpath
import socket
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import pipeline
from . import workspace as ws

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")


class StudioError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _safe_join(root: str, *parts: str) -> str:
    """Join under ``root``, rejecting any ``..`` escape (path-traversal guard)."""
    path = root
    for part in parts:
        for seg in part.split("/"):
            seg = urllib.parse.unquote(seg)
            if not seg or seg in (".", ".."):
                continue
            path = os.path.join(path, seg)
    path = os.path.normpath(path)
    if not (path == os.path.normpath(root) or path.startswith(os.path.normpath(root) + os.sep)):
        raise StudioError("path escapes root", 400)
    return path


class Handler(BaseHTTPRequestHandler):
    server_version = "CVStudio/0.1"
    workspace_dir: str = "data/workspace"

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    # ---- helpers ----------------------------------------------------- #
    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400):
        self._send_json({"error": message}, status)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise StudioError(f"invalid JSON body: {e}", 400)

    def _send_file(self, path: str):
        if not os.path.isfile(path):
            self._send_error_json("not found", 404)
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- routing ------------------------------------------------------ #
    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/" or path == "/studio.html":
                self._send_file(os.path.join(_TOOLS_DIR, "studio.html"))
            elif path.startswith("/workspace/"):
                rel = path[len("/workspace/"):]
                self._send_file(_safe_join(self.workspace_dir, rel))
            elif path == "/api/list_crops":
                paper = (query.get("paper") or [""])[0]
                if not paper:
                    raise StudioError("paper is required", 400)
                self._send_json({"crops": ws.list_crops(self.workspace_dir, paper)})
            elif path == "/api/list_papers":
                self._send_json({"papers": ws.list_papers(self.workspace_dir)})
            else:
                self._send_error_json("not found", 404)
        except StudioError as e:
            self._send_error_json(str(e), e.status)
        except Exception as e:  # pragma: no cover - defensive
            self._send_error_json(f"{type(e).__name__}: {e}", 500)

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        try:
            body = self._read_json_body()
            if path == "/api/open_paper":
                self._handle_open_paper(body)
            elif path == "/api/save_crop":
                self._handle_save_crop(body)
            elif path == "/api/delete_crop":
                self._handle_delete_crop(body)
            elif path == "/api/autocalibrate":
                self._handle_autocalibrate(body)
            elif path == "/api/save_calibration":
                self._handle_save_calibration(body)
            elif path == "/api/measure":
                self._handle_measure(body)
            elif path == "/api/autoextract":
                self._handle_autoextract(body)
            elif path == "/api/trace":
                self._handle_trace(body)
            elif path == "/api/accept_curves":
                self._handle_accept_curves(body)
            else:
                self._send_error_json("not found", 404)
        except StudioError as e:
            self._send_error_json(str(e), e.status)
        except FileNotFoundError as e:
            self._send_error_json(f"file not found: {e}", 404)
        except ValueError as e:
            self._send_error_json(str(e), 400)
        except Exception as e:  # pragma: no cover - defensive
            self._send_error_json(f"{type(e).__name__}: {e}", 500)

    # ---- endpoint handlers --------------------------------------------- #
    def _handle_open_paper(self, body: dict):
        pdf_path = body.get("pdf_path")
        if not pdf_path:
            raise StudioError("pdf_path is required", 400)
        analysis = pipeline.analyze_paper(self.workspace_dir, pdf_path)
        self._send_json({
            "paper": analysis["paper"],
            "pages": analysis["pages"],
        })

    def _handle_save_crop(self, body: dict):
        for key in ("paper", "type", "source", "bbox", "image"):
            if key not in body:
                raise StudioError(f"{key} is required", 400)
        meta = ws.save_crop(
            self.workspace_dir, body["paper"],
            type_=body["type"], source=body["source"], bbox=body["bbox"],
            exclude_rects=body.get("excludeRects") or [],
            image_data_url=body["image"], parent_crop=body.get("parentCrop"),
        )
        self._send_json({"crop": meta["name"],
                         "png": f"/workspace/{body['paper']}/crops/{meta['name']}.png",
                         "meta": meta})

    def _handle_delete_crop(self, body: dict):
        paper, crop = body.get("paper"), body.get("crop")
        if not paper or not crop:
            raise StudioError("paper and crop are required", 400)
        ok = ws.delete_crop(self.workspace_dir, paper, crop)
        self._send_json({"ok": ok})

    def _handle_autocalibrate(self, body: dict):
        paper, crop = body.get("paper"), body.get("crop")
        if not paper or not crop:
            raise StudioError("paper and crop are required", 400)
        out = pipeline.autocalibrate_crop(self.workspace_dir, paper, crop)
        self._send_json(out)

    def _handle_save_calibration(self, body: dict):
        paper, crop = body.get("paper"), body.get("crop")
        calibration = body.get("calibration")
        if not paper or not crop or calibration is None:
            raise StudioError("paper, crop and calibration are required", 400)
        ws.set_crop_calibration(self.workspace_dir, paper, crop, calibration)
        self._send_json({"ok": True})

    def _handle_measure(self, body: dict):
        paper, crop = body.get("paper"), body.get("crop")
        if not paper or not crop:
            raise StudioError("paper and crop are required", 400)
        out = pipeline.measure_crop(self.workspace_dir, paper, crop)
        self._send_json(out)

    def _handle_autoextract(self, body: dict):
        paper, crop = body.get("paper"), body.get("crop")
        if not paper or not crop:
            raise StudioError("paper and crop are required", 400)
        out = pipeline.autoextract_crop(self.workspace_dir, paper, crop)
        self._send_json(out)

    def _handle_trace(self, body: dict):
        paper, crop = body.get("paper"), body.get("crop")
        guides = body.get("guides")
        if not paper or not crop or not guides:
            raise StudioError("paper, crop and guides are required", 400)
        out = pipeline.trace_crop(self.workspace_dir, paper, crop, guides)
        self._send_json(out)

    def _handle_accept_curves(self, body: dict):
        paper, crop = body.get("paper"), body.get("crop")
        curves = body.get("curves")
        if not paper or not crop or curves is None:
            raise StudioError("paper, crop and curves are required", 400)
        pipeline.accept_curves(self.workspace_dir, paper, crop, curves)
        self._send_json({"ok": True})


def make_server(workspace_dir: str, port: int = 0) -> ThreadingHTTPServer:
    """Build (but don't start) the server, bound to ``port`` (0 = OS-assigned)."""
    os.makedirs(workspace_dir, exist_ok=True)
    handler = type("BoundHandler", (Handler,), {"workspace_dir": os.path.abspath(workspace_dir)})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


def run(workspace_dir: str = "data/workspace", port: int = 8799, open_browser: bool = True):
    server = make_server(workspace_dir, port)
    actual_port = server.server_address[1]
    url = f"http://localhost:{actual_port}/"
    print(f"CV Studio serving {url}  (workspace: {os.path.abspath(workspace_dir)})")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
