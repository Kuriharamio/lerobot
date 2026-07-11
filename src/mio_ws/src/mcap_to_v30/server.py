#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mio_ws.src.mcap_to_v30.converter import (
    convert_mcap_source,
    discover_mcap_files,
    scan_mcap_source,
)

STATIC_ROOT = Path(__file__).parent / "static"


class ApplicationState:
    def __init__(self, source: Path) -> None:
        self.source = source.expanduser().resolve()
        self.scan_result: dict[str, Any] | None = None
        self.scan_lock = threading.Lock()
        self.scan_job: dict[str, Any] = {
            "state": "idle",
            "progress": 0,
            "message": "等待解析",
            "source": str(self.source),
        }
        self.job_lock = threading.Lock()
        self.job: dict[str, Any] = {"state": "idle", "progress": 0, "message": "等待导出"}

    def scan_status(self) -> dict[str, Any]:
        should_start = False
        with self.scan_lock:
            if self.scan_job["state"] == "idle":
                self.scan_job = {
                    **self.scan_job,
                    "state": "queued",
                    "progress": 0,
                    "message": "正在发现 MCAP 文件",
                }
                should_start = True
            status = dict(self.scan_job)
            if self.scan_result is not None:
                status["result"] = self.scan_result
        if should_start:
            threading.Thread(target=self._run_scan, name="mcap-scan", daemon=True).start()
        return status

    def _run_scan(self) -> None:
        self._update_scan({"state": "running", "message": "正在解析 MCAP 数据"})
        try:
            result = scan_mcap_source(self.source, self._scan_progress)
            with self.scan_lock:
                self.scan_result = result
                self.scan_job = {
                    **self.scan_job,
                    "state": "completed",
                    "progress": 100,
                    "message": "MCAP 解析完成",
                }
        except Exception as error:
            logging.exception("MCAP scan failed")
            self._update_scan(
                {
                    "state": "failed",
                    "message": str(error),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )

    def _scan_progress(self, update: dict[str, Any]) -> None:
        self._update_scan({"state": "running", **update})

    def _update_scan(self, update: dict[str, Any]) -> None:
        with self.scan_lock:
            self.scan_job = {**self.scan_job, **update}

    def require_scan_result(self) -> dict[str, Any]:
        with self.scan_lock:
            if self.scan_result is None:
                raise RuntimeError("MCAP 数据尚未解析完成。")
            return self.scan_result

    def start_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        mappings = payload.get("mappings")
        parameters = payload.get("parameters")
        if not isinstance(mappings, list) or not isinstance(parameters, dict):
            raise ValueError("Invalid export request.")

        known_topics = {topic["name"] for topic in self.require_scan_result()["topics"]}
        requested_topics = {
            str(topic)
            for mapping in mappings
            if isinstance(mapping, dict)
            for topic in mapping.get("topics", [])
        }
        unknown_topics = sorted(requested_topics - known_topics)
        if unknown_topics:
            raise ValueError(f"Unknown source topics: {', '.join(unknown_topics)}")

        with self.job_lock:
            if self.job["state"] in {"queued", "running"}:
                raise RuntimeError("已有导出任务正在运行。")
            self.job = {"state": "queued", "progress": 0, "message": "正在准备导出"}

        thread = threading.Thread(
            target=self._run_export,
            args=(mappings, parameters),
            name="mcap-export",
            daemon=True,
        )
        thread.start()
        return self.job_status()

    def _run_export(self, mappings: list[dict[str, Any]], parameters: dict[str, Any]) -> None:
        self._update_job({"state": "running", "message": "开始导出"})
        try:
            result = convert_mcap_source(self.source, mappings, parameters, self._progress)
            self._update_job(
                {"state": "completed", "progress": 100, "message": "导出完成", "result": result}
            )
        except Exception as error:
            logging.exception("MCAP conversion failed")
            self._update_job(
                {
                    "state": "failed",
                    "message": str(error),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )

    def _progress(self, update: dict[str, Any]) -> None:
        self._update_job({"state": "running", **update})

    def _update_job(self, update: dict[str, Any]) -> None:
        with self.job_lock:
            self.job = {**self.job, **update}

    def job_status(self) -> dict[str, Any]:
        with self.job_lock:
            return dict(self.job)


class McapUiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], state: ApplicationState) -> None:
        self.state = state
        super().__init__(server_address, McapRequestHandler)


class McapRequestHandler(BaseHTTPRequestHandler):
    server: McapUiServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/scan":
            self._handle_json(self.server.state.scan_status)
            return
        if path == "/api/export/status":
            self._send_json(self.server.state.job_status())
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/export":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._handle_json(lambda: self.server.state.start_export(self._read_json_body()))

    def log_message(self, format_string: str, *args: Any) -> None:
        logging.debug("%s - %s", self.address_string(), format_string % args)

    def _handle_json(self, callback: Any) -> None:
        try:
            self._send_json(callback())
        except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            logging.exception("Request failed")
            self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 2_000_000:
            raise ValueError("Invalid request body size.")
        body = self.rfile.read(content_length)
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object.")
        return parsed

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        if relative not in {"index.html", "app.js", "styles.css"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = STATIC_ROOT / relative
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a local web UI that converts MCAP recordings to LeRobotDataset v3.0."
    )
    parser.add_argument(
        "--mcap",
        type=Path,
        required=True,
        help="MCAP file or directory. Directories are searched recursively for .mcap files.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Local interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Preferred local port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _create_server(host: str, preferred_port: int, state: ApplicationState) -> McapUiServer:
    if not 0 <= preferred_port <= 65535:
        raise ValueError("--port must be between 0 and 65535.")
    if preferred_port == 0:
        return McapUiServer((host, 0), state)
    last_error: OSError | None = None
    for port in range(preferred_port, min(preferred_port + 20, 65536)):
        try:
            return McapUiServer((host, port), state)
        except OSError as error:
            last_error = error
    raise OSError(f"Could not bind to ports {preferred_port}-{preferred_port + 19}") from last_error


def main() -> None:
    args = _parse_args()
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid --log-level: {args.log_level}")
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    source = args.mcap.expanduser().resolve()
    files = discover_mcap_files(source)
    state = ApplicationState(source)
    server = _create_server(args.host, args.port, state)
    port = server.server_address[1]
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{browser_host}:{port}"
    logging.info("Found %d MCAP file(s) under %s", len(files), source)
    logging.info("MCAP to LeRobot v3.0 UI: %s", url)

    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping MCAP converter UI")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
