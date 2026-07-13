#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import shutil
import tempfile
import threading
import traceback
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from mio_ws.src.mcap_utils.common import (
    discover_mcap_files,
    encode_jpeg,
    iter_camera_frames,
    scan_mcap_item,
)
from mio_ws.src.mcap_utils.splitter import (
    ExistingOutputsError,
    split_mcap_file,
    suggested_output_paths,
    suggested_output_pattern,
    validate_split_request,
)

STATIC_ROOT = Path(__file__).parent / "static"
PRELOAD_AHEAD = 3


@dataclass
class CachedItem:
    result: dict[str, Any]
    thumbnails: dict[str, bytes]
    preview_topic: str | None = None
    preview_frames: list[dict[str, Any]] | None = None
    preview_dir: Path | None = None
    preview_result: dict[str, Any] | None = None
    preview_attempted_topic: str | None = None


class ApplicationState:
    def __init__(self, source: Path) -> None:
        self.source = source.expanduser().resolve()
        self.files = discover_mcap_files(self.source)
        self.lock = threading.RLock()
        self.current_index = 0
        self.expected_segments: int | None = None
        self.preferred_camera_topic: str | None = None
        self.preload_cache: dict[int, CachedItem] = {}
        self.preload_failures: set[int] = set()
        self.preload_generation = 0
        self.preload_thread: threading.Thread | None = None
        self.preload_status: dict[str, Any] = {
            "state": "idle",
            "message": "等待选择相机",
            "cached_items": 0,
            "target_items": 0,
        }
        self.history: list[dict[str, Any]] = []
        self.item_job: dict[str, Any] = {"state": "idle", "progress": 0, "message": "等待解析"}
        self.item_result: dict[str, Any] | None = None
        self.thumbnails: dict[str, bytes] = {}
        self.preview_job: dict[str, Any] = {"state": "idle", "progress": 0, "message": "请选择相机"}
        self.preview_topic: str | None = None
        self.preview_frames: list[dict[str, Any]] = []
        self.temp_root = Path(tempfile.mkdtemp(prefix="lerobot-mcap-split-"))
        self.preview_dir: Path | None = None
        self.split_job: dict[str, Any] = {"state": "idle", "progress": 0, "message": "等待分割"}

    def close(self) -> None:
        with self.lock:
            self.preload_generation += 1
            preload_thread = self.preload_thread
            self.preload_thread = None
        if preload_thread is not None and preload_thread is not threading.current_thread():
            preload_thread.join(timeout=2)
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def session_status(self) -> dict[str, Any]:
        should_start = False
        with self.lock:
            completed = self.current_index >= len(self.files)
            if not completed and self.item_job["state"] == "idle":
                self.item_job = {"state": "queued", "progress": 0, "message": "正在读取当前 MCAP"}
                should_start = True
            payload = {
                "source": str(self.source),
                "total_items": len(self.files),
                "current_index": self.current_index,
                "current_number": min(self.current_index + 1, len(self.files)),
                "completed": completed,
                "expected_segments": self.expected_segments,
                "preferred_camera_topic": self.preferred_camera_topic,
                "preload": dict(self.preload_status),
                "history": list(self.history),
                "item": dict(self.item_job),
            }
            if self.item_result is not None:
                payload["item"]["result"] = self.item_result
        if should_start:
            threading.Thread(target=self._scan_current_item, name="mcap-split-scan", daemon=True).start()
        return payload

    def _scan_current_item(self) -> None:
        with self.lock:
            if self.current_index >= len(self.files):
                return
            index = self.current_index
            path = self.files[index]
            self.item_job = {"state": "running", "progress": 5, "message": f"正在解析 {path.name}"}

        def update(progress: float, message: str) -> None:
            with self.lock:
                if index == self.current_index:
                    self.item_job = {"state": "running", "progress": progress, "message": message}

        try:
            cached_item = self._build_item_cache(path, update)
            with self.lock:
                if index != self.current_index:
                    return
                self._adopt_item_cache_locked(cached_item)
                self.item_job = {"state": "completed", "progress": 100, "message": "当前 MCAP 解析完成"}
        except Exception as error:
            logging.exception("Could not scan MCAP item")
            with self.lock:
                if index == self.current_index:
                    self.item_job = {
                        "state": "failed",
                        "progress": 100,
                        "message": str(error),
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }

    def _build_item_cache(
        self,
        path: Path,
        progress: Callable[[float, str], None] | None = None,
    ) -> CachedItem:
        if progress:
            progress(8, f"正在解析 {path.name}")
        result = scan_mcap_item(path)
        candidates = result["cameras"]
        cameras = []
        thumbnails: dict[str, bytes] = {}
        warnings = []
        for camera_index, camera in enumerate(candidates):
            if progress:
                progress(
                    round(55 + camera_index / max(1, len(candidates)) * 40, 1),
                    f"正在生成相机预览 {camera_index + 1}/{len(candidates)}",
                )
            generator = iter_camera_frames(path, camera["topic"])
            try:
                frame = next(generator)
            except Exception as error:
                warnings.append(f"{camera['topic']}: {error}")
                continue
            finally:
                generator.close()
            height, width = frame.image.shape[:2]
            cameras.append({**camera, "shape": [height, width, 3]})
            thumbnails[camera["topic"]] = encode_jpeg(frame.image, quality=78)

        result["cameras"] = cameras
        result["warnings"] = warnings
        result["relative_path"] = str(path.relative_to(self.source)) if self.source.is_dir() else path.name
        result["suggested_output_pattern"] = str(suggested_output_pattern(self.source, path))
        self._refresh_suggested_outputs(result, path)
        for key in ("start_ns", "end_ns"):
            if result[key] is not None:
                result[key] = str(result[key])
        for camera in result["cameras"]:
            camera["first_ns"] = str(camera["first_ns"])
            camera["last_ns"] = str(camera["last_ns"])
        return CachedItem(result=result, thumbnails=thumbnails)

    def _refresh_suggested_outputs(self, result: dict[str, Any], path: Path) -> None:
        suggested_segments = self.expected_segments or 2
        result["suggested_outputs"] = [
            str(item) for item in suggested_output_paths(self.source, path, suggested_segments)
        ]

    def _adopt_item_cache_locked(self, cached_item: CachedItem) -> None:
        path = self.files[self.current_index]
        self._refresh_suggested_outputs(cached_item.result, path)
        self.thumbnails = cached_item.thumbnails
        self.item_result = cached_item.result
        if (
            cached_item.preview_topic == self.preferred_camera_topic
            and cached_item.preview_frames is not None
            and cached_item.preview_dir is not None
            and cached_item.preview_result is not None
        ):
            self.preview_topic = cached_item.preview_topic
            self.preview_frames = cached_item.preview_frames
            self.preview_dir = cached_item.preview_dir
            self.preview_job = {
                "state": "completed",
                "progress": 100,
                "message": "已加载后台预览缓存",
                "topic": cached_item.preview_topic,
                "result": cached_item.preview_result,
            }

    def thumbnail(self, topic: str) -> bytes:
        with self.lock:
            data = self.thumbnails.get(topic)
        if data is None:
            raise ValueError(f"No thumbnail is available for camera topic {topic!r}.")
        return data

    def start_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        topic = str(payload.get("topic", "")).strip()
        item_index = payload.get("item_index")
        with self.lock:
            self._require_current_item(item_index)
            if self.item_result is None:
                raise RuntimeError("Current MCAP item has not been scanned yet.")
            known_topics = {camera["topic"] for camera in self.item_result["cameras"]}
            if topic not in known_topics:
                raise ValueError(f"Unknown or unsupported camera topic: {topic}")
            if topic == self.preview_topic and self.preview_job["state"] in {
                "queued",
                "running",
                "completed",
            }:
                self._set_preferred_camera_locked(topic)
                if self.preview_job["state"] == "completed":
                    self._schedule_preload_locked()
                return self.preview_status()
            if self.preview_job["state"] in {"queued", "running"}:
                raise RuntimeError("A camera preview is already being prepared.")
            self._set_preferred_camera_locked(topic)
            self._clear_preview_locked()
            self.preview_topic = topic
            self.preview_job = {
                "state": "queued",
                "progress": 0,
                "message": "正在准备相机预览",
                "topic": topic,
            }
            index = self.current_index
            path = self.files[index]
        threading.Thread(
            target=self._prepare_preview,
            args=(index, path, topic),
            name="mcap-camera-preview",
            daemon=True,
        ).start()
        return self.preview_status()

    def _prepare_preview(self, item_index: int, path: Path, topic: str) -> None:
        def update(progress: dict[str, Any]) -> None:
            with self.lock:
                if item_index == self.current_index and topic == self.preview_topic:
                    self.preview_job = {**self.preview_job, "state": "running", **progress}

        try:
            update({"progress": 0, "message": "正在解码相机帧"})
            preview_dir, frames, result = self._build_preview_cache(
                path,
                topic,
                self.temp_root / f"current-{item_index:06d}",
                update,
            )
            with self.lock:
                if item_index != self.current_index or topic != self.preview_topic:
                    shutil.rmtree(preview_dir, ignore_errors=True)
                    return
                self.preview_dir = preview_dir
                self.preview_frames = frames
                self.preview_job = {
                    "state": "completed",
                    "progress": 100,
                    "message": "相机预览准备完成",
                    "topic": topic,
                    "result": result,
                }
                self._schedule_preload_locked()
        except Exception as error:
            logging.exception("Could not prepare camera preview")
            with self.lock:
                if item_index == self.current_index and topic == self.preview_topic:
                    self.preview_job = {
                        "state": "failed",
                        "progress": 100,
                        "message": str(error),
                        "error": str(error),
                    }

    def _build_preview_cache(
        self,
        path: Path,
        topic: str,
        preview_dir: Path,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
        shutil.rmtree(preview_dir, ignore_errors=True)
        preview_dir.mkdir(parents=True)
        frames: list[dict[str, Any]] = []
        try:
            for frame_index, frame in enumerate(iter_camera_frames(path, topic, progress)):
                frame_path = preview_dir / f"{frame_index:08d}.jpg"
                frame_path.write_bytes(encode_jpeg(frame.image))
                frames.append({"index": frame_index, "timestamp_ns": frame.timestamp_ns})
            if len(frames) < 2:
                raise ValueError("Camera preview must contain at least two decoded frames.")
            duration_s = (frames[-1]["timestamp_ns"] - frames[0]["timestamp_ns"]) * 1e-9
            fps = (len(frames) - 1) / duration_s if duration_s > 0 else 0.0
            result = {
                "topic": topic,
                "frame_count": len(frames),
                "first_timestamp_ns": str(frames[0]["timestamp_ns"]),
                "last_timestamp_ns": str(frames[-1]["timestamp_ns"]),
                "duration_s": duration_s,
                "fps": round(fps, 2),
            }
            return preview_dir, frames, result
        except Exception:
            shutil.rmtree(preview_dir, ignore_errors=True)
            raise

    def _set_preferred_camera_locked(self, topic: str) -> None:
        if topic == self.preferred_camera_topic:
            return
        self.preferred_camera_topic = topic
        self.preload_generation += 1
        self.preload_failures.clear()
        for cached_item in self.preload_cache.values():
            if cached_item.preview_dir is not None:
                shutil.rmtree(cached_item.preview_dir, ignore_errors=True)
            cached_item.preview_topic = None
            cached_item.preview_frames = None
            cached_item.preview_dir = None
            cached_item.preview_result = None
            cached_item.preview_attempted_topic = None
        self.preload_status = {
            "state": "queued",
            "message": "等待后台预加载",
            "cached_items": 0,
            "target_items": min(PRELOAD_AHEAD, max(0, len(self.files) - self.current_index - 1)),
        }

    def _preload_targets_locked(self, topic: str) -> list[int]:
        stop = min(len(self.files), self.current_index + PRELOAD_AHEAD + 1)
        return [
            index
            for index in range(self.current_index + 1, stop)
            if index not in self.preload_failures
            and (
                index not in self.preload_cache or self.preload_cache[index].preview_attempted_topic != topic
            )
        ]

    def _schedule_preload_locked(self) -> None:
        topic = self.preferred_camera_topic
        if topic is None or self.current_index >= len(self.files):
            return
        targets = self._preload_targets_locked(topic)
        target_count = min(PRELOAD_AHEAD, max(0, len(self.files) - self.current_index - 1))
        cached_count = target_count - len(targets)
        if not targets:
            self.preload_status = {
                "state": "completed",
                "message": f"已预加载 {cached_count} 条后续数据",
                "cached_items": cached_count,
                "target_items": target_count,
            }
            return
        if self.preload_thread is not None and self.preload_thread.is_alive():
            return
        generation = self.preload_generation
        self.preload_status = {
            "state": "running",
            "message": f"正在预加载后续数据（{cached_count}/{target_count}）",
            "cached_items": cached_count,
            "target_items": target_count,
        }
        thread = threading.Thread(
            target=self._run_preload,
            args=(generation, topic),
            name="mcap-split-preload",
            daemon=True,
        )
        self.preload_thread = thread
        thread.start()

    def _run_preload(self, generation: int, topic: str) -> None:
        try:
            while True:
                with self.lock:
                    if generation != self.preload_generation or topic != self.preferred_camera_topic:
                        return
                    targets = self._preload_targets_locked(topic)
                    if not targets:
                        return
                    index = targets[0]
                    path = self.files[index]
                    cached_item = self.preload_cache.get(index)
                    target_count = min(
                        PRELOAD_AHEAD,
                        max(0, len(self.files) - self.current_index - 1),
                    )
                    cached_count = target_count - len(targets)
                    self.preload_status = {
                        "state": "running",
                        "message": f"正在预加载第 {index + 1} 条数据（{cached_count}/{target_count}）",
                        "cached_items": cached_count,
                        "target_items": target_count,
                    }

                try:
                    if cached_item is None:
                        cached_item = self._build_item_cache(path)
                        with self.lock:
                            if (
                                generation != self.preload_generation
                                or topic != self.preferred_camera_topic
                                or index <= self.current_index
                            ):
                                return
                            self.preload_cache[index] = cached_item

                    known_topics = {camera["topic"] for camera in cached_item.result["cameras"]}
                    if topic in known_topics:
                        preview_dir, frames, result = self._build_preview_cache(
                            path,
                            topic,
                            self.temp_root / f"preload-{generation:04d}-{index:06d}",
                        )
                    else:
                        preview_dir, frames, result = None, None, None

                    with self.lock:
                        if (
                            generation != self.preload_generation
                            or topic != self.preferred_camera_topic
                            or index <= self.current_index
                        ):
                            if preview_dir is not None:
                                shutil.rmtree(preview_dir, ignore_errors=True)
                            return
                        cached_item.preview_attempted_topic = topic
                        cached_item.preview_topic = topic if preview_dir is not None else None
                        cached_item.preview_frames = frames
                        cached_item.preview_dir = preview_dir
                        cached_item.preview_result = result
                except Exception as error:
                    logging.warning("Could not preload MCAP item %s: %s", path, error)
                    with self.lock:
                        if generation == self.preload_generation:
                            if cached_item is not None and index in self.preload_cache:
                                cached_item.preview_attempted_topic = topic
                            else:
                                self.preload_failures.add(index)
        finally:
            with self.lock:
                if self.preload_thread is threading.current_thread():
                    self.preload_thread = None
                if generation == self.preload_generation and topic == self.preferred_camera_topic:
                    self._schedule_preload_locked()

    def preview_status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.preview_job)

    def preview_timeline(self) -> dict[str, Any]:
        with self.lock:
            if self.preview_job["state"] != "completed":
                raise RuntimeError("Camera preview is not ready.")
            frames = [
                {"index": frame["index"], "timestamp_ns": str(frame["timestamp_ns"])}
                for frame in self.preview_frames
            ]
            return {"topic": self.preview_topic, "frames": frames}

    def preview_frame(self, frame_index: int) -> bytes:
        with self.lock:
            if self.preview_job["state"] != "completed" or self.preview_dir is None:
                raise RuntimeError("Camera preview is not ready.")
            if not 0 <= frame_index < len(self.preview_frames):
                raise ValueError(f"Frame index is out of range: {frame_index}")
            path = self.preview_dir / f"{frame_index:08d}.jpg"
        return path.read_bytes()

    def start_split(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_index = payload.get("item_index")
        topic = str(payload.get("topic", "")).strip()
        raw_breakpoints = payload.get("breakpoints_ns")
        raw_outputs = payload.get("output_paths")
        if not isinstance(raw_breakpoints, list):
            raise ValueError("breakpoints_ns must be a list of integer timestamps.")
        if not all(
            isinstance(item, int) or (isinstance(item, str) and item.isdigit()) for item in raw_breakpoints
        ):
            raise ValueError("breakpoints_ns must contain integer timestamp strings.")
        breakpoints = [int(item) for item in raw_breakpoints]
        confirmed_segment_count = payload.get("confirm_segment_count") is True
        overwrite_existing = payload.get("overwrite_existing") is True
        if not isinstance(raw_outputs, list) or not all(isinstance(item, str) for item in raw_outputs):
            raise ValueError("output_paths must be a list of file paths.")
        output_paths = [Path(item) for item in raw_outputs]

        with self.lock:
            self._require_current_item(item_index)
            if self.preview_job["state"] != "completed" or topic != self.preview_topic:
                raise RuntimeError("Prepare the selected camera preview before splitting.")
            timestamps = [frame["timestamp_ns"] for frame in self.preview_frames]
            timestamp_set = set(timestamps[1:])
            if any(timestamp not in timestamp_set for timestamp in breakpoints):
                raise ValueError("Every breakpoint must match a camera frame after the first frame.")
            segment_count = len(breakpoints) + 1
            if (
                self.expected_segments is not None
                and segment_count != self.expected_segments
                and not confirmed_segment_count
            ):
                raise ValueError(
                    f"This batch normally uses {self.expected_segments} segments, but the current "
                    f"item uses {segment_count}. Explicit confirmation is required."
                )
            if self.split_job["state"] in {"queued", "running"}:
                raise RuntimeError("An MCAP split job is already running.")
            index = self.current_index
            path = self.files[index]
            validate_split_request(path, breakpoints, output_paths, overwrite_existing)
            self.split_job = {"state": "queued", "progress": 0, "message": "正在准备分割"}
        threading.Thread(
            target=self._run_split,
            args=(index, path, breakpoints, output_paths, overwrite_existing),
            name="mcap-split-export",
            daemon=True,
        ).start()
        return self.split_status()

    def _run_split(
        self,
        item_index: int,
        path: Path,
        breakpoints: list[int],
        output_paths: list[Path],
        overwrite_existing: bool,
    ) -> None:
        def update(progress: dict[str, Any]) -> None:
            with self.lock:
                if item_index == self.current_index:
                    self.split_job = {**self.split_job, "state": "running", **progress}

        try:
            update({"message": "开始分割 MCAP", "progress": 0})
            result = split_mcap_file(
                path,
                breakpoints,
                output_paths,
                update,
                overwrite_existing=overwrite_existing,
            )
            with self.lock:
                if item_index != self.current_index:
                    return
                if self.expected_segments is None:
                    self.expected_segments = result["segments"]
                self.history.append(
                    {
                        "index": item_index,
                        "source": str(path),
                        "state": "completed",
                        "segments": result["segments"],
                        "outputs": result["outputs"],
                    }
                )
                self.split_job = {
                    "state": "completed",
                    "progress": 100,
                    "message": "分割完成，正在进入下一条数据",
                    "result": result,
                }
                self._advance_locked()
        except Exception as error:
            logging.exception("MCAP split failed")
            with self.lock:
                if item_index == self.current_index:
                    self.split_job = {
                        "state": "failed",
                        "progress": 100,
                        "message": str(error),
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }

    def split_status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.split_job)

    def skip_current(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._require_current_item(payload.get("item_index"))
            if self.split_job["state"] in {"queued", "running"}:
                raise RuntimeError("Cannot skip while an MCAP split job is running.")
            path = self.files[self.current_index]
            self.history.append(
                {"index": self.current_index, "source": str(path), "state": "skipped", "outputs": []}
            )
            self._advance_locked()
        return self.session_status()

    def _require_current_item(self, item_index: Any) -> None:
        if not isinstance(item_index, int) or item_index != self.current_index:
            raise ValueError("The requested item is no longer current. Refresh the session state.")
        if self.current_index >= len(self.files):
            raise RuntimeError("All MCAP files have already been processed.")

    def _clear_preview_locked(self) -> None:
        if self.preview_dir is not None:
            shutil.rmtree(self.preview_dir, ignore_errors=True)
        self.preview_dir = None
        self.preview_frames = []
        self.preview_topic = None
        self.preview_job = {"state": "idle", "progress": 0, "message": "请选择相机"}

    def _advance_locked(self) -> None:
        self._clear_preview_locked()
        self.thumbnails = {}
        self.item_result = None
        self.current_index += 1
        cached_item = self.preload_cache.pop(self.current_index, None)
        self.preload_failures.discard(self.current_index)
        if cached_item is None:
            self.item_job = {"state": "idle", "progress": 0, "message": "等待解析"}
        else:
            self._adopt_item_cache_locked(cached_item)
            self.item_job = {
                "state": "completed",
                "progress": 100,
                "message": "已加载后台条目缓存",
            }
        for index in [index for index in self.preload_cache if index < self.current_index]:
            stale_item = self.preload_cache.pop(index)
            if stale_item.preview_dir is not None:
                shutil.rmtree(stale_item.preview_dir, ignore_errors=True)
        self._schedule_preload_locked()


class McapSplitUiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], state: ApplicationState) -> None:
        self.state = state
        super().__init__(server_address, McapSplitRequestHandler)


class McapSplitRequestHandler(BaseHTTPRequestHandler):
    server: McapSplitUiServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/session":
            self._handle_json(self.server.state.session_status)
            return
        if path == "/api/preview/status":
            self._send_json(self.server.state.preview_status())
            return
        if path == "/api/preview/timeline":
            self._handle_json(self.server.state.preview_timeline)
            return
        if path == "/api/split/status":
            self._send_json(self.server.state.split_status())
            return
        if path == "/api/camera/thumbnail":
            self._handle_image(lambda: self.server.state.thumbnail(self._single_query(query, "topic")))
            return
        if path == "/api/preview/frame":
            self._handle_image(
                lambda: self.server.state.preview_frame(int(self._single_query(query, "index")))
            )
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        routes = {
            "/api/preview": self.server.state.start_preview,
            "/api/split": self.server.state.start_split,
            "/api/item/skip": self.server.state.skip_current,
        }
        callback = routes.get(path)
        if callback is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._handle_json(lambda: callback(self._read_json_body()))

    def log_message(self, format_string: str, *args: Any) -> None:
        logging.debug("%s - %s", self.address_string(), format_string % args)

    def _handle_json(self, callback: Any) -> None:
        try:
            self._send_json(callback())
        except ExistingOutputsError as error:
            self._send_json(
                {
                    "error": str(error),
                    "code": "outputs_exist",
                    "paths": [str(path) for path in error.paths],
                },
                HTTPStatus.CONFLICT,
            )
        except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            logging.exception("Request failed")
            self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_image(self, callback: Any) -> None:
        try:
            data = callback()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            logging.debug("Client disconnected before the image response completed")
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 2_000_000:
            raise ValueError("Invalid request body size.")
        parsed = json.loads(self.rfile.read(content_length))
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object.")
        return parsed

    def _single_query(self, query: dict[str, list[str]], name: str) -> str:
        values = query.get(name)
        if not values or len(values) != 1:
            raise ValueError(f"Missing query parameter: {name}")
        return values[0]

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
    parser = argparse.ArgumentParser(description="Launch a local web UI for splitting MCAP recordings.")
    parser.add_argument(
        "--mcap",
        type=Path,
        required=True,
        help="MCAP file or directory. Directories are searched recursively for .mcap files.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Local interface to bind.")
    parser.add_argument("--port", type=int, default=8766, help="Preferred local port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _create_server(host: str, preferred_port: int, state: ApplicationState) -> McapSplitUiServer:
    if not 0 <= preferred_port <= 65535:
        raise ValueError("--port must be between 0 and 65535.")
    if preferred_port == 0:
        return McapSplitUiServer((host, 0), state)
    last_error: OSError | None = None
    for port in range(preferred_port, min(preferred_port + 20, 65536)):
        try:
            return McapSplitUiServer((host, port), state)
        except OSError as error:
            last_error = error
    raise OSError(f"Could not bind to ports {preferred_port}-{preferred_port + 19}") from last_error


def main() -> None:
    args = _parse_args()
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid --log-level: {args.log_level}")
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    state = ApplicationState(args.mcap)
    server = _create_server(args.host, args.port, state)
    port = server.server_address[1]
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{browser_host}:{port}"
    logging.info("Found %d MCAP file(s) under %s", len(state.files), state.source)
    logging.info("MCAP split UI: %s", url)

    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping MCAP split UI")
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
