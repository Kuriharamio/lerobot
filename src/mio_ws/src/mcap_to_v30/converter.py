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

import bisect
import io
import logging
import math
import struct
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lerobot.datasets import LeRobotDataset

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class McapSample:
    timestamp_s: float
    value: Any


@dataclass(frozen=True)
class EncodedVideoFrame:
    format: str
    data: bytes


@dataclass(frozen=True)
class EpisodeStreams:
    streams: dict[str, list[McapSample]]
    times: dict[str, list[float]]


@dataclass(frozen=True)
class FeatureMapping:
    target: str
    topics: tuple[str, ...]


@dataclass(frozen=True)
class FeatureSpec:
    mapping: FeatureMapping
    kind: str


def discover_mcap_files(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".mcap":
            raise ValueError(f"Input file is not an MCAP file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)

    files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".mcap")
    if not files:
        raise ValueError(f"No .mcap files found recursively under {source}")
    return files


def _format_timestamp(timestamp_ns: int | None) -> str | None:
    if not timestamp_ns:
        return None
    return datetime.fromtimestamp(timestamp_ns * 1e-9).astimezone().isoformat(timespec="seconds")


def _topic_kind(schema_name: str) -> str:
    normalized = schema_name.lower()
    if "image" in normalized or "video" in normalized:
        return "image"
    if any(token in normalized for token in ("joint", "array", "vector", "pose", "twist", "float")):
        return "vector"
    return "unknown"


def scan_mcap_source(source: Path) -> dict[str, Any]:
    try:
        from mcap.reader import make_reader
    except ImportError as err:
        raise ImportError("MCAP support is missing. Install it with: uv pip install mcap") from err

    files = discover_mcap_files(source)
    source = source.expanduser().resolve()
    topic_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "frames": 0,
            "files": 0,
            "intervals": 0,
            "duration_s": 0.0,
            "schemas": set(),
            "schema_encodings": set(),
            "message_encodings": set(),
        }
    )
    metadata_variants: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    file_summaries: list[dict[str, Any]] = []
    total_messages = 0
    total_duration_s = 0.0
    first_time_ns: int | None = None
    last_time_ns: int | None = None

    for path in files:
        per_topic: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "first_ns": None, "last_ns": None}
        )
        with path.open("rb") as stream:
            reader = make_reader(stream)
            summary = reader.get_summary()
            statistics = summary.statistics if summary is not None else None

            for metadata in reader.iter_metadata():
                for key, value in metadata.metadata.items():
                    variant_key = str(value)
                    variant = metadata_variants[(metadata.name, key)].setdefault(
                        variant_key, {"value": str(value), "files": []}
                    )
                    if len(variant["files"]) < 20:
                        relative_path = str(path.relative_to(source)) if source.is_dir() else path.name
                        variant["files"].append(relative_path)

            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                topic_file = per_topic[topic]
                topic_file["count"] += 1
                topic_file["first_ns"] = message.log_time if topic_file["first_ns"] is None else min(
                    topic_file["first_ns"], message.log_time
                )
                topic_file["last_ns"] = message.log_time if topic_file["last_ns"] is None else max(
                    topic_file["last_ns"], message.log_time
                )

                totals = topic_totals[topic]
                if schema is not None:
                    totals["schemas"].add(schema.name or "unknown")
                    totals["schema_encodings"].add(schema.encoding or "unknown")
                totals["message_encodings"].add(channel.message_encoding or "unknown")

        file_message_count = sum(item["count"] for item in per_topic.values())
        file_start_ns = min(
            (item["first_ns"] for item in per_topic.values() if item["first_ns"] is not None), default=None
        )
        file_end_ns = max(
            (item["last_ns"] for item in per_topic.values() if item["last_ns"] is not None), default=None
        )
        file_duration_s = (
            (file_end_ns - file_start_ns) * 1e-9
            if file_start_ns is not None and file_end_ns is not None
            else 0.0
        )
        relative_path = str(path.relative_to(source)) if source.is_dir() else path.name
        file_summaries.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "messages": file_message_count,
                "duration_s": round(file_duration_s, 3),
                "start_time": _format_timestamp(file_start_ns),
                "end_time": _format_timestamp(file_end_ns),
                "attachments": statistics.attachment_count if statistics is not None else 0,
            }
        )

        total_messages += file_message_count
        total_duration_s += file_duration_s
        if file_start_ns is not None:
            first_time_ns = file_start_ns if first_time_ns is None else min(first_time_ns, file_start_ns)
        if file_end_ns is not None:
            last_time_ns = file_end_ns if last_time_ns is None else max(last_time_ns, file_end_ns)

        for topic, item in per_topic.items():
            totals = topic_totals[topic]
            totals["frames"] += item["count"]
            totals["files"] += 1
            if item["count"] > 1 and item["first_ns"] is not None and item["last_ns"] is not None:
                totals["intervals"] += item["count"] - 1
                totals["duration_s"] += (item["last_ns"] - item["first_ns"]) * 1e-9

    topic_shapes = _probe_topic_shapes(files, set(topic_totals))
    topics = []
    for name, totals in sorted(topic_totals.items()):
        fps = totals["intervals"] / totals["duration_s"] if totals["duration_s"] > 0 else 0.0
        schemas = sorted(totals["schemas"])
        topics.append(
            {
                "name": name,
                "frames": totals["frames"],
                "fps": round(fps, 2),
                "files": totals["files"],
                "schema": ", ".join(schemas) if schemas else "unknown",
                "schema_encoding": ", ".join(sorted(totals["schema_encodings"])),
                "message_encoding": ", ".join(sorted(totals["message_encodings"])),
                "kind": _topic_kind(" ".join(schemas)),
                "shape": topic_shapes.get(name),
            }
        )

    metadata_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (group, key), variants in sorted(metadata_variants.items()):
        values = list(variants.values())
        metadata_groups[group].append(
            {
                "key": key,
                "value": values[0]["value"] if len(values) == 1 else None,
                "variants": values if len(values) > 1 else [],
                "variant_count": len(values),
            }
        )

    image_fps = next((topic["fps"] for topic in topics if topic["kind"] == "image" and topic["fps"] > 0), 0)
    fallback_fps = next((topic["fps"] for topic in topics if topic["fps"] > 0), 30)
    suggested_fps = max(1, round(image_fps or fallback_fps))
    source_name = source.stem if source.is_file() else source.name
    suggested_root = source.parent / f"{source_name}_lerobot_v30"

    task_value = "MCAP converted task"
    task_variants = metadata_variants.get(("episode", "task"), {})
    if len(task_variants) == 1:
        task_value = next(iter(task_variants.values()))["value"]

    return {
        "source": str(source),
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "total_messages": total_messages,
        "total_duration_s": round(total_duration_s, 3),
        "start_time": _format_timestamp(first_time_ns),
        "end_time": _format_timestamp(last_time_ns),
        "topics": topics,
        "files": file_summaries,
        "metadata": [{"name": name, "fields": fields} for name, fields in metadata_groups.items()],
        "suggestions": {
            "fps": suggested_fps,
            "repo_id": f"local/{source_name.replace(' ', '_')}_v30",
            "root": str(suggested_root),
            "task": task_value,
        },
    }


def _message_timestamp_s(message: Any) -> float:
    timestamp_ns = getattr(message, "publish_time", None) or getattr(message, "log_time", None)
    if timestamp_ns is None:
        raise ValueError("MCAP message does not expose publish_time or log_time.")
    return float(timestamp_ns) * 1e-9


def _fb_root_table(buffer: bytes) -> int:
    return struct.unpack_from("<I", buffer, 0)[0]


def _fb_vtable_offsets(buffer: bytes, table: int) -> tuple[int, ...]:
    signed_offset = struct.unpack_from("<i", buffer, table)[0]
    vtable = table - signed_offset
    vtable_len = struct.unpack_from("<H", buffer, vtable)[0]
    return struct.unpack_from("<" + "H" * ((vtable_len - 4) // 2), buffer, vtable + 4)


def _fb_table_field(buffer: bytes, table: int, field_index: int) -> int | None:
    offsets = _fb_vtable_offsets(buffer, table)
    if field_index >= len(offsets) or offsets[field_index] == 0:
        return None
    return table + offsets[field_index]


def _fb_string(buffer: bytes, field_location: int) -> str:
    start = field_location + struct.unpack_from("<I", buffer, field_location)[0]
    size = struct.unpack_from("<I", buffer, start)[0]
    return buffer[start + 4 : start + 4 + size].decode()


def _fb_vector_start(buffer: bytes, field_location: int) -> int:
    return field_location + struct.unpack_from("<I", buffer, field_location)[0]


def _fb_vector_uint8(buffer: bytes, field_location: int) -> bytes:
    start = _fb_vector_start(buffer, field_location)
    size = struct.unpack_from("<I", buffer, start)[0]
    return bytes(buffer[start + 4 : start + 4 + size])


def _fb_vector_tables(buffer: bytes, field_location: int) -> list[int]:
    start = _fb_vector_start(buffer, field_location)
    size = struct.unpack_from("<I", buffer, start)[0]
    tables = []
    for index in range(size):
        element = start + 4 + index * 4
        tables.append(element + struct.unpack_from("<I", buffer, element)[0])
    return tables


def _decode_foxglove_joint_states(buffer: bytes) -> dict[str, Any]:
    root = _fb_root_table(buffer)
    joints_location = _fb_table_field(buffer, root, 1)
    if joints_location is None:
        raise ValueError("foxglove.JointStates message does not contain joints.")

    names = []
    values: dict[str, list[float]] = {
        "position": [],
        "velocity": [],
        "acceleration": [],
        "effort": [],
    }
    for joint_table in _fb_vector_tables(buffer, joints_location):
        name_location = _fb_table_field(buffer, joint_table, 0)
        names.append(
            _fb_string(buffer, name_location) if name_location is not None else f"joint_{len(names)}"
        )
        for field_index, key in enumerate(values, start=1):
            value_location = _fb_table_field(buffer, joint_table, field_index)
            value = (
                struct.unpack_from("<d", buffer, value_location)[0]
                if value_location is not None
                else np.nan
            )
            values[key].append(value)

    result: dict[str, Any] = {"name": names}
    for key, key_values in values.items():
        array = np.asarray(key_values, dtype=np.float32)
        if not np.all(np.isnan(array)):
            result[key] = array
    return result


def _decode_foxglove_compressed_video(buffer: bytes) -> EncodedVideoFrame:
    root = _fb_root_table(buffer)
    data_location = _fb_table_field(buffer, root, 2)
    format_location = _fb_table_field(buffer, root, 3)
    if data_location is None or format_location is None:
        raise ValueError("foxglove.CompressedVideo message must contain data and format.")
    return EncodedVideoFrame(
        format=_fb_string(buffer, format_location),
        data=_fb_vector_uint8(buffer, data_location),
    )


def _decode_raw_mcap_message(
    schema: Any, channel: Any, message: Any, ros2_decoder_factory: Any | None
) -> Any:
    schema_encoding = getattr(schema, "encoding", None)
    schema_name = getattr(schema, "name", None)
    message_encoding = getattr(channel, "message_encoding", None)

    if schema_encoding == "flatbuffer" and message_encoding == "flatbuffer":
        if schema_name == "foxglove.JointStates":
            return _decode_foxglove_joint_states(message.data)
        if schema_name == "foxglove.CompressedVideo":
            return _decode_foxglove_compressed_video(message.data)
        raise ValueError(f"Unsupported Foxglove flatbuffer schema: {schema_name!r}")

    if ros2_decoder_factory is not None:
        decoder = ros2_decoder_factory.decoder_for(message_encoding, schema)
        if decoder is not None:
            return decoder(message.data)

    raise ValueError(
        f"Unsupported MCAP encoding for schema {schema_name!r}. "
        "ROS 2 messages require mcap-ros2-support; supported Foxglove schemas are "
        "JointStates and CompressedVideo."
    )


def _video_codec_name(format_name: str) -> str:
    normalized = format_name.lower()
    if normalized == "h265":
        return "hevc"
    if normalized in {"h264", "vp9", "av1"}:
        return normalized
    raise ValueError(f"Unsupported Foxglove compressed video format {format_name!r}.")


def _normalize_length_prefixed_video_packet(data: bytes) -> bytes:
    if data.startswith((b"\x00\x00\x00\x01", b"\x00\x00\x01")):
        return data

    position = 0
    units = []
    while position + 4 <= len(data):
        unit_size = int.from_bytes(data[position : position + 4], "big")
        position += 4
        if unit_size <= 0 or position + unit_size > len(data):
            return data
        units.append(b"\x00\x00\x00\x01" + data[position : position + unit_size])
        position += unit_size
    return b"".join(units) if units and position == len(data) else data


def _decode_encoded_video_samples(
    samples: list[McapSample],
    *,
    path: Path | None = None,
    topic: str | None = None,
    warn_invalid: bool = True,
) -> list[McapSample]:
    try:
        import av
    except ImportError as err:
        raise ImportError("Video decoding requires PyAV. Install it with: uv pip install av") from err

    decoded: list[McapSample] = []
    codec_name = _video_codec_name(samples[0].value.format)
    codec = av.CodecContext.create(codec_name, "r")
    invalid_packets = 0
    for sample in samples:
        packet_data = sample.value.data
        if codec_name in {"h264", "hevc"}:
            packet_data = _normalize_length_prefixed_video_packet(packet_data)
        if not packet_data:
            invalid_packets += 1
            continue
        try:
            frames = codec.decode(av.Packet(packet_data))
        except av.error.InvalidDataError:
            invalid_packets += 1
            continue
        for frame in frames:
            image = frame.to_ndarray(format="rgb24")
            decoded.append(McapSample(sample.timestamp_s, np.ascontiguousarray(image)))
    try:
        delayed_frames = codec.decode(None)
    except av.error.InvalidDataError:
        invalid_packets += 1
        delayed_frames = []
    for frame in delayed_frames:
        image = frame.to_ndarray(format="rgb24")
        decoded.append(McapSample(samples[-1].timestamp_s, np.ascontiguousarray(image)))
    if not decoded:
        location = f" in {path}" if path is not None else ""
        topic_label = f" for topic {topic!r}" if topic is not None else ""
        raise ValueError(
            f"No video frames could be decoded{topic_label}{location}; "
            f"codec={codec_name}, packets={len(samples)}, invalid_packets={invalid_packets}."
        )
    if invalid_packets and warn_invalid:
        logging.warning(
            "Skipped %d invalid %s packet(s) for topic %s in %s",
            invalid_packets,
            codec_name,
            topic or "<unknown>",
            path or "<unknown>",
        )
    return decoded


def _read_mcap_streams(path: Path, topics: set[str]) -> EpisodeStreams:
    try:
        from mcap.reader import make_reader
    except ImportError as err:
        raise ImportError("MCAP support is missing. Install it with: uv pip install mcap") from err

    try:
        from mcap_ros2.decoder import DecoderFactory

        ros2_decoder_factory = DecoderFactory()
    except ImportError:
        ros2_decoder_factory = None

    streams: dict[str, list[McapSample]] = {topic: [] for topic in topics}
    with path.open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages(topics=sorted(topics)):
            value = _decode_raw_mcap_message(schema, channel, message, ros2_decoder_factory)
            streams[channel.topic].append(McapSample(_message_timestamp_s(message), value))

    for topic, samples in streams.items():
        samples.sort(key=lambda sample: sample.timestamp_s)
        if samples and all(isinstance(sample.value, EncodedVideoFrame) for sample in samples):
            streams[topic] = _decode_encoded_video_samples(samples, path=path, topic=topic)
    times = {topic: [sample.timestamp_s for sample in samples] for topic, samples in streams.items()}
    return EpisodeStreams(streams=streams, times=times)


def _get_attr_or_item(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value[key]
    return getattr(value, key)


def _as_numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        array = np.asarray([value])
    else:
        return None
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    return array.astype(np.float32, copy=False).reshape(-1)


def _flatten_vector(value: Any) -> np.ndarray:
    direct = _as_numeric_array(value)
    if direct is not None:
        return direct

    for attribute in (
        "data",
        "position",
        "positions",
        "velocity",
        "velocities",
        "effort",
        "command",
        "commands",
    ):
        if isinstance(value, dict) and attribute in value:
            return _flatten_vector(value[attribute])
        if hasattr(value, attribute):
            return _flatten_vector(getattr(value, attribute))

    if all(hasattr(value, attribute) for attribute in ("x", "y", "z")):
        parts = [value.x, value.y, value.z]
        if hasattr(value, "w"):
            parts.append(value.w)
        return _flatten_vector(parts)
    if hasattr(value, "linear") and hasattr(value, "angular"):
        return np.concatenate([_flatten_vector(value.linear), _flatten_vector(value.angular)])
    if hasattr(value, "position") and hasattr(value, "orientation"):
        return np.concatenate([_flatten_vector(value.position), _flatten_vector(value.orientation)])
    raise ValueError(f"Could not infer a numeric vector from message type {type(value)!r}.")


def _bytes_from_ros_sequence(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    return bytes(data)


def _scale_single_channel_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    array = np.nan_to_num(array.astype(np.float32), copy=False)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    minimum = float(finite.min())
    maximum = float(finite.max())
    if math.isclose(minimum, maximum):
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - minimum) / (maximum - minimum) * 255.0, 0, 255).astype(np.uint8)


def _decode_raw_image_buffer(message: Any, dtype: np.dtype, channels: int) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(getattr(message, "step", width * channels * dtype.itemsize))
    expected_row_bytes = width * channels * dtype.itemsize
    if step < expected_row_bytes:
        raise ValueError(f"ROS image step {step} is smaller than expected row bytes {expected_row_bytes}.")
    rows = np.frombuffer(_bytes_from_ros_sequence(message.data), dtype=np.uint8).reshape(height, step)
    return np.ascontiguousarray(rows[:, :expected_row_bytes]).view(dtype).reshape(height, width, channels)


def _decode_image(message: Any) -> np.ndarray:
    if isinstance(message, np.ndarray):
        image = message
    elif hasattr(message, "format") and hasattr(message, "data") and not hasattr(message, "height"):
        image = np.asarray(Image.open(io.BytesIO(_bytes_from_ros_sequence(message.data))).convert("RGB"))
    elif all(hasattr(message, attribute) for attribute in ("height", "width", "encoding", "data")):
        encoding = str(message.encoding).lower()
        if encoding in {"rgb8", "bgr8"}:
            image = _decode_raw_image_buffer(message, np.dtype(np.uint8), 3)
            if encoding == "bgr8":
                image = image[..., ::-1]
        elif encoding in {"rgba8", "bgra8"}:
            image = _decode_raw_image_buffer(message, np.dtype(np.uint8), 4)
            if encoding == "bgra8":
                image = image[..., [2, 1, 0, 3]]
            image = image[..., :3]
        elif encoding in {"mono8", "8uc1"}:
            mono = _decode_raw_image_buffer(message, np.dtype(np.uint8), 1)[..., 0]
            image = np.repeat(mono[..., None], 3, axis=2)
        elif encoding in {"mono16", "16uc1"}:
            mono = _decode_raw_image_buffer(message, np.dtype(np.uint16), 1)[..., 0]
            image = np.repeat(_scale_single_channel_to_uint8(mono)[..., None], 3, axis=2)
        elif encoding == "32fc1":
            mono = _decode_raw_image_buffer(message, np.dtype(np.float32), 1)[..., 0]
            image = np.repeat(_scale_single_channel_to_uint8(mono)[..., None], 3, axis=2)
        else:
            raise ValueError(f"Unsupported ROS image encoding {message.encoding!r}.")
    else:
        raise ValueError(f"Could not decode image from message type {type(message)!r}.")

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError(f"Image must have HWC shape with 3 or 4 channels, got {image.shape}.")
    return np.ascontiguousarray(image[..., :3], dtype=np.uint8)


def _looks_like_image(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim in {2, 3}
    if hasattr(value, "format") and hasattr(value, "data") and not hasattr(value, "height"):
        return True
    return all(hasattr(value, attribute) for attribute in ("height", "width", "encoding", "data"))


def _shape_from_value(value: Any) -> list[int]:
    if _looks_like_image(value):
        image = _decode_image(value)
        return [int(image.shape[0]), int(image.shape[1]), 3]
    return [int(_flatten_vector(value).shape[0])]


def _probe_topic_shapes(files: list[Path], topics: set[str]) -> dict[str, list[int]]:
    try:
        from mcap.reader import make_reader
    except ImportError:
        return {}

    try:
        from mcap_ros2.decoder import DecoderFactory

        ros2_decoder_factory = DecoderFactory()
    except ImportError:
        ros2_decoder_factory = None

    unresolved = set(topics)
    shapes: dict[str, list[int]] = {}
    for path in files:
        if not unresolved:
            break
        video_samples: dict[str, list[McapSample]] = defaultdict(list)
        failed_topics = set()
        with path.open("rb") as stream:
            reader = make_reader(stream)
            for schema, channel, message in reader.iter_messages(topics=sorted(unresolved)):
                topic = channel.topic
                if topic not in unresolved or topic in failed_topics:
                    continue
                try:
                    value = _decode_raw_mcap_message(schema, channel, message, ros2_decoder_factory)
                except (ValueError, AttributeError, KeyError, struct.error):
                    failed_topics.add(topic)
                    continue

                if isinstance(value, EncodedVideoFrame):
                    samples = video_samples[topic]
                    samples.append(McapSample(_message_timestamp_s(message), value))
                    should_try = len(samples) == 3 or len(samples) % 15 == 0
                    if not should_try:
                        continue
                    try:
                        decoded = _decode_encoded_video_samples(
                            samples, path=path, topic=topic, warn_invalid=False
                        )
                    except ValueError:
                        continue
                    shapes[topic] = _shape_from_value(decoded[0].value)
                    unresolved.remove(topic)
                else:
                    try:
                        shapes[topic] = _shape_from_value(value)
                    except (ValueError, AttributeError, KeyError, TypeError):
                        failed_topics.add(topic)
                        continue
                    unresolved.remove(topic)

                if not unresolved:
                    break

        for topic, samples in video_samples.items():
            if topic not in unresolved or not samples:
                continue
            try:
                decoded = _decode_encoded_video_samples(
                    samples, path=path, topic=topic, warn_invalid=False
                )
                shapes[topic] = _shape_from_value(decoded[0].value)
            except (ValueError, AttributeError, KeyError, TypeError):
                continue
            unresolved.remove(topic)
    return shapes


def _nearest_sample(
    episode: EpisodeStreams, topic: str, timestamp_s: float, max_delta_s: float
) -> McapSample | None:
    times = episode.times[topic]
    if not times:
        return None
    position = bisect.bisect_left(times, timestamp_s)
    candidates = []
    if position < len(times):
        candidates.append(position)
    if position > 0:
        candidates.append(position - 1)
    best = min(candidates, key=lambda index: abs(times[index] - timestamp_s))
    if abs(times[best] - timestamp_s) > max_delta_s:
        return None
    return episode.streams[topic][best]


def _validate_mappings(raw_mappings: list[dict[str, Any]]) -> list[FeatureMapping]:
    if not raw_mappings:
        raise ValueError("Create at least one LeRobot mapping before export.")
    mappings = []
    targets = set()
    reserved = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    for raw in raw_mappings:
        target = str(raw.get("target", "")).strip()
        topics = tuple(str(topic).strip() for topic in raw.get("topics", []) if str(topic).strip())
        if not target or not topics:
            raise ValueError("Every mapping needs a target name and at least one source topic.")
        if target in targets:
            raise ValueError(f"Duplicate LeRobot target: {target}")
        if target in reserved:
            raise ValueError(f"{target!r} is managed internally by LeRobot and cannot be mapped.")
        targets.add(target)
        mappings.append(FeatureMapping(target=target, topics=topics))
    return mappings


def _infer_features(
    first_episode: EpisodeStreams, mappings: list[FeatureMapping], use_videos: bool
) -> tuple[dict[str, dict[str, Any]], list[FeatureSpec]]:
    features: dict[str, dict[str, Any]] = {}
    specs: list[FeatureSpec] = []
    for mapping in mappings:
        values = [first_episode.streams[topic][0].value for topic in mapping.topics]
        image_flags = [_looks_like_image(value) for value in values]
        if any(image_flags):
            if len(values) != 1 or not all(image_flags):
                raise ValueError(f"Image target {mapping.target!r} must map from exactly one image topic.")
            image = _decode_image(values[0])
            features[mapping.target] = {
                "dtype": "video" if use_videos else "image",
                "shape": (3, int(image.shape[0]), int(image.shape[1])),
                "names": ["channels", "height", "width"],
            }
            specs.append(FeatureSpec(mapping=mapping, kind="image"))
        else:
            vector = np.concatenate([_flatten_vector(value) for value in values]).astype(
                np.float32, copy=False
            )
            features[mapping.target] = {"dtype": "float32", "shape": tuple(vector.shape), "names": None}
            specs.append(FeatureSpec(mapping=mapping, kind="vector"))
    return features, specs


def _validate_streams(episode: EpisodeStreams, topics: set[str], path: Path) -> None:
    missing = sorted(topic for topic in topics if not episode.streams.get(topic))
    if missing:
        raise ValueError(f"{path} is missing selected topic(s): {', '.join(missing)}")


def _episode_bounds(episode: EpisodeStreams, topics: set[str]) -> tuple[float, float]:
    start = max(episode.streams[topic][0].timestamp_s for topic in topics)
    end = min(episode.streams[topic][-1].timestamp_s for topic in topics)
    if end <= start:
        raise ValueError("Selected topic time ranges do not overlap.")
    return start, end


def _mapping_value(
    episode: EpisodeStreams, spec: FeatureSpec, timestamp_s: float, max_delta_s: float
) -> np.ndarray:
    values = []
    for topic in spec.mapping.topics:
        sample = _nearest_sample(episode, topic, timestamp_s, max_delta_s)
        if sample is None:
            raise ValueError(f"No nearby sample for {topic} at {timestamp_s:.6f}s")
        values.append(sample.value)
    if spec.kind == "image":
        return _decode_image(values[0])
    return np.concatenate([_flatten_vector(value) for value in values]).astype(np.float32, copy=False)


def convert_mcap_source(
    source: Path,
    raw_mappings: list[dict[str, Any]],
    parameters: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    files = discover_mcap_files(source)
    mappings = _validate_mappings(raw_mappings)
    topics = {topic for mapping in mappings for topic in mapping.topics}

    repo_id = str(parameters.get("repo_id", "")).strip()
    root_value = str(parameters.get("root", "")).strip()
    task = str(parameters.get("task", "")).strip()
    robot_type = str(parameters.get("robot_type", "")).strip() or None
    fps = int(parameters.get("fps", 0))
    use_videos = bool(parameters.get("use_videos", True))
    push_to_hub = bool(parameters.get("push_to_hub", False))
    private = bool(parameters.get("private", False))
    if not repo_id or not root_value or not task:
        raise ValueError("repo-id, root, and task are required.")
    if fps <= 0:
        raise ValueError("FPS must be a positive integer.")
    root = Path(root_value).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Export root already exists: {root}")

    if progress:
        progress({"stage": "schema", "message": "正在解析 LeRobot 特征结构", "progress": 0})
    first_episode = _read_mcap_streams(files[0], topics)
    _validate_streams(first_episode, topics, files[0])
    features, specs = _infer_features(first_episode, mappings, use_videos)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
    )
    total_frames = 0
    max_delta_s = 0.6 / fps
    try:
        for file_index, path in enumerate(files):
            if progress:
                progress(
                    {
                        "stage": "episode",
                        "message": f"正在转换 {path.name}",
                        "current_file": str(path),
                        "episode": file_index + 1,
                        "episodes": len(files),
                        "progress": round(file_index / len(files) * 100, 1),
                    }
                )
            episode = first_episode if file_index == 0 else _read_mcap_streams(path, topics)
            _validate_streams(episode, topics, path)
            start_s, end_s = _episode_bounds(episode, topics)
            target_frames = int(math.floor((end_s - start_s) * fps)) + 1
            added = 0
            for frame_index in range(target_frames):
                timestamp_s = start_s + frame_index / fps
                frame: dict[str, Any] = {"task": task}
                try:
                    for spec in specs:
                        frame[spec.mapping.target] = _mapping_value(episode, spec, timestamp_s, max_delta_s)
                except ValueError:
                    continue
                dataset.add_frame(frame)
                added += 1
            if added == 0:
                raise ValueError(f"No complete frames could be built from {path}")
            dataset.save_episode()
            total_frames += added
    finally:
        dataset.finalize()

    if push_to_hub:
        if progress:
            progress({"stage": "upload", "message": "正在上传到 Hugging Face Hub", "progress": 99})
        dataset.push_to_hub(private=private, tags=["mcap"])

    result = {
        "root": str(dataset.root),
        "repo_id": repo_id,
        "episodes": dataset.num_episodes,
        "frames": total_frames,
        "fps": fps,
    }
    if progress:
        progress({"stage": "done", "message": "导出完成", "progress": 100, "result": result})
    return result
