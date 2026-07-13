#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import io
import math
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class CameraFrame:
    timestamp_ns: int
    image: np.ndarray


@dataclass(frozen=True)
class EncodedVideoPacket:
    format: str
    data: bytes


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


def topic_kind(schema_name: str) -> str:
    normalized = schema_name.lower()
    if "image" in normalized or "video" in normalized:
        return "image"
    if any(token in normalized for token in ("joint", "array", "vector", "pose", "twist", "float")):
        return "vector"
    return "unknown"


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


def _fb_vector_uint8(buffer: bytes, field_location: int) -> bytes:
    start = field_location + struct.unpack_from("<I", buffer, field_location)[0]
    size = struct.unpack_from("<I", buffer, start)[0]
    return bytes(buffer[start + 4 : start + 4 + size])


def decode_foxglove_compressed_video(buffer: bytes) -> EncodedVideoPacket:
    root = _fb_root_table(buffer)
    data_location = _fb_table_field(buffer, root, 2)
    format_location = _fb_table_field(buffer, root, 3)
    if data_location is None or format_location is None:
        raise ValueError("foxglove.CompressedVideo message must contain data and format.")
    return EncodedVideoPacket(
        format=_fb_string(buffer, format_location),
        data=_fb_vector_uint8(buffer, data_location),
    )


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


def decode_image(message: Any) -> np.ndarray:
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


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


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


def _ros2_decoder_factory() -> Any | None:
    try:
        from mcap_ros2.decoder import DecoderFactory

        return DecoderFactory()
    except ImportError:
        return None


def _decode_camera_message(schema: Any, channel: Any, data: bytes, factory: Any | None) -> Any:
    schema_name = getattr(schema, "name", "") or ""
    if (
        getattr(schema, "encoding", None) == "flatbuffer"
        and getattr(channel, "message_encoding", None) == "flatbuffer"
        and schema_name == "foxglove.CompressedVideo"
    ):
        return decode_foxglove_compressed_video(data)
    if factory is not None:
        decoder = factory.decoder_for(channel.message_encoding, schema)
        if decoder is not None:
            return decoder(data)
    raise ValueError(f"Unsupported camera schema {schema_name!r} on topic {channel.topic!r}.")


def iter_camera_frames(
    path: Path,
    topic: str,
    progress: ProgressCallback | None = None,
) -> Iterator[CameraFrame]:
    try:
        import av
        from mcap.reader import make_reader
    except ImportError as err:
        raise ImportError("Camera preview requires mcap, mcap-ros2-support, Pillow, and PyAV.") from err

    factory = _ros2_decoder_factory()
    codec: Any | None = None
    codec_name: str | None = None
    last_timestamp_ns = 0
    yielded = 0
    with path.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        expected = 0
        if summary is not None and summary.statistics is not None:
            channel_ids = {
                channel_id for channel_id, channel in summary.channels.items() if channel.topic == topic
            }
            expected = sum(
                summary.statistics.channel_message_counts.get(channel_id, 0) for channel_id in channel_ids
            )

        for index, (schema, channel, message) in enumerate(reader.iter_messages(topics=[topic]), start=1):
            last_timestamp_ns = message.log_time
            decoded = _decode_camera_message(schema, channel, message.data, factory)
            if isinstance(decoded, EncodedVideoPacket):
                next_codec_name = _video_codec_name(decoded.format)
                if codec is None:
                    codec_name = next_codec_name
                    codec = av.CodecContext.create(codec_name, "r")
                elif codec_name != next_codec_name:
                    raise ValueError(f"Camera codec changed from {codec_name} to {next_codec_name}.")
                packet_data = decoded.data
                if codec_name in {"h264", "hevc"}:
                    packet_data = _normalize_length_prefixed_video_packet(packet_data)
                if not packet_data:
                    continue
                try:
                    frames = codec.decode(av.Packet(packet_data))
                except av.error.InvalidDataError:
                    continue
                for frame in frames:
                    yielded += 1
                    yield CameraFrame(
                        message.log_time, np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                    )
            else:
                yielded += 1
                yield CameraFrame(message.log_time, decode_image(decoded))
            if progress and (index == 1 or index % 25 == 0):
                progress(
                    {
                        "message": f"正在准备相机预览：{index}/{expected or '?'}",
                        "progress": round(index / expected * 100, 1) if expected else 0,
                        "decoded_frames": yielded,
                    }
                )

    if codec is not None:
        try:
            delayed_frames = codec.decode(None)
        except av.error.InvalidDataError:
            delayed_frames = []
        for frame in delayed_frames:
            yielded += 1
            yield CameraFrame(last_timestamp_ns, np.ascontiguousarray(frame.to_ndarray(format="rgb24")))
    if yielded == 0:
        raise ValueError(f"No camera frames could be decoded from topic {topic!r} in {path}.")


def scan_mcap_item(path: Path) -> dict[str, Any]:
    try:
        from mcap.reader import make_reader
    except ImportError as err:
        raise ImportError("MCAP support is missing. Install it with: uv pip install mcap") from err

    per_topic: dict[str, dict[str, Any]] = {}
    start_ns: int | None = None
    end_ns: int | None = None
    total_messages = 0
    with path.open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages():
            total_messages += 1
            start_ns = message.log_time if start_ns is None else min(start_ns, message.log_time)
            end_ns = message.log_time if end_ns is None else max(end_ns, message.log_time)
            schema_name = getattr(schema, "name", "") or "unknown"
            if topic_kind(schema_name) != "image":
                continue
            item = per_topic.setdefault(
                channel.topic,
                {
                    "topic": channel.topic,
                    "schema": schema_name,
                    "frames": 0,
                    "first_ns": message.log_time,
                    "last_ns": message.log_time,
                },
            )
            item["frames"] += 1
            item["first_ns"] = min(item["first_ns"], message.log_time)
            item["last_ns"] = max(item["last_ns"], message.log_time)

    cameras = []
    for item in per_topic.values():
        duration_s = (item["last_ns"] - item["first_ns"]) * 1e-9
        fps = (item["frames"] - 1) / duration_s if item["frames"] > 1 and duration_s > 0 else 0.0
        cameras.append({**item, "fps": round(fps, 2)})
    cameras.sort(key=lambda item: item["topic"])
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "messages": total_messages,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_s": ((end_ns - start_ns) * 1e-9 if start_ns is not None and end_ns is not None else 0.0),
        "cameras": cameras,
    }
