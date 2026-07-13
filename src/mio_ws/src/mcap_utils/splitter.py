#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import bisect
import os
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from mio_ws.src.mcap_utils.common import ProgressCallback


def suggested_output_paths(source: Path, segments: int) -> list[Path]:
    output_dir = source.parent / f"{source.stem}_split"
    return [output_dir / f"{source.stem}_part_{index:03d}.mcap" for index in range(segments)]


def validate_split_request(source: Path, cut_timestamps_ns: list[int], output_paths: list[Path]) -> None:
    source = source.expanduser().resolve()
    if not cut_timestamps_ns:
        raise ValueError("At least one breakpoint is required.")
    if cut_timestamps_ns != sorted(set(cut_timestamps_ns)):
        raise ValueError("Breakpoints must be unique and sorted by timestamp.")
    if len(output_paths) != len(cut_timestamps_ns) + 1:
        raise ValueError("N breakpoints require exactly N+1 output paths.")

    resolved_paths = [path.expanduser().resolve() for path in output_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Output paths must be unique.")
    for path in resolved_paths:
        if path.suffix.lower() != ".mcap":
            raise ValueError(f"Output path must end with .mcap: {path}")
        if path == source:
            raise ValueError("An output path cannot overwrite the input MCAP file.")
        if path.exists():
            raise FileExistsError(f"Output file already exists: {path}")


def split_mcap_file(
    source: Path,
    cut_timestamps_ns: list[int],
    output_paths: list[Path],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    try:
        from mcap.reader import make_reader
        from mcap.writer import Writer
    except ImportError as err:
        raise ImportError("MCAP support is missing. Install it with: uv pip install mcap") from err

    source = source.expanduser().resolve()
    outputs = [path.expanduser().resolve() for path in output_paths]
    validate_split_request(source, cut_timestamps_ns, outputs)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    temporary_paths = [output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp") for output in outputs]
    renamed: list[Path] = []
    message_counts = [0 for _ in outputs]
    try:
        with source.open("rb") as source_stream, ExitStack() as stack:
            reader = make_reader(source_stream)
            header = reader.get_header()
            summary = reader.get_summary()
            total_messages = (
                summary.statistics.message_count
                if summary is not None and summary.statistics is not None
                else 0
            )
            metadata_records = list(reader.iter_metadata())
            attachments = list(reader.iter_attachments())

            streams = [stack.enter_context(path.open("wb")) for path in temporary_paths]
            writers = [Writer(stream) for stream in streams]
            schema_maps: list[dict[int, int]] = [{} for _ in writers]
            channel_maps: list[dict[int, int]] = [{} for _ in writers]
            for writer in writers:
                writer.start(profile=header.profile, library=header.library)
                for metadata in metadata_records:
                    writer.add_metadata(metadata.name, dict(metadata.metadata))
                for attachment in attachments:
                    writer.add_attachment(
                        create_time=attachment.create_time,
                        log_time=attachment.log_time,
                        name=attachment.name,
                        media_type=attachment.media_type,
                        data=attachment.data,
                    )

            for message_index, (schema, channel, message) in enumerate(reader.iter_messages(), start=1):
                segment_index = bisect.bisect_right(cut_timestamps_ns, message.log_time)
                writer = writers[segment_index]
                if schema is None:
                    schema_id = 0
                else:
                    schema_id = schema_maps[segment_index].get(schema.id, 0)
                    if schema_id == 0:
                        schema_id = writer.register_schema(schema.name, schema.encoding, schema.data)
                        schema_maps[segment_index][schema.id] = schema_id
                channel_id = channel_maps[segment_index].get(channel.id, 0)
                if channel_id == 0:
                    channel_id = writer.register_channel(
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_id,
                        metadata=dict(channel.metadata),
                    )
                    channel_maps[segment_index][channel.id] = channel_id
                writer.add_message(
                    channel_id=channel_id,
                    log_time=message.log_time,
                    publish_time=message.publish_time,
                    sequence=message.sequence,
                    data=message.data,
                )
                message_counts[segment_index] += 1
                if progress and (message_index == 1 or message_index % 1000 == 0):
                    percent = message_index / total_messages * 96 if total_messages else 0
                    progress(
                        {
                            "message": f"正在写入消息 {message_index}/{total_messages or '?'}",
                            "progress": round(percent, 1),
                            "messages": message_index,
                            "total_messages": total_messages,
                            "segment": segment_index + 1,
                            "segments": len(outputs),
                        }
                    )

            if any(count == 0 for count in message_counts):
                empty = [str(index + 1) for index, count in enumerate(message_counts) if count == 0]
                raise ValueError(f"Breakpoint selection produced empty segment(s): {', '.join(empty)}")
            for writer in writers:
                writer.finish()

        for temporary, output in zip(temporary_paths, outputs, strict=True):
            os.link(temporary, output)
            renamed.append(output)
            temporary.unlink()
    except Exception:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        for path in renamed:
            path.unlink(missing_ok=True)
        raise

    result = {
        "source": str(source),
        "outputs": [str(path) for path in outputs],
        "message_counts": message_counts,
        "segments": len(outputs),
    }
    if progress:
        progress({"message": "MCAP 分割完成", "progress": 100, "result": result})
    return result
