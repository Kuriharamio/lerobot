# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mcap.reader import make_reader
from mcap.writer import Writer
from PIL import Image

from mio_ws.src.mcap_utils.common import decode_image, discover_mcap_files, scan_mcap_item
from mio_ws.src.mcap_utils.mcap_split.server import ApplicationState
from mio_ws.src.mcap_utils.splitter import (
    split_mcap_file,
    suggested_output_paths,
    validate_split_request,
)


def _write_test_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="test-profile", library="test-library")
        image_schema = writer.register_schema("sensor_msgs/msg/Image", "ros2msg", b"image-schema")
        state_schema = writer.register_schema("example.State", "jsonschema", b"{}")
        image_channel = writer.register_channel("/camera/front", "cdr", image_schema, {"camera": "front"})
        state_channel = writer.register_channel("/state", "json", state_schema, {"role": "state"})
        writer.add_metadata("episode", {"task": "test-task"})
        writer.add_attachment(
            create_time=10,
            log_time=20,
            name="calibration.json",
            media_type="application/json",
            data=b'{"fx": 100}',
        )
        records = [
            (image_channel, 100, b"image-0"),
            (state_channel, 150, b"state-0"),
            (image_channel, 200, b"image-1"),
            (state_channel, 250, b"state-1"),
            (image_channel, 300, b"image-2"),
            (state_channel, 350, b"state-2"),
            (image_channel, 400, b"image-3"),
        ]
        for sequence, (channel, timestamp, data) in enumerate(records, start=7):
            writer.add_message(
                channel_id=channel,
                log_time=timestamp,
                publish_time=timestamp + 3,
                sequence=sequence,
                data=data,
            )
        writer.finish()


def _read_mcap(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        reader = make_reader(stream)
        header = reader.get_header()
        messages = [
            {
                "schema": schema.name if schema is not None else None,
                "topic": channel.topic,
                "channel_metadata": dict(channel.metadata),
                "log_time": message.log_time,
                "publish_time": message.publish_time,
                "sequence": message.sequence,
                "data": message.data,
            }
            for schema, channel, message in reader.iter_messages()
        ]
        metadata = [(item.name, dict(item.metadata)) for item in reader.iter_metadata()]
        attachments = [
            (item.create_time, item.log_time, item.name, item.media_type, item.data)
            for item in reader.iter_attachments()
        ]
    return {
        "profile": header.profile,
        "library": header.library,
        "messages": messages,
        "metadata": metadata,
        "attachments": attachments,
    }


def test_discover_mcap_files_recursively_and_validate_input(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "b.MCAP"
    first = tmp_path / "a.mcap"
    first.touch()
    second.touch()
    (tmp_path / "ignored.txt").touch()
    (tmp_path / "empty").mkdir()

    assert discover_mcap_files(tmp_path) == [first, second]
    assert discover_mcap_files(first) == [first]

    with pytest.raises(ValueError, match="not an MCAP"):
        discover_mcap_files(tmp_path / "ignored.txt")
    with pytest.raises(ValueError, match="No .mcap files"):
        discover_mcap_files(tmp_path / "empty")


def test_split_mcap_preserves_records_and_uses_half_open_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    _write_test_mcap(source)
    outputs = [tmp_path / "part-0.mcap", tmp_path / "part-1.mcap", tmp_path / "part-2.mcap"]

    result = split_mcap_file(source, [200, 400], outputs)

    assert result["message_counts"] == [2, 4, 1]
    parts = [_read_mcap(path) for path in outputs]
    assert [[message["log_time"] for message in part["messages"]] for part in parts] == [
        [100, 150],
        [200, 250, 300, 350],
        [400],
    ]
    for part in parts:
        assert part["profile"] == "test-profile"
        assert part["library"] == "test-library"
        assert part["metadata"] == [("episode", {"task": "test-task"})]
        assert part["attachments"] == [(10, 20, "calibration.json", "application/json", b'{"fx": 100}')]

    combined = [message for part in parts for message in part["messages"]]
    original = _read_mcap(source)["messages"]
    assert combined == original
    assert combined[0]["channel_metadata"] == {"camera": "front"}
    assert combined[1]["channel_metadata"] == {"role": "state"}


@pytest.mark.parametrize(
    ("cuts", "outputs", "message"),
    [
        ([], ["one.mcap"], "At least one breakpoint"),
        ([300, 200], ["one.mcap", "two.mcap", "three.mcap"], "unique and sorted"),
        ([200, 200], ["one.mcap", "two.mcap", "three.mcap"], "unique and sorted"),
        ([200], ["one.mcap"], r"exactly N\+1"),
        ([200], ["one.mcap", "one.mcap"], "must be unique"),
        ([200], ["one.txt", "two.mcap"], "must end with .mcap"),
    ],
)
def test_validate_split_request_rejects_invalid_requests(
    tmp_path: Path, cuts: list[int], outputs: list[str], message: str
) -> None:
    source = tmp_path / "source.mcap"
    source.touch()
    with pytest.raises((ValueError, FileExistsError), match=message):
        validate_split_request(source, cuts, [tmp_path / output for output in outputs])


def test_validate_split_request_rejects_input_and_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    source.touch()
    existing = tmp_path / "existing.mcap"
    existing.touch()

    with pytest.raises(ValueError, match="cannot overwrite"):
        validate_split_request(source, [200], [source, tmp_path / "other.mcap"])
    with pytest.raises(FileExistsError, match="already exists"):
        validate_split_request(source, [200], [existing, tmp_path / "other.mcap"])


def test_split_mcap_rolls_back_empty_segments_and_temporary_files(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    _write_test_mcap(source)
    outputs = [tmp_path / "one.mcap", tmp_path / "two.mcap"]

    with pytest.raises(ValueError, match="empty segment"):
        split_mcap_file(source, [99], outputs)

    assert not any(path.exists() for path in outputs)
    assert not list(tmp_path.glob(".*.tmp"))


def test_suggested_output_paths_use_a_sibling_directory(tmp_path: Path) -> None:
    source = tmp_path / "episode.mcap"
    assert suggested_output_paths(source, 3) == [
        tmp_path / "episode_split" / "episode_part_000.mcap",
        tmp_path / "episode_split" / "episode_part_001.mcap",
        tmp_path / "episode_split" / "episode_part_002.mcap",
    ]


def test_scan_mcap_item_counts_messages_and_camera_topics(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    _write_test_mcap(source)

    result = scan_mcap_item(source)

    assert result["messages"] == 7
    assert result["start_ns"] == 100
    assert result["end_ns"] == 400
    assert result["cameras"] == [
        {
            "topic": "/camera/front",
            "schema": "sensor_msgs/msg/Image",
            "frames": 4,
            "first_ns": 100,
            "last_ns": 400,
            "fps": 10_000_000.0,
        }
    ]


def test_decode_ros_raw_and_compressed_images() -> None:
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    raw_message = SimpleNamespace(
        height=1,
        width=2,
        encoding="rgb8",
        step=6,
        data=rgb.tobytes(),
    )
    np.testing.assert_array_equal(decode_image(raw_message), rgb)

    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    compressed_message = SimpleNamespace(format="png", data=buffer.getvalue())
    np.testing.assert_array_equal(decode_image(compressed_message), rgb)


def test_preview_timeline_serializes_epoch_nanoseconds_without_precision_loss(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    _write_test_mcap(source)
    state = ApplicationState(source)
    timestamp = 1_750_000_000_123_456_789
    try:
        state.preview_job = {"state": "completed"}
        state.preview_topic = "/camera/front"
        state.preview_frames = [{"index": 0, "timestamp_ns": timestamp}]

        timeline = state.preview_timeline()

        assert timeline["frames"] == [{"index": 0, "timestamp_ns": str(timestamp)}]
    finally:
        state.close()
