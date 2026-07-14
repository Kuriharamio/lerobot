# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import io
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mcap.reader import make_reader
from mcap.writer import Writer
from mcap_ros2.writer import Writer as Ros2Writer
from PIL import Image

from mio_ws.src.mcap_utils.common import decode_image, discover_mcap_files, scan_mcap_item
from mio_ws.src.mcap_utils.mcap_split.server import ApplicationState, McapSplitRequestHandler
from mio_ws.src.mcap_utils.mcap_to_v30.converter import (
    EpisodeStreams,
    McapSample,
    _describe_topic_value,
    _infer_features,
    _validate_mappings,
    scan_mcap_source,
)
from mio_ws.src.mcap_utils.splitter import (
    ExistingOutputsError,
    split_mcap_file,
    suggested_output_paths,
    suggested_output_pattern,
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


def _write_camera_mcap(path: Path, topic: str = "/camera/front") -> None:
    message_definition = """uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = Ros2Writer(stream)
        schema = writer.register_msgdef("example_msgs/msg/Image", message_definition)
        for index in range(4):
            image = np.full((2, 3, 3), index * 40, dtype=np.uint8)
            writer.write_message(
                topic,
                schema,
                {
                    "height": 2,
                    "width": 3,
                    "encoding": "rgb8",
                    "is_bigendian": 0,
                    "step": 9,
                    "data": image.reshape(-1).tolist(),
                },
                log_time=1_750_000_000_000_000_000 + index * 100_000_000,
                publish_time=1_750_000_000_000_000_017 + index * 100_000_000,
                sequence=index,
            )
        writer.finish()


def _write_joint_state_mcap(path: Path, topic: str = "/joint_states") -> None:
    message_definition = """string[] name
float64[] position
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = Ros2Writer(stream)
        schema = writer.register_msgdef("example_msgs/msg/JointState", message_definition)
        writer.write_message(
            topic,
            schema,
            {
                "name": ["left_joint_1", "left_gripper", "right_joint_1", "right_gripper"],
                "position": [0.1, 0.2, 0.3, 0.4],
            },
            log_time=1_750_000_000_000_000_000,
            publish_time=1_750_000_000_000_000_017,
            sequence=0,
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
    with pytest.raises(ExistingOutputsError, match="already exists") as error:
        validate_split_request(source, [200], [existing, tmp_path / "other.mcap"])
    assert error.value.paths == (existing,)
    validate_split_request(
        source,
        [200],
        [existing, tmp_path / "other.mcap"],
        overwrite_existing=True,
    )


def test_split_mcap_overwrites_existing_outputs_after_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    _write_test_mcap(source)
    outputs = [tmp_path / "one.mcap", tmp_path / "two.mcap"]
    for output in outputs:
        output.write_bytes(b"previous output")

    result = split_mcap_file(source, [300], outputs, overwrite_existing=True)

    assert result["message_counts"] == [4, 3]
    assert [[message["log_time"] for message in _read_mcap(path)["messages"]] for path in outputs] == [
        [100, 150, 200, 250],
        [300, 350, 400],
    ]
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.backup"))


def test_split_mcap_rolls_back_empty_segments_and_temporary_files(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    _write_test_mcap(source)
    outputs = [tmp_path / "one.mcap", tmp_path / "two.mcap"]

    with pytest.raises(ValueError, match="empty segment"):
        split_mcap_file(source, [99], outputs)

    assert not any(path.exists() for path in outputs)
    assert not list(tmp_path.glob(".*.tmp"))


def test_suggested_output_paths_create_batch_level_subdatasets(tmp_path: Path) -> None:
    batch_source = tmp_path / "open_lid_mcap"
    item_source = batch_source / "open_lid_1" / "xx.mcap"
    item_source.parent.mkdir(parents=True)
    item_source.touch()

    assert suggested_output_paths(batch_source, item_source, 3) == [
        tmp_path / "open_lid_mcap_split_1" / "open_lid_1" / "xx.mcap",
        tmp_path / "open_lid_mcap_split_2" / "open_lid_1" / "xx.mcap",
        tmp_path / "open_lid_mcap_split_3" / "open_lid_1" / "xx.mcap",
    ]
    assert suggested_output_pattern(batch_source, item_source) == (
        tmp_path / "open_lid_mcap_split_{segment}" / "open_lid_1" / "xx.mcap"
    )


def test_suggested_output_paths_are_siblings_of_a_nested_batch(tmp_path: Path) -> None:
    batch_source = tmp_path / "open_lid_mcap" / "open_lid_1"
    batch_source.mkdir(parents=True)
    item_source = batch_source / "xx.mcap"
    item_source.touch()

    assert suggested_output_paths(batch_source, item_source, 2) == [
        tmp_path / "open_lid_mcap" / "open_lid_1_split_1" / "xx.mcap",
        tmp_path / "open_lid_mcap" / "open_lid_1_split_2" / "xx.mcap",
    ]


def test_suggested_output_paths_for_a_single_file_keep_its_name(tmp_path: Path) -> None:
    source = tmp_path / "episode.mcap"
    source.touch()

    assert suggested_output_paths(source, source, 2) == [
        tmp_path / "episode_split_1" / "episode.mcap",
        tmp_path / "episode_split_2" / "episode.mcap",
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


def test_scan_mcap_source_generates_names_from_topic_and_index(tmp_path: Path) -> None:
    source = tmp_path / "joint_states.mcap"
    _write_joint_state_mcap(source)

    result = scan_mcap_source(source)

    topic = result["topics"][0]
    assert topic["shape"] == [4]
    assert topic["names"] == [
        "joint_states.dim_1",
        "joint_states.dim_2",
        "joint_states.dim_3",
        "joint_states.dim_4",
    ]


def test_feature_mapping_uses_automatic_or_user_supplied_names() -> None:
    left = SimpleNamespace(name=["left_joint_1", "left_gripper"], position=[0.1, 0.2])
    right = SimpleNamespace(name=["right_joint_1", "right_gripper"], position=[0.3, 0.4])
    episode = EpisodeStreams(
        streams={
            "/left/joint_states": [McapSample(0.0, left)],
            "/right/joint_states": [McapSample(0.0, right)],
        },
        times={"/left/joint_states": [0.0], "/right/joint_states": [0.0]},
    )
    automatic = _validate_mappings(
        [
            {
                "target": "observation.state",
                "topics": ["/left/joint_states", "/right/joint_states"],
            }
        ]
    )
    automatic_features, _ = _infer_features(episode, automatic, use_videos=True)
    assert automatic_features["observation.state"] == {
        "dtype": "float32",
        "shape": (4,),
        "names": [
            "left.joint_states.dim_1",
            "left.joint_states.dim_2",
            "right.joint_states.dim_1",
            "right.joint_states.dim_2",
        ],
    }

    custom_names = ["arm.left", "gripper.left", "arm.right", "gripper.right"]
    customized = _validate_mappings(
        [
            {
                "target": "action",
                "topics": ["/left/joint_states", "/right/joint_states"],
                "names": custom_names,
            }
        ]
    )
    customized_features, _ = _infer_features(episode, customized, use_videos=True)
    assert customized_features["action"]["names"] == custom_names


def test_feature_mapping_rejects_invalid_names() -> None:
    value = SimpleNamespace(name=["joint_1", "joint_2"], position=[0.1, 0.2])
    episode = EpisodeStreams(
        streams={"/joint_states": [McapSample(0.0, value)]},
        times={"/joint_states": [0.0]},
    )
    mappings = _validate_mappings([{"target": "action", "topics": ["/joint_states"], "names": ["only_one"]}])

    with pytest.raises(ValueError, match="2 values but 1 names"):
        _infer_features(episode, mappings, use_videos=True)

    with pytest.raises(ValueError, match="must be unique"):
        _validate_mappings(
            [
                {
                    "target": "action",
                    "topics": ["/joint_states"],
                    "names": ["joint.pos", "joint.pos"],
                }
            ]
        )


def test_describe_topic_value_generates_stable_fallback_names() -> None:
    assert _describe_topic_value([1.0, 2.0], "/arm/command") == {
        "shape": [2],
        "names": ["arm.command.dim_1", "arm.command.dim_2"],
    }


def test_describe_topic_value_uses_one_based_index_for_each_dimension() -> None:
    value = SimpleNamespace(name=[f"j{index}" for index in range(6)], position=np.arange(6))

    assert _describe_topic_value(value, "/observation/follower_left/arm/end_effector_pose")["names"] == [
        "observation.follower_left.arm.end_effector_pose.dim_1",
        "observation.follower_left.arm.end_effector_pose.dim_2",
        "observation.follower_left.arm.end_effector_pose.dim_3",
        "observation.follower_left.arm.end_effector_pose.dim_4",
        "observation.follower_left.arm.end_effector_pose.dim_5",
        "observation.follower_left.arm.end_effector_pose.dim_6",
    ]


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


def test_batch_segment_count_difference_requires_explicit_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    _write_test_mcap(source)
    state = ApplicationState(source)
    try:
        state.expected_segments = 5
        state.preview_job = {"state": "completed"}
        state.preview_topic = "/camera/front"
        state.preview_frames = [
            {"index": 0, "timestamp_ns": 100},
            {"index": 1, "timestamp_ns": 200},
            {"index": 2, "timestamp_ns": 300},
        ]

        with pytest.raises(ValueError, match="normally uses 5 segments"):
            state.start_split(
                {
                    "item_index": 0,
                    "topic": "/camera/front",
                    "breakpoints_ns": ["200", "300"],
                    "output_paths": [
                        str(tmp_path / "part-1.mcap"),
                        str(tmp_path / "part-2.mcap"),
                        str(tmp_path / "part-3.mcap"),
                    ],
                }
            )

        state.item_job = {"state": "completed", "progress": 100, "message": "test"}
        assert state.session_status()["expected_segments"] == 5
    finally:
        state.close()


def test_preferred_camera_preloads_and_rolls_three_future_items(tmp_path: Path) -> None:
    source = tmp_path / "batch"
    for index in range(5):
        _write_camera_mcap(source / f"episode_{index:03d}.mcap")
    state = ApplicationState(source)

    def wait_until(predicate: Callable[[], bool], timeout_s: float = 5) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("Timed out waiting for background preload")

    try:
        state._scan_current_item()
        state.start_preview({"item_index": 0, "topic": "/camera/front"})
        wait_until(lambda: state.preview_status()["state"] == "completed")
        wait_until(lambda: state.preload_status["state"] == "completed")

        assert state.preferred_camera_topic == "/camera/front"
        assert set(state.preload_cache) == {1, 2, 3}
        assert all(
            cached.preview_topic == "/camera/front" and cached.preview_frames
            for cached in state.preload_cache.values()
        )

        with state.lock:
            state._advance_locked()

        assert state.current_index == 1
        assert state.item_job["state"] == "completed"
        assert state.preview_job["state"] == "completed"
        assert state.preview_topic == "/camera/front"
        cached_preview_dir = state.preview_dir
        assert state.start_preview({"item_index": 1, "topic": "/camera/front"})["state"] == "completed"
        assert state.preview_dir == cached_preview_dir
        wait_until(lambda: state.preload_status["state"] == "completed")
        assert set(state.preload_cache) == {2, 3, 4}
    finally:
        state.close()


@pytest.mark.parametrize("connection_error", [BrokenPipeError(), ConnectionResetError()])
def test_image_response_ignores_client_disconnects(connection_error: OSError) -> None:
    class DisconnectedWriter:
        def write(self, data: bytes) -> None:
            raise connection_error

    handler = object.__new__(McapSplitRequestHandler)
    handler.wfile = DisconnectedWriter()
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None

    handler._handle_image(lambda: b"jpeg")


def test_existing_outputs_return_a_structured_conflict_response(tmp_path: Path) -> None:
    existing = tmp_path / "existing.mcap"
    error = ExistingOutputsError([existing])
    responses = []

    def raise_conflict() -> None:
        raise error

    handler = object.__new__(McapSplitRequestHandler)
    handler._send_json = lambda payload, status=200: responses.append((payload, status))

    handler._handle_json(raise_conflict)

    payload, status = responses[0]
    assert status == 409
    assert payload == {
        "error": f"Output file already exists: {existing}",
        "code": "outputs_exist",
        "paths": [str(existing)],
    }
