from __future__ import annotations

from dataclasses import dataclass

from lerobot.cameras import CameraConfig


@CameraConfig.register_subclass("mujoco")
@dataclass
class MujocoCameraConfig(CameraConfig):
    """Camera that renders from a named MuJoCo `<camera>` in the MJCF model."""

    camera_name: str


__all__ = ["MujocoCameraConfig"]

