from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np
import time
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera

from .config_mujoco_camera import MujocoCameraConfig


@dataclass(slots=True)
class _RendererState:
    renderer: mujoco.Renderer
    camera_id: int


class MujocoCamera(Camera):
    """Renders RGB frames from a MuJoCo camera in an existing simulation."""

    def __init__(
        self,
        config: MujocoCameraConfig,
        *,
        get_model_data: Callable[[], tuple[mujoco.MjModel, mujoco.MjData]],
        lock: Any | None = None,
    ):
        super().__init__(config)
        self.config = config
        self._get_model_data = get_model_data
        self._lock = lock
        self._state: _RendererState | None = None
        self._last_frame: NDArray[Any] | None = None
        self._last_frame_ts_s: float | None = None

    @property
    def is_connected(self) -> bool:
        return self._state is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # Cameras in MuJoCo are model-specific; discovery is handled by the robot that owns the model.
        return []

    def connect(self, warmup: bool = True) -> None:
        del warmup
        if self._state is not None:
            return

        model, _ = self._get_model_data()
        if self.width is None or self.height is None:
            raise ValueError("MujocoCamera requires width and height to be set in config.")

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name)
        if cam_id < 0:
            raise ValueError(
                f"MuJoCo camera '{self.config.camera_name}' not found in model. "
                "Make sure the MJCF contains a <camera name='...'> with this name."
            )

        renderer = mujoco.Renderer(model, height=int(self.height), width=int(self.width))
        self._state = _RendererState(renderer=renderer, camera_id=int(cam_id))

    def read(self) -> NDArray[Any]:
        if self._state is None:
            raise RuntimeError("Camera is not connected.")

        _, data = self._get_model_data()

        if self._lock is None:
            self._state.renderer.update_scene(data, camera=self._state.camera_id)
            frame = self._state.renderer.render()
        else:
            with self._lock:
                self._state.renderer.update_scene(data, camera=self._state.camera_id)
                frame = self._state.renderer.render()

        # mujoco.Renderer returns RGB uint8 image
        out = np.asarray(frame, dtype=np.uint8)
        self._last_frame = out
        self._last_frame_ts_s = time.time()
        return out

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        del timeout_ms
        # Rendering is synchronous; return a fresh frame each call.
        return self.read()

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        # For MuJoCo rendering we don't have a hardware-backed async buffer; instead we cache the
        # last rendered frame and re-render if the cached one is too old.
        if self._state is None:
            raise RuntimeError("Camera is not connected.")

        if self._last_frame is not None and self._last_frame_ts_s is not None:
            age_ms = (time.time() - self._last_frame_ts_s) * 1000.0
            if age_ms <= float(max_age_ms):
                return self._last_frame

        return self.read()

    def disconnect(self) -> None:
        if self._state is None:
            return
        try:
            self._state.renderer.close()
        except Exception:
            pass
        self._state = None
        self._last_frame = None
        self._last_frame_ts_s = None


__all__ = ["MujocoCamera"]

