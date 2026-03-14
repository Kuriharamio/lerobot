from __future__ import annotations

import logging
import math
from xmlrpc.client import ServerProxy

from lerobot.processor import RobotAction
from lerobot.teleoperators import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_so100_sim2real import SO100Sim2RealConfig

LOGGER = logging.getLogger(__name__)

SO100_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO100_BODY_JOINT_NAMES: tuple[str, ...] = SO100_JOINT_NAMES[:-1]


class SO100Sim2Real(Teleoperator):
    config_class = SO100Sim2RealConfig
    name = "so100_sim2real"

    def __init__(self, config: SO100Sim2RealConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._rpc: ServerProxy | None = None
        self._body_joint_zero_offsets = self._build_body_joint_zero_offsets()
        self._last_action: dict[str, float] = {
            f"{joint}.pos": 0.0 for joint in SO100_JOINT_NAMES
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in SO100_JOINT_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        rpc_url = f"http://{self.config.host}:{self.config.port}"
        self._rpc = ServerProxy(rpc_url, allow_none=True)
        if not bool(self._rpc.ping()):
            raise ConnectionError(f"SO100 sim2real teleop RPC ping failed: {rpc_url}")

        self._connected = True
        LOGGER.info("SO100 sim2real teleop connected to %s", rpc_url)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        assert self._rpc is not None
        payload = self._rpc.get_joint_state()
        if not isinstance(payload, dict):
            return dict(self._last_action)

        action: dict[str, float] = {}
        for joint in SO100_BODY_JOINT_NAMES:
            raw_value = float(payload.get(joint, 0.0))
            value = math.degrees(raw_value) if self.config.use_degrees else raw_value
            value = value - self._body_joint_zero_offsets[joint]
            action[f"{joint}.pos"] = float(value)

        gripper_raw = float(payload.get("gripper", 0.0))
        action["gripper.pos"] = self._map_gripper_to_percent(gripper_raw)
        self._last_action = dict(action)
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        del feedback
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        self._rpc = None
        self._connected = False
        LOGGER.info("SO100 sim2real teleop disconnected")

    def _map_gripper_to_percent(self, gripper_value: float) -> float:
        minimum = float(self.config.gripper_joint_min)
        maximum = float(self.config.gripper_joint_max)
        if maximum <= minimum:
            return float(max(0.0, min(100.0, gripper_value)))

        ratio = (gripper_value - minimum) / (maximum - minimum)
        ratio = max(0.0, min(1.0, ratio))
        return float(ratio * 100.0)

    def _build_body_joint_zero_offsets(self) -> dict[str, float]:
        configured = [float(value) for value in self.config.sim_zero_joint_angles]
        expected_len = len(SO100_BODY_JOINT_NAMES)

        if len(configured) != expected_len:
            LOGGER.warning(
                "SO100 sim2real expected %s zero-joint angles but got %s; fallback to all-zero offsets.",
                expected_len,
                len(configured),
            )
            configured = [0.0] * expected_len

        return {
            joint: configured[index]
            for index, joint in enumerate(SO100_BODY_JOINT_NAMES)
        }


__all__ = ["SO100Sim2Real"]
