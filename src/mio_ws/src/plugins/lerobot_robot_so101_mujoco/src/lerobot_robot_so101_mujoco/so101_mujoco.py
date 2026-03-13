from __future__ import annotations

import json
import logging
import socket
from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot

from .config_so101_mujoco import (
    SO101_JOINT_NAMES,
    SO101MujocoRobotConfig,
    make_ee_delta_action_features,
    make_joint_observation,
)


LOGGER = logging.getLogger("so101_mujoco_robot")


class SO101MujocoRobot(Robot):
    config_class = SO101MujocoRobotConfig
    name = "so101_mujoco"

    def __init__(self, config: SO101MujocoRobotConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in SO101_JOINT_NAMES}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return make_ee_delta_action_features()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        LOGGER.info(
            "Connecting to simulation server %s:%d",
            self.config.remote_host,
            self.config.remote_port,
        )
        self._rpc({"cmd": "ping"})
        self._connected = True
        LOGGER.info("Connected to simulation server")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected.")

        response = self._rpc({"cmd": "get_joint_state"})
        joint_state = response.get("joint_state_rad")
        if not isinstance(joint_state, list) or len(joint_state) != len(SO101_JOINT_NAMES):
            raise RuntimeError("Invalid joint_state_rad payload from simulation server")

        joint_map = {
            joint_name: float(joint_state[index]) for index, joint_name in enumerate(SO101_JOINT_NAMES)
        }
        return make_joint_observation(joint_map)

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected.")

        adapted_action = self._adapt_action(action)
        response = self._rpc({"cmd": "send_action", "action": adapted_action})
        returned_action = response.get("action")
        if isinstance(returned_action, dict):
            return {str(k): float(v) for k, v in returned_action.items()}
        return adapted_action

    def disconnect(self) -> None:
        self._connected = False

    def _coerce_numeric_action(self, action: RobotAction) -> dict[str, float]:
        numeric_action: dict[str, float] = {}
        for key, value in action.items():
            try:
                numeric_action[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return numeric_action

    def _adapt_action(self, action: RobotAction) -> dict[str, float]:
        numeric_action = self._coerce_numeric_action(action)
        return {
            "delta_x": -float(numeric_action.get("delta_y", 0.0)),
            "delta_y": float(numeric_action.get("delta_x", 0.0)),
            "delta_z": float(numeric_action.get("delta_z", 0.0)),
            "delta_rx": float(numeric_action.get("delta_rx", 0.0)),
            "delta_ry": float(numeric_action.get("delta_ry", 0.0)),
            "delta_rz": float(numeric_action.get("delta_rz", 0.0)),
            "gripper": float(numeric_action.get("gripper", 1.0)),
        }

    def _rpc(self, request: dict[str, object]) -> dict[str, object]:
        try:
            with socket.create_connection(
                (self.config.remote_host, int(self.config.remote_port)),
                timeout=float(self.config.remote_timeout_s),
            ) as conn:
                payload = (json.dumps(request) + "\n").encode("utf-8")
                conn.sendall(payload)

                buffer = bytearray()
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    if b"\n" in chunk:
                        break
        except OSError as exc:
            LOGGER.error(
                "RPC failed cmd=%s host=%s port=%d error=%s",
                request.get("cmd"),
                self.config.remote_host,
                self.config.remote_port,
                exc,
            )
            raise

        if not buffer:
            raise ConnectionError("Simulation server returned an empty response")

        line = buffer.split(b"\n", maxsplit=1)[0]
        response = json.loads(line.decode("utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("Invalid response from simulation server")
        if not bool(response.get("ok", False)):
            error = response.get("error", "unknown_error")
            raise RuntimeError(f"Simulation server error: {error}")
        return response


__all__ = ["SO101MujocoRobot"]
