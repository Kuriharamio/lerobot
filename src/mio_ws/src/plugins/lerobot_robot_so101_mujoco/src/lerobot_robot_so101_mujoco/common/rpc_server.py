from __future__ import annotations

import json
import socketserver
import threading
from typing import Any


class JointRPCState:
    def __init__(self, joint_dim: int):
        self._lock = threading.Lock()
        self._joint_dim = int(joint_dim)
        self._joint_state = [0.0] * self._joint_dim
        self._latest_delta_action = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_rx": 0.0,
            "delta_ry": 0.0,
            "delta_rz": 0.0,
            "gripper": 1.0,
        }

    def set_delta_action(self, action: dict[str, float]) -> None:
        with self._lock:
            self._latest_delta_action = {
                "delta_x": float(action.get("delta_x", 0.0)),
                "delta_y": float(action.get("delta_y", 0.0)),
                "delta_z": float(action.get("delta_z", 0.0)),
                "delta_rx": float(action.get("delta_rx", 0.0)),
                "delta_ry": float(action.get("delta_ry", 0.0)),
                "delta_rz": float(action.get("delta_rz", 0.0)),
                "gripper": float(action.get("gripper", 1.0)),
            }

    def consume_delta_action(self) -> dict[str, float]:
        with self._lock:
            action = dict(self._latest_delta_action)
            self._latest_delta_action = {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "delta_rx": 0.0,
                "delta_ry": 0.0,
                "delta_rz": 0.0,
                "gripper": 1.0,
            }
            return action

    def set_joint_state(self, joint_state: list[float]) -> None:
        if len(joint_state) != self._joint_dim:
            return
        with self._lock:
            self._joint_state = [float(v) for v in joint_state]

    def get_joint_state(self) -> list[float]:
        with self._lock:
            return list(self._joint_state)


class JointRPCRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: JointRPCState = self.server.state

        raw = self.rfile.readline().decode("utf-8").strip()
        if not raw:
            return

        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            self._write_json({"ok": False, "error": "invalid_json"})
            return

        command = request.get("cmd")

        if command == "ping":
            self._write_json({"ok": True})
            return

        if command == "get_joint_state":
            self._write_json({"ok": True, "joint_state_rad": state.get_joint_state()})
            return

        if command == "send_action":
            action = request.get("action")
            if not isinstance(action, dict):
                self._write_json({"ok": False, "error": "action_not_dict"})
                return
            try:
                casted = {str(k): float(v) for k, v in action.items()}
            except (TypeError, ValueError):
                self._write_json({"ok": False, "error": "action_invalid"})
                return

            state.set_delta_action(casted)
            self._write_json({"ok": True, "action": casted})
            return

        if command == "reset":
            self.server.reset_requested.set()
            self._write_json({"ok": True})
            return

        self._write_json({"ok": False, "error": f"unknown_cmd:{command}"})

    def _write_json(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))


class JointRPCServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type,
        state: JointRPCState,
        reset_requested: threading.Event,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.state = state
        self.reset_requested = reset_requested


__all__ = ["JointRPCRequestHandler", "JointRPCServer", "JointRPCState"]
