from __future__ import annotations

import logging
import os
import signal
import threading
import time
from functools import cached_property

import mujoco.viewer
import numpy as np

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot

from .common.math_utils import (
    apply_rpy_delta,
)
from .common.ik_solver import (
    IKSolver,
    IKSolverConfig,
)
from .common.mujoco_utils import (
    MujocoContext,
    align_mocap_to_end_effector,
    get_joint_limits,
    get_joint_positions,
    get_mocap_pose,
    make_mujoco_context,
    reset,
    set_mocap_color,
    set_joint_positions,
    set_mocap_pose,
    step,
    forward,
)
from .common.rpc_server import JointStateRPCServer
from .config_so100_mujoco import (
    SO100_GRIPPER_NAME,
    SO100_JOINT_NAMES,
    SO100MujocoRobotConfig,
    make_ee_delta_action_features,
    make_joint_observation,
)


LOGGER = logging.getLogger("so100_mujoco_robot")


class SO100MujocoRobot(Robot):
    config_class = SO100MujocoRobotConfig
    name = "so100_mujoco"

    def __init__(self, config: SO100MujocoRobotConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._mujoco_context: MujocoContext | None = None
        self._last_sim_time: float = 0.0
        self._ik_solver: IKSolver | None = None
        self._target_position_xyz: np.ndarray | None = None
        self._target_quaternion_wxyz: np.ndarray | None = None
        self._last_joint_state_rad: list[float] = []
        self._latest_delta_action: dict[str, float] = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_rx": 0.0,
            "delta_ry": 0.0,
            "delta_rz": 0.0,
            "gripper": 1.0,
        }
        self._lock = threading.Lock()
        self._sim_thread: threading.Thread | None = None
        self._publisher_thread: threading.Thread | None = None
        self._rpc_server: JointStateRPCServer | None = None
        self._stop_event = threading.Event()

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in SO100_JOINT_NAMES}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return make_ee_delta_action_features()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._connected:
            return

        LOGGER.info("Initializing internal MuJoCo simulation for SO100 robot")
        self._mujoco_context = make_mujoco_context(
            xml_path=self.config.xml_path,
        )
        self._ik_solver = IKSolver(
            mujoco_context=self._mujoco_context,
            end_effector_site=self.config.end_effector_site,
            target_mocap_body=self.config.target_mocap_body,
            config=IKSolverConfig(
                control_dt=self.config.control_dt,
                solver=self.config.ik_solver,
                success_position_tolerance=self.config.success_position_tolerance,
                success_orientation_tolerance_rad=self.config.success_orientation_tolerance_rad,
                ik_max_iters=self.config.ik_max_iters,
                ik_internal_dt=self.config.ik_internal_dt,
                ik_velocity_stuck_threshold=self.config.ik_velocity_stuck_threshold,
            ),
        )

        reset(self._mujoco_context)
        align_mocap_to_end_effector(self._mujoco_context, self._ik_solver.mocap_id, self._ik_solver.site_id)
        self._ik_solver.reset()
        self._target_position_xyz, self._target_quaternion_wxyz = get_mocap_pose(
            self._mujoco_context,
            self._ik_solver.mocap_id,
        )
        with self._lock:
            self._last_joint_state_rad = get_joint_positions(self._mujoco_context)

        self._stop_event.clear()
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

        if self.config.rpc_enabled:
            self._rpc_server = JointStateRPCServer(host=self.config.rpc_host, port=self.config.rpc_port)
            self._rpc_server.start()
            self._publisher_thread = threading.Thread(target=self._publish_joint_state_loop, daemon=True)
            self._publisher_thread.start()
            LOGGER.info(
                "SO100 MuJoCo joint-state RPC server started at %s:%s",
                self.config.rpc_host,
                self.config.rpc_port,
            )

        self._connected = True
        LOGGER.info("SO100 MuJoCo internal simulation started")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def get_observation(self) -> RobotObservation:
        if not self._connected or self._mujoco_context is None:
            raise ConnectionError(f"{self} is not connected.")

        with self._lock:
            joint_state = list(self._last_joint_state_rad)
        if len(joint_state) != len(SO100_JOINT_NAMES):
            raise RuntimeError("Invalid joint state length from internal simulation backend")

        joint_map = {
            joint_name: float(joint_state[index]) for index, joint_name in enumerate(SO100_JOINT_NAMES)
        }
        return make_joint_observation(joint_map)

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected or self._mujoco_context is None:
            raise ConnectionError(f"{self} is not connected.")

        adapted_action = self._adapt_action(action)
        with self._lock:
            self._latest_delta_action = dict(adapted_action)
        return adapted_action

    def disconnect(self) -> None:
        if not self._connected:
            return

        self._connected = False
        self._stop_event.set()
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=1.0)
            self._sim_thread = None
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=1.0)
            self._publisher_thread = None
        if self._rpc_server is not None:
            self._rpc_server.stop()
            self._rpc_server = None
        self._mujoco_context = None
        self._ik_solver = None

    def _publish_joint_state_loop(self) -> None:
        publish_period_s = 1.0 / max(float(self.config.rpc_publish_hz), 1e-3)
        while not self._stop_event.is_set():
            if self._rpc_server is not None:
                with self._lock:
                    joint_state = list(self._last_joint_state_rad)
                if len(joint_state) == len(SO100_JOINT_NAMES):
                    self._rpc_server.update_joint_state(
                        {
                            joint_name: float(joint_state[index])
                            for index, joint_name in enumerate(SO100_JOINT_NAMES)
                        }
                    )
            time.sleep(publish_period_s)

    def _simulation_loop(self) -> None:
        with mujoco.viewer.launch_passive(
            self._mujoco_context.model,
            self._mujoco_context.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            while not self._stop_event.is_set() and viewer.is_running():
                with viewer.lock():
                    self._step_once()
                viewer.sync()
        if not self._stop_event.is_set():
            self._stop_event.set()
            try:
                os.killpg(os.getpgrp(), signal.SIGINT)
            except Exception:
                try:
                    os.kill(os.getpid(), signal.SIGINT)
                except Exception:
                    pass

    def _consume_delta_action(self) -> dict[str, float]:
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

    def _step_once(self) -> None:
        current_time = float(self._mujoco_context.data.time)
        if current_time < self._last_sim_time or (current_time == 0.0 and self._last_sim_time > 0.0):
            LOGGER.info("MuJoCo simulation reset.")
            align_mocap_to_end_effector(
                self._mujoco_context,
                self._ik_solver.mocap_id,
                self._ik_solver.site_id,
            )
            self._ik_solver.reset()
            self._target_position_xyz, self._target_quaternion_wxyz = get_mocap_pose(
                self._mujoco_context,
                self._ik_solver.mocap_id,
            )
            with self._lock:
                self._last_joint_state_rad = get_joint_positions(self._mujoco_context)

        self._last_sim_time = current_time

        action = self._consume_delta_action()
        delta_xyz = np.array(
            [
                float(action.get("delta_x", 0.0)),
                float(action.get("delta_y", 0.0)),
                float(action.get("delta_z", 0.0)),
            ],
            dtype=np.float64,
        ) * float(self.config.delta_position_scale_m)
        delta_rpy = np.array(
            [
                float(action.get("delta_rx", 0.0)),
                float(action.get("delta_ry", 0.0)),
                float(action.get("delta_rz", 0.0)),
            ],
            dtype=np.float64,
        ) * float(self.config.delta_rotation_scale_rad)

        base_position_xyz, base_quaternion_wxyz = get_mocap_pose(self._mujoco_context, self._ik_solver.mocap_id)
        self._target_position_xyz = base_position_xyz + delta_xyz
        self._target_quaternion_wxyz = apply_rpy_delta(base_quaternion_wxyz, delta_rpy)

        has_delta = (float(np.linalg.norm(delta_xyz)) > 0.0) or (float(np.linalg.norm(delta_rpy)) > 0.0)
        if has_delta:
            set_mocap_pose(self._mujoco_context, self._ik_solver.mocap_id, self._target_position_xyz, self._target_quaternion_wxyz)
            forward(self._mujoco_context)

        next_joint_state, ik_success = self._ik_solver.solve(
            current_joint_positions_rad=self._last_joint_state_rad,
        )
        set_mocap_color(self._mujoco_context, self._ik_solver.mocap_box_id, ik_success)
        next_joint_state = list(next_joint_state)

        gripper_index = SO100_JOINT_NAMES.index(SO100_GRIPPER_NAME)
        gripper_mode = float(action.get("gripper", 1.0))
        if gripper_mode < 0.5:
            next_joint_state[gripper_index] -= float(self.config.gripper_step_rad)
        elif gripper_mode > 1.5:
            next_joint_state[gripper_index] += float(self.config.gripper_step_rad)

        joint_limits = get_joint_limits(self._mujoco_context)
        for index, (lower, upper) in enumerate(joint_limits):
            next_joint_state[index] = float(np.clip(next_joint_state[index], lower, upper))

        set_joint_positions(self._mujoco_context, next_joint_state)
        step(self._mujoco_context)

        with self._lock:
            self._last_joint_state_rad = get_joint_positions(self._mujoco_context)

    def _adapt_action(self, action: RobotAction) -> dict[str, float]:
        numeric_action: dict[str, float] = {}
        for key, value in action.items():
            try:
                numeric_action[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return {
            "delta_x": -float(numeric_action.get("delta_y", 0.0)),
            "delta_y": float(numeric_action.get("delta_x", 0.0)),
            "delta_z": float(numeric_action.get("delta_z", 0.0)),
            "delta_rx": float(numeric_action.get("delta_rx", 0.0)),
            "delta_ry": float(numeric_action.get("delta_ry", 0.0)),
            "delta_rz": float(numeric_action.get("delta_rz", 0.0)),
            "gripper": float(numeric_action.get("gripper", 1.0)),
        }


__all__ = ["SO100MujocoRobot"]
