from __future__ import annotations

import threading

import numpy as np

from .ik_solver import IKSolver, IKSolverConfig, IKTargetPose
from .math_utils import apply_rpy_delta
from .mujoco_interface import MujocoInterface
from .rpc_server import JointRPCRequestHandler, JointRPCServer, JointRPCState
from ..config_so101_mujoco import SO101_GRIPPER_NAME, SO101_JOINT_NAMES


class SO101MujocoBackend:
    def __init__(
        self,
        interface: MujocoInterface,
        host: str,
        port: int,
        control_dt: float,
        delta_position_scale_m: float,
        delta_rotation_scale_rad: float,
        gripper_step_rad: float,
        ik_solver: str,
        ik_max_iters: int,
        ik_internal_dt: float,
        ik_velocity_stuck_threshold: float,
        success_position_tolerance: float,
        success_orientation_tolerance_rad: float,
    ) -> None:
        self.interface = interface
        self.control_dt = float(control_dt)
        self.delta_position_scale_m = float(delta_position_scale_m)
        self.delta_rotation_scale_rad = float(delta_rotation_scale_rad)
        self.gripper_step_rad = float(gripper_step_rad)

        self._target_position_xyz, self._target_quaternion_wxyz = self.interface.get_mocap_pose()
        self._last_joint_state_rad = self.interface.get_joint_positions()

        self._rpc_state = JointRPCState(joint_dim=len(SO101_JOINT_NAMES))
        self._rpc_state.set_joint_state(self._last_joint_state_rad)
        self._reset_requested = threading.Event()
        self._rpc_server = JointRPCServer(
            (host, int(port)),
            JointRPCRequestHandler,
            state=self._rpc_state,
            reset_requested=self._reset_requested,
        )
        self._rpc_thread = threading.Thread(target=self._rpc_server.serve_forever, daemon=True)

        self._ik_solver = IKSolver(
            interface=self.interface,
            config=IKSolverConfig(
                control_dt=self.control_dt,
                solver=ik_solver,
                success_position_tolerance=success_position_tolerance,
                success_orientation_tolerance_rad=success_orientation_tolerance_rad,
                ik_max_iters=ik_max_iters,
                ik_internal_dt=ik_internal_dt,
                ik_velocity_stuck_threshold=ik_velocity_stuck_threshold,
            ),
        )

    @property
    def model(self):
        return self.interface.model

    @property
    def data(self):
        return self.interface.data

    def connect(self) -> None:
        self.interface.reset()
        self.interface.align_mocap_to_end_effector()
        self._target_position_xyz, self._target_quaternion_wxyz = self.interface.get_mocap_pose()
        self._last_joint_state_rad = self.interface.get_joint_positions()
        self._rpc_state.set_joint_state(self._last_joint_state_rad)
        self._rpc_thread.start()

    def disconnect(self) -> None:
        self._rpc_server.shutdown()
        self._rpc_server.server_close()

    def step(self) -> None:
        if self._reset_requested.is_set():
            self.interface.reset()
            self.interface.align_mocap_to_end_effector()
            self._target_position_xyz, self._target_quaternion_wxyz = self.interface.get_mocap_pose()
            self._last_joint_state_rad = self.interface.get_joint_positions()
            self._rpc_state.set_joint_state(self._last_joint_state_rad)
            self._reset_requested.clear()
            return

        mocap_position, mocap_quaternion = self.interface.get_mocap_pose()
        self._target_position_xyz = np.asarray(mocap_position, dtype=np.float64)
        self._target_quaternion_wxyz = np.asarray(mocap_quaternion, dtype=np.float64)

        action = self._rpc_state.consume_delta_action()

        delta_xyz = np.array(
            [
                float(action.get("delta_x", 0.0)),
                float(action.get("delta_y", 0.0)),
                float(action.get("delta_z", 0.0)),
            ],
            dtype=np.float64,
        ) * self.delta_position_scale_m
        delta_rpy = np.array(
            [
                float(action.get("delta_rx", 0.0)),
                float(action.get("delta_ry", 0.0)),
                float(action.get("delta_rz", 0.0)),
            ],
            dtype=np.float64,
        ) * self.delta_rotation_scale_rad

        self._target_position_xyz = self._target_position_xyz + delta_xyz
        self._target_quaternion_wxyz = apply_rpy_delta(self._target_quaternion_wxyz, delta_rpy)

        ik_result = self._ik_solver.solve(
            target_pose=IKTargetPose(
                position_xyz=self._target_position_xyz,
                quaternion_wxyz=self._target_quaternion_wxyz,
            ),
            current_joint_positions_rad=self._last_joint_state_rad,
        )

        next_joint_state = list(ik_result.joint_positions_rad)
        gripper_index = SO101_JOINT_NAMES.index(SO101_GRIPPER_NAME)
        gripper_mode = float(action.get("gripper", 1.0))
        if gripper_mode < 0.5:
            next_joint_state[gripper_index] -= self.gripper_step_rad
        elif gripper_mode > 1.5:
            next_joint_state[gripper_index] += self.gripper_step_rad

        joint_limits = self.interface.get_joint_limits()
        for index, (lower, upper) in enumerate(joint_limits):
            next_joint_state[index] = float(np.clip(next_joint_state[index], lower, upper))

        self.interface.set_joint_positions(next_joint_state)
        self.interface.step()

        self._last_joint_state_rad = self.interface.get_joint_positions()
        self._rpc_state.set_joint_state(self._last_joint_state_rad)


SO101MujocoSimRobot = SO101MujocoBackend

__all__ = ["SO101MujocoBackend", "SO101MujocoSimRobot"]
