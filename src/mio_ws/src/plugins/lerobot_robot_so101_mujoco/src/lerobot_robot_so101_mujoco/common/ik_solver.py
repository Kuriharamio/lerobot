from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .mujoco_interface import IKResult, MujocoInterface


@dataclass(slots=True)
class IKTargetPose:
    position_xyz: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass(slots=True)
class IKSolverConfig:
    control_dt: float = 0.02
    solver: str = "quadprog"
    success_position_tolerance: float = 5e-3
    success_orientation_tolerance_rad: float = 2e-1
    ik_max_iters: int = 50
    ik_internal_dt: float = 2e-3
    ik_velocity_stuck_threshold: float = 1e-8


class IKSolver:
    def __init__(self, interface: MujocoInterface, config: IKSolverConfig):
        self.interface = interface
        self.config = config

    def solve(self, target_pose: IKTargetPose, current_joint_positions_rad: list[float]) -> IKResult:
        return self.interface.solve_ik(
            target_position_xyz=np.asarray(target_pose.position_xyz, dtype=np.float64).reshape(3),
            target_quaternion_wxyz=np.asarray(target_pose.quaternion_wxyz, dtype=np.float64).reshape(4),
            current_joint_positions_rad=current_joint_positions_rad,
            dt=float(self.config.control_dt),
            max_iters=int(self.config.ik_max_iters),
            internal_dt=float(self.config.ik_internal_dt),
            velocity_stuck_threshold=float(self.config.ik_velocity_stuck_threshold),
            solver=self.config.solver,
            success_position_tolerance=float(self.config.success_position_tolerance),
            success_orientation_tolerance_rad=float(self.config.success_orientation_tolerance_rad),
        )


__all__ = ["IKSolver", "IKSolverConfig", "IKTargetPose"]
