from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from .mujoco_utils import MujocoContext


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
    def __init__(
        self,
        *,
        mujoco_context: MujocoContext,
        end_effector_site: str,
        target_mocap_body: str,
        config: IKSolverConfig,
    ):
        self.mujoco_context = mujoco_context
        self.config = config

        self.end_effector_site = str(end_effector_site)
        self.target_mocap_body = str(target_mocap_body)

        self.site_id = int(self.mujoco_context.model.site(self.end_effector_site).id)
        self.mocap_body_id = int(self.mujoco_context.model.body(self.target_mocap_body).id)
        self.mocap_id = int(np.asarray(self.mujoco_context.model.body(self.target_mocap_body).mocapid).reshape(-1)[0])
        if self.mocap_id < 0:
            raise ValueError(f"Body '{self.target_mocap_body}' is not a mocap body")

        geom_id = mujoco.mj_name2id(self.mujoco_context.model, mujoco.mjtObj.mjOBJ_GEOM, "arm_target_box")
        self.mocap_box_id = int(geom_id) if int(geom_id) >= 0 else -1

        self.configuration = mink.Configuration(self.mujoco_context.model)
        self.frame_task = mink.FrameTask(
            frame_name=self.end_effector_site,
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=10.0,
        )
        self.posture_task = mink.PostureTask(model=self.mujoco_context.model, cost=1e-1)
        self.tasks = [self.frame_task, self.posture_task]
        self.limits = [mink.ConfigurationLimit(self.mujoco_context.model)]

        self.reset()

    def reset(self) -> None:
        self.configuration.update()
        self.posture_task.set_target_from_configuration(self.configuration)

    def solve(self, current_joint_positions_rad: list[float]) -> tuple[list[float], bool]:

        q = self.configuration.q.copy()
        for i, qpos_adr in enumerate(self.mujoco_context.joint_qpos_adrs):
            q[int(qpos_adr)] = float(current_joint_positions_rad[i])
        self.configuration.update(q)
        self.posture_task.set_target_from_configuration(self.configuration)

        self.frame_task.set_target(
            mink.SE3.from_mocap_name(
                self.mujoco_context.model,
                self.mujoco_context.data,
                self.target_mocap_body,
            )
        )

        solved_without_exception = True
        remaining = float(self.config.control_dt)
        iterations = 0
        achieved = False 

        while remaining > 0.0 and iterations < int(self.config.ik_max_iters):
            sub_dt = min(float(self.config.ik_internal_dt), remaining)
            try:
                velocity = mink.solve_ik(
                    configuration=self.configuration,
                    tasks=self.tasks,
                    dt=sub_dt,
                    solver=self.config.solver,
                    limits=self.limits,
                )
            except Exception as exc:
                print(f"IK solve exception: {exc}")
                solved_without_exception = False
                break

            self.configuration.integrate_inplace(velocity, sub_dt)
            remaining -= sub_dt
            iterations += 1

            err = np.asarray(self.frame_task.compute_error(self.configuration), dtype=np.float64).reshape(-1)
            pos_ok = float(np.linalg.norm(err[:3])) <= float(self.config.success_position_tolerance)
            ori_ok = float(np.linalg.norm(err[3:])) <= float(self.config.success_orientation_tolerance_rad)
            achieved = bool(pos_ok and ori_ok)
            if achieved:
                break

            if float(np.linalg.norm(velocity)) < float(self.config.ik_velocity_stuck_threshold):
                break

        solved_joint_positions = [float(self.configuration.q[qpos_adr]) for qpos_adr in self.mujoco_context.joint_qpos_adrs]
        success = bool(solved_without_exception and achieved)

        return solved_joint_positions, success


__all__ = [
    "IKSolverConfig",
    "IKSolver",
]
