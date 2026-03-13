from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np


@dataclass(slots=True)
class IKResult:
    joint_positions_rad: list[float]
    success: bool
    position_error_m: float
    orientation_error_rad: float
    iterations: int


class MujocoInterface:
    def __init__(
        self,
        xml_path: Path,
        joint_names: tuple[str, ...],
        end_effector_site: str,
        target_mocap_body: str,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.joint_names = joint_names
        self.end_effector_site = end_effector_site
        self.target_mocap_body = target_mocap_body

        self._site_id = self.model.site(end_effector_site).id
        self._mocap_body_id = self.model.body(target_mocap_body).id
        self._mocap_id = int(np.asarray(self.model.body(target_mocap_body).mocapid).reshape(-1)[0])
        if self._mocap_id < 0:
            raise ValueError(f"Body '{target_mocap_body}' is not a mocap body")

        self._joint_actuator_ids: list[int] = []
        self._joint_qpos_adrs: list[int] = []
        self._name_to_index: dict[str, int] = {}
        self._joint_limits: list[tuple[float, float]] = []

        for joint_index, joint_name in enumerate(joint_names):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{joint_name}' not found in model")
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                raise ValueError(f"Actuator '{joint_name}' has invalid joint id")
            qpos_adr = int(self.model.jnt_qposadr[joint_id])

            if bool(self.model.jnt_limited[joint_id]):
                lower = float(self.model.jnt_range[joint_id, 0])
                upper = float(self.model.jnt_range[joint_id, 1])
            else:
                lower, upper = -np.inf, np.inf

            self._joint_actuator_ids.append(actuator_id)
            self._joint_qpos_adrs.append(qpos_adr)
            self._name_to_index[joint_name] = joint_index
            self._joint_limits.append((lower, upper))

        self._mocap_box_id = -1
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "arm_target_box")
        if geom_id >= 0:
            self._mocap_box_id = int(geom_id)

        self._ok_rgba = np.array([0.0, 1.0, 0.0, 0.1], dtype=np.float32)
        self._fail_rgba = np.array([1.0, 0.0, 0.0, 0.1], dtype=np.float32)

        self._ik_configuration = mink.Configuration(self.model)
        self._ik_frame_task = mink.FrameTask(
            frame_name=end_effector_site,
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=10.0,
        )
        self._ik_posture_task = mink.PostureTask(model=self.model, cost=1e-1)
        self._ik_tasks = [self._ik_frame_task, self._ik_posture_task]
        self._ik_limits = [mink.ConfigurationLimit(self.model)]

    def reset(self) -> None:
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._ik_configuration.update()
        self._ik_posture_task.set_target_from_configuration(self._ik_configuration)
        self.set_mocap_color(True)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def get_joint_positions(self) -> list[float]:
        return [float(self.data.qpos[qpos_adr]) for qpos_adr in self._joint_qpos_adrs]

    def set_joint_positions(self, joint_positions_rad: list[float]) -> None:
        if len(joint_positions_rad) != len(self._joint_actuator_ids):
            raise ValueError("joint_positions_rad length mismatch")
        for index, value in enumerate(joint_positions_rad):
            lower, upper = self._joint_limits[index]
            clamped = float(np.clip(float(value), lower, upper))
            self.data.ctrl[self._joint_actuator_ids[index]] = clamped

    def get_joint_limits(self) -> list[tuple[float, float]]:
        return list(self._joint_limits)

    def get_mocap_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(self.data.mocap_pos[self._mocap_id], dtype=np.float64).copy()
        quaternion = np.asarray(self.data.mocap_quat[self._mocap_id], dtype=np.float64).copy()
        return position, quaternion

    def set_mocap_pose(self, position_xyz: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        self.data.mocap_pos[self._mocap_id] = np.asarray(position_xyz, dtype=np.float64).reshape(3)
        self.data.mocap_quat[self._mocap_id] = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)

    def set_mocap_color(self, success: bool) -> None:
        if self._mocap_box_id < 0:
            return
        self.model.geom_rgba[self._mocap_box_id] = self._ok_rgba if success else self._fail_rgba

    def get_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        endpoint_pos = np.asarray(self.data.site_xpos[self._site_id], dtype=np.float64).copy()
        endpoint_xmat = np.asarray(self.data.site_xmat[self._site_id], dtype=np.float64).copy()
        endpoint_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(endpoint_quat, endpoint_xmat)
        endpoint_quat /= max(float(np.linalg.norm(endpoint_quat)), 1e-12)
        return endpoint_pos, endpoint_quat

    def align_mocap_to_end_effector(self) -> None:
        pos, quat = self.get_end_effector_pose()
        self.set_mocap_pose(pos, quat)
        self.forward()

    def solve_ik(
        self,
        target_position_xyz: np.ndarray,
        target_quaternion_wxyz: np.ndarray,
        current_joint_positions_rad: list[float],
        dt: float,
        max_iters: int,
        internal_dt: float,
        velocity_stuck_threshold: float,
        solver: str,
        success_position_tolerance: float,
        success_orientation_tolerance_rad: float,
    ) -> IKResult:
        self.set_joint_positions(current_joint_positions_rad)
        self.forward()

        self.set_mocap_pose(target_position_xyz, target_quaternion_wxyz)
        self.forward()

        self._ik_configuration.update()
        self._ik_posture_task.set_target_from_configuration(self._ik_configuration)
        self._ik_frame_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, self.target_mocap_body))

        solved_without_exception = True
        remaining = float(dt)
        iterations = 0

        while remaining > 0.0 and iterations < int(max_iters):
            sub_dt = min(float(internal_dt), remaining)
            try:
                velocity = mink.solve_ik(
                    configuration=self._ik_configuration,
                    tasks=self._ik_tasks,
                    dt=sub_dt,
                    solver=solver,
                    limits=self._ik_limits,
                )
            except Exception:
                solved_without_exception = False
                break

            self._ik_configuration.integrate_inplace(velocity, sub_dt)
            remaining -= sub_dt
            iterations += 1

            if float(np.linalg.norm(velocity)) < float(velocity_stuck_threshold):
                break

        solved_joint_positions = [
            float(self._ik_configuration.q[qpos_adr]) for qpos_adr in self._joint_qpos_adrs
        ]
        self.set_joint_positions(solved_joint_positions)
        self.step()

        position_error_m, orientation_error_rad = self._compute_tracking_errors()
        success = (
            solved_without_exception
            and position_error_m <= float(success_position_tolerance)
            and orientation_error_rad <= float(success_orientation_tolerance_rad)
        )
        self.set_mocap_color(success)

        return IKResult(
            joint_positions_rad=solved_joint_positions,
            success=success,
            position_error_m=position_error_m,
            orientation_error_rad=orientation_error_rad,
            iterations=iterations,
        )

    def _compute_tracking_errors(self) -> tuple[float, float]:
        endpoint_pos, endpoint_quat = self.get_end_effector_pose()
        target_pos = np.asarray(self.data.mocap_pos[self._mocap_id], dtype=np.float64).reshape(3)
        target_quat = np.asarray(self.data.mocap_quat[self._mocap_id], dtype=np.float64).reshape(4)

        target_quat /= max(float(np.linalg.norm(target_quat)), 1e-12)
        dot = float(np.clip(np.abs(np.dot(endpoint_quat, target_quat)), 0.0, 1.0))
        orientation_error = float(2.0 * np.arccos(dot))
        position_error = float(np.linalg.norm(target_pos - endpoint_pos))
        return position_error, orientation_error


__all__ = ["IKResult", "MujocoInterface"]
