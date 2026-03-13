from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

@dataclass(slots=True)
class MujocoContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    joint_actuator_ids: list[int]
    joint_qpos_adrs: list[int]
    joint_limits: list[tuple[float, float]]


def make_mujoco_context(xml_path: Path,) -> MujocoContext:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    joint_actuator_ids: list[int] = []
    joint_qpos_adrs: list[int] = []
    joint_limits: list[tuple[float, float]] = []

    for actuator_id in range(int(model.nu)):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        jnt_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64).reshape(2)
        lower = float(jnt_range[0])
        upper = float(jnt_range[1])

        joint_actuator_ids.append(int(actuator_id))
        joint_qpos_adrs.append(int(qpos_adr))
        joint_limits.append((lower, upper))

    return MujocoContext(
        model=model,
        data=data,
        joint_actuator_ids=joint_actuator_ids,
        joint_qpos_adrs=joint_qpos_adrs,
        joint_limits=joint_limits,
    )


def reset(ctx: MujocoContext) -> None:
    ctx.data.qvel[:] = 0.0
    ctx.data.act[:] = 0.0
    mujoco.mj_forward(ctx.model, ctx.data)


def forward(ctx: MujocoContext) -> None:
    mujoco.mj_forward(ctx.model, ctx.data)


def step(ctx: MujocoContext) -> None:
    mujoco.mj_step(ctx.model, ctx.data)


def get_joint_positions(ctx: MujocoContext) -> list[float]:
    return [float(ctx.data.qpos[qpos_adr]) for qpos_adr in ctx.joint_qpos_adrs]


def set_joint_positions(ctx: MujocoContext, joint_positions_rad: list[float]) -> None:
    if len(joint_positions_rad) != len(ctx.joint_actuator_ids):
        raise ValueError("joint_positions_rad length mismatch")
    for index, value in enumerate(joint_positions_rad):
        lower, upper = ctx.joint_limits[index]
        clamped = float(np.clip(float(value), lower, upper))
        ctx.data.ctrl[ctx.joint_actuator_ids[index]] = clamped


def get_joint_limits(ctx: MujocoContext) -> list[tuple[float, float]]:
    return list(ctx.joint_limits)


def set_mocap_color(mj_ctx: MujocoContext, mocap_box_id: int, success: bool) -> None:
    if mocap_box_id < 0:
        return
    ok_rgba = np.array([0.0, 1.0, 0.0, 0.1], dtype=np.float32)
    fail_rgba = np.array([1.0, 0.0, 0.0, 0.1], dtype=np.float32)
    mj_ctx.model.geom_rgba[mocap_box_id] = ok_rgba if success else fail_rgba


def get_mocap_pose(mj_ctx: MujocoContext, mocap_id: int) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(mj_ctx.data.mocap_pos[mocap_id], dtype=np.float64).copy()
    quaternion = np.asarray(mj_ctx.data.mocap_quat[mocap_id], dtype=np.float64).copy()
    return position, quaternion


def set_mocap_pose(mj_ctx: MujocoContext, mocap_id: int, position_xyz: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
    mj_ctx.data.mocap_pos[mocap_id] = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    mj_ctx.data.mocap_quat[mocap_id] = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)


def get_end_effector_pose(mj_ctx: MujocoContext, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    endpoint_pos = np.asarray(mj_ctx.data.site_xpos[site_id], dtype=np.float64).copy()
    endpoint_xmat = np.asarray(mj_ctx.data.site_xmat[site_id], dtype=np.float64).copy()
    endpoint_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(endpoint_quat, endpoint_xmat)
    endpoint_quat /= max(float(np.linalg.norm(endpoint_quat)), 1e-12)
    return endpoint_pos, endpoint_quat


def align_mocap_to_end_effector(mj_ctx: MujocoContext, mocap_id: int, site_id: int) -> None:
    pos, quat = get_end_effector_pose(mj_ctx, site_id)
    set_mocap_pose(mj_ctx, mocap_id, pos, quat)
    forward(mj_ctx)


__all__ = [
    "MujocoContext",
    "make_mujoco_context",
    "reset",
    "forward",
    "step",
    "get_joint_positions",
    "set_joint_positions",
    "get_joint_limits",
    "set_mocap_color",
    "get_mocap_pose",
    "set_mocap_pose",
    "get_end_effector_pose",
    "align_mocap_to_end_effector",
]
