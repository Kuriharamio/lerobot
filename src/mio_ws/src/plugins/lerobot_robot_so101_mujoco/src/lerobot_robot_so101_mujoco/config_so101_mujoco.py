from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot.robots import RobotConfig

SO101_JOINT_NAMES: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SO101_GRIPPER_NAME: str = SO101_JOINT_NAMES[-1]
SO101_DEFAULT_EE_SITE = "gripperframe"
SO101_DEFAULT_MOCAP_TARGET = "arm_target"

def default_xml_path() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "models" / "so101" / "so101.xml"
        if candidate.is_file():
            return candidate
    return Path("models/so101/so101.xml")


def make_joint_action_features() -> dict[str, type]:
    return {f"{joint}.pos": float for joint in SO101_JOINT_NAMES}


def make_ee_delta_action_features() -> dict[str, type]:
    return {
        "delta_x": float,
        "delta_y": float,
        "delta_z": float,
        "delta_rx": float,
        "delta_ry": float,
        "delta_rz": float,
        "gripper": float,
    }


def make_joint_observation(joint_positions: dict[str, float]) -> dict[str, float]:
    return {f"{joint}.pos": float(joint_positions[joint]) for joint in SO101_JOINT_NAMES}

@RobotConfig.register_subclass("so101_mujoco")
@dataclass(kw_only=True)
class SO101MujocoRobotConfig(RobotConfig):
	xml_path: Path = default_xml_path()
	end_effector_site: str = SO101_DEFAULT_EE_SITE
	target_mocap_body: str = SO101_DEFAULT_MOCAP_TARGET
	control_dt: float = 0.02
	delta_position_scale_m: float = 0.0005
	delta_rotation_scale_rad: float = 0.01
	gripper_step_rad: float = 0.08
	ik_solver: str = "quadprog"
	ik_max_iters: int = 50
	ik_internal_dt: float = 0.002
	ik_velocity_stuck_threshold: float = 1e-8
	success_position_tolerance: float = 5e-3
	success_orientation_tolerance_rad: float = 2e-1


__all__ = ["SO101MujocoRobotConfig"]
