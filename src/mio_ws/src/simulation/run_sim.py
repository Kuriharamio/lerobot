from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot_robot_so101_mujoco.common.mujoco_interface import MujocoInterface
from lerobot_robot_so101_mujoco.common.backend import SO101MujocoBackend
from lerobot_robot_so101_mujoco.config_so101_mujoco import SO101_DEFAULT_EE_SITE, SO101_DEFAULT_MOCAP_TARGET, SO101_JOINT_NAMES, default_xml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO101 MuJoCo simulation runner")
    parser.add_argument("--xml", type=Path, default=default_xml_path(), help="Path to so101.xml")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="TCP host for RPC server")
    parser.add_argument("--port", type=int, default=8765, help="TCP port for RPC server")
    parser.add_argument("--delta-scale-m", type=float, default=0.004, help="Translation metres per delta unit")
    parser.add_argument("--delta-scale-rad", type=float, default=0.02, help="Rotation radians per delta_r* unit")
    parser.add_argument("--gripper-step-rad", type=float, default=0.08, help="Gripper delta in radians")
    return parser.parse_args()


def run() -> None:
    args = parse_args()

    interface = MujocoInterface(
        xml_path=args.xml,
        joint_names=SO101_JOINT_NAMES,
        end_effector_site=SO101_DEFAULT_EE_SITE,
        target_mocap_body=SO101_DEFAULT_MOCAP_TARGET,
    )
    robot = SO101MujocoBackend(
        interface=interface,
        host=args.host,
        port=int(args.port),
        control_dt=float(args.dt),
        delta_position_scale_m=float(args.delta_scale_m),
        delta_rotation_scale_rad=float(args.delta_scale_rad),
        gripper_step_rad=float(args.gripper_step_rad),
        ik_solver="quadprog",
        ik_max_iters=50,
        ik_internal_dt=0.002,
        ik_velocity_stuck_threshold=1e-8,
        success_position_tolerance=5e-3,
        success_orientation_tolerance_rad=2e-1,
    )
    robot.connect()

    try:
        with mujoco.viewer.launch_passive(robot.model, robot.data, show_left_ui=False, show_right_ui=False) as viewer:
            while viewer.is_running():
                t_start = time.time()
                with viewer.lock():
                    robot.step()

                viewer.sync()
                time.sleep(max(0.0, float(args.dt) - (time.time() - t_start)))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    run()
