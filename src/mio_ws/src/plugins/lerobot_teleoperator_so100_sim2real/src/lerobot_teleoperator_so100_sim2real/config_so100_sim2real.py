from dataclasses import dataclass, field

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so100_sim2real")
@dataclass
class SO100Sim2RealConfig(TeleoperatorConfig):
    host: str = "127.0.0.1"
    port: int = 18081
    use_degrees: bool = True
    # Sim-joint angles (in teleop output unit, default: degrees) when the real robot is at its physical zero pose.
    # Joint order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
    sim_zero_joint_angles: list[float] = field(default_factory=lambda: [0.0, 101.4135, -92.81916, -34.377468, 90.0])
    # Per-joint sign mapping for sim->real conversion. Use -1.0 or 1.0.
    # Joint order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
    sim_joint_directions: list[float] = field(default_factory=lambda: [-1.0, 1.0, 1.0, 1.0, 1.0])
    gripper_joint_min: float = 0.0
    gripper_joint_max: float = 1.0
