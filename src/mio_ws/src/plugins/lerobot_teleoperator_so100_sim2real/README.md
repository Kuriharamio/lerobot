# lerobot_teleoperator_so100_sim2real

Installable LeRobot teleoperator plugin that receives SO100 MuJoCo joint states through XML-RPC and outputs teleop actions for controlling a real `so100_follower` robot.

Install locally:

```bash
pip install -e .
```

Run:

```bash
lerobot-teleoperate \
    --robot.type=so100_follower \
    --teleop.type=so100_sim2real \
    --robot.port=/dev/ttyUSB0 \
    --fps=50
```

Zero-pose alignment (optional):

`sim_zero_joint_angles` defines the simulated joint angles when the real arm is at physical zero.
Order: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]`

`sim_joint_directions` defines per-joint motion direction mapping from sim to real (`-1` or `1`).
Order: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]`



