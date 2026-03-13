# lerobot_robot_so101_mujoco

Installable LeRobot robot plugin package for SO101 MuJoCo integration.

Install locally:

```bash
pip install -e .
```

## 遥操（lerobot-teleoperate）

使用 LeRobot 内置键盘遥操器启动 teleoperate：

```bash
lerobot-teleoperate \
	--robot.type=so101_mujoco \
	--teleop.type=keyboard_ee \
	--fps=50
```
