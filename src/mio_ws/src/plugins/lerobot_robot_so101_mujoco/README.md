# lerobot_robot_so101_mujoco

Installable LeRobot robot plugin package for SO101 MuJoCo integration.

Install locally:

```bash
pip install -e .
```

## 两终端遥操（仿真 + lerobot-teleoperate）

先在终端 1 启动仿真服务（会打开 MuJoCo viewer）：

```bash
python src/mio_ws/src/simulation/run_sim.py --host 127.0.0.1 --port 8765
```

再在终端 2 使用 LeRobot 内置键盘遥操器启动 teleoperate：

```bash
lerobot-teleoperate \
	--robot.type=so101_mujoco \
	--robot.remote_host=127.0.0.1 \
	--robot.remote_port=8765 \
	--teleop.type=keyboard_ee \
	--fps=60
```

动作链路：
- `keyboard_ee` 输出末端位姿增量（`delta_x/y/z` + `delta_rx/ry/rz`）和 `gripper`
- 仿真端的机械臂类 `SO101MujocoSimRobot` 内部创建 RPC 服务端并执行 IK
- teleoperate 端机器人只发送增量动作，仿真端返回并维护 6 维关节状态（单位 rad）

并行控制：
- 可以同时使用键盘控制和 MuJoCo viewer 中拖动 mocap 控制
- 每个仿真步以当前 mocap 位姿为基准，再叠加键盘增量后做 IK 求解
