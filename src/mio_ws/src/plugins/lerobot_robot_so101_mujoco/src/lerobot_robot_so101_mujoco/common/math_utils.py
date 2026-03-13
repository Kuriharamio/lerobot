from __future__ import annotations

import numpy as np


def clamp(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def axis_angle_quaternion(ax: float, ay: float, az: float, angle: float) -> np.ndarray:
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    return np.array(
        [
            np.cos(half_angle),
            ax * sin_half,
            ay * sin_half,
            az * sin_half,
        ],
        dtype=np.float64,
    )


def multiply_quaternions(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(x) for x in q1]
    w2, x2, y2, z2 = [float(x) for x in q2]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(q, dtype=np.float64) / norm


def euler_xyz_to_quaternion(rx: float, ry: float, rz: float) -> np.ndarray:
    qx = axis_angle_quaternion(1.0, 0.0, 0.0, float(rx))
    qy = axis_angle_quaternion(0.0, 1.0, 0.0, float(ry))
    qz = axis_angle_quaternion(0.0, 0.0, 1.0, float(rz))
    return normalize_quaternion(multiply_quaternions(qz, multiply_quaternions(qy, qx)))


def apply_rpy_delta(quaternion_wxyz: np.ndarray, delta_rpy: np.ndarray) -> np.ndarray:
    current = normalize_quaternion(np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4))
    drx, dry, drz = [float(v) for v in np.asarray(delta_rpy, dtype=np.float64).reshape(3)]
    delta_quat = euler_xyz_to_quaternion(drx, dry, drz)
    return normalize_quaternion(multiply_quaternions(delta_quat, current))
