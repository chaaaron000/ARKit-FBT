"""쿼터니언/강체 변환과 ARKit -> Unity(VRChat) 좌표 변환.

내부 계산은 모두 ARKit 좌표계(오른손, +Y 위, 미터)에서 한다.
쿼터니언은 항상 (x, y, z, w) 순서다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return IDENTITY_Q.copy()
    return q / n


def quat_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conj(q) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def quat_rotate(q, v) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)])


def quat_angle(q) -> float:
    """회전량(라디안, 0..pi)."""
    w = min(1.0, abs(float(quat_normalize(q)[3])))
    return 2.0 * math.acos(w)


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(slots=True)
class Rigid:
    """p' = rot * p + pos"""

    rot: np.ndarray
    pos: np.ndarray

    @staticmethod
    def identity() -> Rigid:
        return Rigid(IDENTITY_Q.copy(), np.zeros(3))

    def apply(self, p) -> np.ndarray:
        return quat_rotate(self.rot, p) + self.pos

    def inverse(self) -> Rigid:
        inv = quat_conj(self.rot)
        return Rigid(inv, -quat_rotate(inv, self.pos))

    def __matmul__(self, other: Rigid) -> Rigid:
        """(a @ b).apply(p) == a.apply(b.apply(p))"""
        return Rigid(quat_normalize(quat_mul(self.rot, other.rot)), self.apply(other.pos))


# ---------------------------------------------------------------------------
# ARKit(오른손) -> Unity(왼손): Z축을 뒤집는다.
# 위치: (x, y, z) -> (x, y, -z)
# 회전: S R S (S = diag(1, 1, -1)) 는 쿼터니언에서 (x, y, z, w) -> (-x, -y, z, w)
# ---------------------------------------------------------------------------

def to_unity_pos(p) -> np.ndarray:
    return np.array([p[0], p[1], -p[2]], dtype=float)


def to_unity_quat(q) -> np.ndarray:
    return np.array([-q[0], -q[1], q[2], q[3]], dtype=float)


def unity_euler_from_quat(q) -> tuple[float, float, float]:
    """Unity 쿼터니언 -> Unity 오일러각 (x, y, z) 도 단위.

    Unity는 Z, X, Y 순서로 적용한다: R = Ry * Rx * Rz.
    """
    m = quat_to_matrix(q)
    sx = max(-1.0, min(1.0, -m[1, 2]))
    x = math.asin(sx)
    if math.cos(x) > 1e-6:
        y = math.atan2(m[0, 2], m[2, 2])
        z = math.atan2(m[1, 0], m[1, 1])
    else:  # 짐벌락
        y = math.atan2(-m[2, 0], m[0, 0])
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


def unity_quat_from_euler(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """테스트용: Unity Quaternion.Euler 와 같은 결과."""
    qx = quat_from_axis_angle([1, 0, 0], math.radians(x_deg))
    qy = quat_from_axis_angle([0, 1, 0], math.radians(y_deg))
    qz = quat_from_axis_angle([0, 0, 1], math.radians(z_deg))
    return quat_mul(quat_mul(qy, qx), qz)
