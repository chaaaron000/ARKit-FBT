import math

import numpy as np
import pytest

from arkit_fbt_relay.mathutil import (
    Rigid, quat_from_axis_angle, quat_mul, quat_normalize, quat_rotate, quat_to_matrix,
    to_unity_pos, to_unity_quat, unity_euler_from_quat, unity_quat_from_euler,
)

rng = np.random.default_rng(1)


def random_quat():
    return quat_normalize(rng.normal(size=4))


def same_rotation(a, b, tol=1e-6):
    return abs(abs(float(np.dot(quat_normalize(a), quat_normalize(b)))) - 1) < tol


def test_quat_matrix_matches_rotate():
    for _ in range(50):
        q, v = random_quat(), rng.normal(size=3)
        assert np.allclose(quat_to_matrix(q) @ v, quat_rotate(q, v))


def test_unity_euler_roundtrip():
    for _ in range(500):
        q = random_quat()
        x, y, z = unity_euler_from_quat(q)
        assert same_rotation(unity_quat_from_euler(x, y, z), q)


@pytest.mark.parametrize("x", [90.0, -90.0])
def test_unity_euler_gimbal(x):
    q = unity_quat_from_euler(x, 30.0, 0.0)
    assert same_rotation(unity_quat_from_euler(*unity_euler_from_quat(q)), q, tol=1e-5)


def test_unity_euler_known_values():
    # Unity: Quaternion.Euler(0, 90, 0) 은 +Y축 기준 90도
    assert np.allclose(unity_euler_from_quat(quat_from_axis_angle([0, 1, 0], math.pi / 2)), (0, 90, 0))
    assert np.allclose(unity_euler_from_quat(quat_from_axis_angle([1, 0, 0], 0.3)), (math.degrees(0.3), 0, 0))


def test_handedness_conversion_is_consistent():
    """회전 후 변환 == 변환 후 회전 (위치와 회전 변환이 같은 거울상을 쓰는지)."""
    for _ in range(100):
        q, v = random_quat(), rng.normal(size=3)
        lhs = to_unity_pos(quat_rotate(q, v))
        rhs = quat_rotate(to_unity_quat(q), to_unity_pos(v))
        assert np.allclose(lhs, rhs)


def test_rigid_compose_inverse():
    for _ in range(50):
        a = Rigid(random_quat(), rng.normal(size=3))
        b = Rigid(random_quat(), rng.normal(size=3))
        p = rng.normal(size=3)
        assert np.allclose((a @ b).apply(p), a.apply(b.apply(p)))
        assert np.allclose((a.inverse() @ a).apply(p), p)


def test_quat_mul_associative():
    a, b, c = random_quat(), random_quat(), random_quat()
    assert np.allclose(quat_mul(quat_mul(a, b), c), quat_mul(a, quat_mul(b, c)))
