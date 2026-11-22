import math

import numpy as np

from arkit_fbt_relay.fake_phone import synth_joints
from arkit_fbt_relay.mathutil import (
    Rigid, quat_from_axis_angle, quat_mul, quat_rotate, to_unity_quat, unity_euler_from_quat,
)
from arkit_fbt_relay.pipeline import Processor, Settings, unity_yaw_from_forward
from arkit_fbt_relay.protocol import BodyFrame, Pose

CAM = Rigid(np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))


def frame(t, joints, cam=CAM, seq=0, tracked_joints=None):
    return BodyFrame(seq=seq, timestamp=t, tracked=joints is not None, scale=1.0,
                     camera=Pose(cam.pos, cam.rot), root=None, joints=joints or {},
                     tracked_joints=tracked_joints)


def raw_settings(**kw):
    # 필터·튐 제거·자세 보정 없이 기하만 보기 위한 설정
    base = dict(min_cutoff=1e6, beta=0.0, rot_min_cutoff=1e6, rot_beta=0.0, depth_ratio=1.0,
                spike_window=1, max_rel_speed=0.0, upright_correction=False, head_horizontal="head")
    base.update(kw)
    return Settings(**base)


def tilt(joints, deg, axis=(1.0, 0.0, 0.0)):
    """골반을 중심으로 뼈대 전체를 기울인다 (ARKit 깊이 추정 오차 흉내)."""
    q = quat_from_axis_angle(axis, math.radians(deg))
    pivot = joints["hips_joint"].pos
    return {n: Pose(pivot + quat_rotate(q, p.pos - pivot), quat_mul(q, p.rot)) for n, p in joints.items()}


def test_default_trackers():
    out = Processor(Settings()).process(frame(0.0, synth_joints(0.0)))
    assert out.tracked
    assert sorted(t.index for t in out.trackers) == [1, 2, 3]
    assert out.head_pos is not None


def test_facing_camera_and_yaw():
    p = Processor(raw_settings())
    for i in range(10):
        out = p.process(frame(i / 60, synth_joints(i / 60)))
    # 사람이 카메라(+Z 방향)를 보고 있다
    assert out.facing > 0.95
    assert np.allclose(out.forward, [0, 0, 1], atol=0.05)
    # ARKit +Z == Unity -Z  -> yaw 180
    assert abs(abs(unity_yaw_from_forward(out.forward)) - 180) < 3


def test_facing_learns_mirrored_labels():
    """ARKit 좌/우 규약이 반대여도 카메라 쪽을 정면으로 본다."""
    p = Processor(raw_settings())
    for i in range(10):
        j = synth_joints(i / 60)
        swapped = {}
        for name, pose in j.items():
            if name.startswith("left_"):
                swapped["right_" + name[5:]] = pose
            elif name.startswith("right_"):
                swapped["left_" + name[6:]] = pose
            else:
                swapped[name] = pose
        out = p.process(frame(i / 60, swapped))
    assert out.facing > 0.95


def test_turned_body_yaw():
    p = Processor(raw_settings())
    turn = math.radians(40)
    for i in range(10):
        out = p.process(frame(i / 60, synth_joints(i / 60, body_yaw=turn)))
    yaw_straight = 180.0
    yaw = unity_yaw_from_forward(out.forward)
    diff = (yaw - yaw_straight + 180) % 360 - 180
    assert abs(abs(diff) - 40) < 3


def test_camera_drift_is_cancelled():
    """ARKit 월드가 미끄러져도(카메라·관절이 같이 움직임) 결과는 그대로여야 한다."""
    p = Processor(raw_settings())
    base = p.process(frame(0.0, synth_joints(0.0)))
    drift_steps = 40
    for k in range(1, drift_steps + 1):
        d = Rigid(quat_from_axis_angle([0, 1, 0], 0.001 * k), np.array([0.002 * k, 0.0005 * k, -0.001 * k]))
        cam = d @ CAM
        j = {n: Pose(d.apply(pp.pos), quat_mul(d.rot, pp.rot)) for n, pp in synth_joints(0.0).items()}
        out = p.process(frame(k / 60, j, cam=cam))
    assert not out.events
    hip0 = next(t for t in base.trackers if t.name == "hip").pos
    hip1 = next(t for t in out.trackers if t.name == "hip").pos
    assert np.linalg.norm(hip1 - hip0) < 1e-3


def test_camera_moved_resets_reference():
    p = Processor(raw_settings())
    p.process(frame(0.0, synth_joints(0.0)))
    moved = Rigid(CAM.rot, CAM.pos + np.array([0.5, 0, 0]))
    out = p.process(frame(1 / 60, synth_joints(1 / 60), cam=moved))
    assert "camera_moved" in out.events


def test_single_frame_spike_is_removed():
    p = Processor(raw_settings(spike_window=3))
    for i in range(5):
        p.process(frame(i / 60, synth_joints(0.0)))
    j = synth_joints(0.0)
    j["left_foot_joint"] = Pose(j["left_foot_joint"].pos + np.array([1.0, 0, 0]), j["left_foot_joint"].rot)
    out = p.process(frame(5 / 60, j))
    foot = next(t for t in out.trackers if t.name == "left_foot").pos
    assert np.allclose(foot, synth_joints(0.0)["left_foot_joint"].pos, atol=1e-6)


def test_relative_speed_limit():
    p = Processor(raw_settings(max_rel_speed=3.0))
    p.process(frame(0.0, synth_joints(0.0)))
    j = synth_joints(0.0)
    j["left_foot_joint"] = Pose(j["left_foot_joint"].pos + np.array([0.5, 0, 0]), j["left_foot_joint"].rot)
    out = p.process(frame(1 / 60, j))
    foot = next(t for t in out.trackers if t.name == "left_foot").pos
    moved = np.linalg.norm(foot - synth_joints(0.0)["left_foot_joint"].pos)
    assert abs(moved - 3.0 / 60) < 1e-6  # 한 프레임에 3m/s 만큼만 움직인다


def test_hands_up_detected():
    p = Processor(Settings())
    assert not p.process(frame(0.0, synth_joints(0.0))).hands_up
    assert p.process(frame(0.1, synth_joints(0.1, hands_up=True))).hands_up


def test_filter_reduces_noise():
    rng = np.random.default_rng(3)
    p = Processor(Settings())
    errs = []
    truth = synth_joints(0.0)["hips_joint"].pos
    for i in range(240):
        out = p.process(frame(i / 60, synth_joints(0.0, noise=0.02, rng=rng)))
        if i > 30:
            errs.append(np.linalg.norm(next(t for t in out.trackers if t.name == "hip").pos - truth))
    assert np.mean(errs) < 0.02  # 잡음(축당 2cm, 3D 약 3.2cm)보다 작아야 한다


def test_upright_correction_removes_tilt():
    """카메라 쪽으로 12° 기운 뼈대를 보정하면 가슴이 골반 바로 위로 온다."""
    p = Processor(raw_settings(upright_correction=True, upright_tau=0.2))
    for i in range(120):
        out = p.process(frame(i / 60, tilt(synth_joints(0.0), 12)))
    assert 10 < p.lean_deg < 14
    hips, chest = out.skeleton["hips_joint"], out.skeleton["spine_7_joint"]
    assert np.hypot(*(chest - hips)[[0, 2]]) < 0.02
    # calibrate_upright 는 즉시 맞춘다
    p2 = Processor(raw_settings(upright_correction=True, upright_tau=1000))
    p2.process(frame(0.0, tilt(synth_joints(0.0), 12)))
    assert p2.calibrate_upright() is not None and 10 < p2.lean_deg < 14


def test_whole_body_depth_jitter_cancels_in_relative_positions():
    """몸 전체가 카메라 깊이 방향으로 출렁여도 머리 기준 상대 위치는 흔들리지 않는다."""
    rng = np.random.default_rng(5)
    p = Processor(Settings(upright_correction=False))
    rels = []
    for i in range(180):
        shift = np.array([0.0, 0.0, rng.normal(0, 0.08)])
        j = {n: Pose(v.pos + shift, v.rot) for n, v in synth_joints(0.0).items()}
        out = p.process(frame(i / 60, j))
        foot = next(t for t in out.trackers if t.name == "left_foot").pos
        rels.append(foot - out.head_pos)
    # 8cm 출렁임이 1mm 미만으로 (깊이 축 방향이 몸 위치에 따라 아주 조금 바뀌는 만큼만 남음)
    assert np.std(np.array(rels[30:]), axis=0).max() < 1e-3


def test_untracked_joints_hold_last_value():
    p = Processor(raw_settings())
    out0 = p.process(frame(0.0, synth_joints(0.0)))
    foot0 = next(t for t in out0.trackers if t.name == "left_foot").pos
    j = synth_joints(0.0)
    j["left_foot_joint"] = Pose(j["left_foot_joint"].pos + np.array([0.0, 0.5, 0.7]), j["left_foot_joint"].rot)
    good = frozenset(n for n in j if n != "left_foot_joint")  # 왼발만 '추적 안 됨'(엉터리 추정값)
    out1 = p.process(frame(1 / 60, j, tracked_joints=good))
    foot1 = next(t for t in out1.trackers if t.name == "left_foot").pos
    assert np.allclose(foot1, foot0)


def test_untracked_hips_hold_whole_output():
    p = Processor(raw_settings())
    out0 = p.process(frame(0.0, synth_joints(0.0)))
    out1 = p.process(frame(1 / 60, synth_joints(0.5), tracked_joints=frozenset()))
    assert out1.tracked
    for a, b in zip(out0.trackers, out1.trackers):
        assert np.allclose(a.pos, b.pos)


def test_flat_feet_rotation_is_yaw_only():
    p = Processor(Settings(enabled_trackers=["left_foot", "right_knee"]))
    j = {n: Pose(v.pos, quat_mul(quat_from_axis_angle([1, 0, 0], 1.2), v.rot)) for n, v in synth_joints(0.0).items()}
    for i in range(30):
        out = p.process(frame(i / 60, j))
    for t in out.trackers:
        x, _y, z = unity_euler_from_quat(to_unity_quat(t.rot))
        assert abs(x) < 1e-3 and abs(z) < 1e-3


def test_head_horizontal_from_chest():
    p = Processor(raw_settings(head_horizontal="chest", enabled_trackers=["chest"]))
    out = p.process(frame(0.0, tilt(synth_joints(0.0), 10)))
    chest = next(t for t in out.trackers if t.name == "chest").pos
    assert np.allclose(out.head_pos[[0, 2]], chest[[0, 2]])
    assert out.head_pos[1] > chest[1] + 0.15


def test_not_tracked_frame():
    out = Processor(Settings()).process(frame(0.0, None))
    assert not out.tracked and not out.trackers
