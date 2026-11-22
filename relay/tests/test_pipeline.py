import math

import numpy as np

from arkit_fbt_relay.fake_phone import synth_joints
from arkit_fbt_relay.mathutil import Rigid, quat_from_axis_angle, quat_mul
from arkit_fbt_relay.pipeline import Processor, Settings, unity_yaw_from_forward
from arkit_fbt_relay.protocol import BodyFrame, Pose

CAM = Rigid(np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))


def frame(t, joints, cam=CAM, seq=0):
    return BodyFrame(seq=seq, timestamp=t, tracked=joints is not None, scale=1.0,
                     camera=Pose(cam.pos, cam.rot), root=None, joints=joints or {})


def raw_settings(**kw):
    # 필터 영향 없이 기하만 보기 위해 필터를 거의 끈다
    return Settings(min_cutoff=1e6, beta=0.0, rot_min_cutoff=1e6, rot_beta=0.0, **kw)


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


def test_outlier_is_rejected():
    p = Processor(raw_settings(max_speed=6.0))
    for i in range(5):
        p.process(frame(i / 60, synth_joints(0.0)))
    j = synth_joints(0.0)
    j["left_foot_joint"] = Pose(j["left_foot_joint"].pos + np.array([1.0, 0, 0]), j["left_foot_joint"].rot)
    out = p.process(frame(5 / 60, j))
    foot = next(t for t in out.trackers if t.name == "left_foot").pos
    assert np.allclose(foot, synth_joints(0.0)["left_foot_joint"].pos, atol=1e-6)


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


def test_not_tracked_frame():
    out = Processor(Settings()).process(frame(0.0, None))
    assert not out.tracked and not out.trackers
