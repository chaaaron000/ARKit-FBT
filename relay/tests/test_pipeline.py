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
    p = Processor(raw_settings(lock_yaw_to_camera=False))
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
    """몸 전체가 카메라 깊이 방향으로 출렁여도 머리 기준 상대 위치는 흔들리지 않는다.

    발 고정은 끈다: 고정된 발은 월드(앵커) 좌표에 있어서, 머리 기준으로는 앵커의 느린 움직임이 남는다
    (설계상 맞바꾼 것. test_foot_lock_* 참고)."""
    rng = np.random.default_rng(5)
    p = Processor(Settings(upright_correction=False, foot_lock=False))
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


def test_lock_yaw_to_camera_ignores_wrong_arkit_orientation():
    """ARKit이 몸을 60° 돌아간 것으로 잘못 잡아도 골반·발 방향은 카메라 정면을 유지한다."""
    p = Processor(Settings(enabled_trackers=["hip", "left_foot"]))
    for i in range(60):
        out = p.process(frame(i / 60, synth_joints(i / 60, body_yaw=math.radians(60))))
    assert out.facing < 0.6  # ARKit 기준으로는 많이 돌아간 상태
    assert np.allclose(out.forward, [0, 0, 1], atol=0.05)  # 카메라(+Z) 쪽
    for t in out.trackers:
        x, y, z = unity_euler_from_quat(to_unity_quat(t.rot))
        assert abs(x) < 1e-3 and abs(z) < 1e-3
        # 카메라 정면 방향의 yaw 회전 (ARKit 몸 방향 60° 와 무관). VRChat 캘리브레이션이 오프셋을 기억하므로
        # 절대값보다 '항상 같은 값'인 것이 중요하다.
        assert abs(y) < 3


def test_head_horizontal_from_chest():
    p = Processor(raw_settings(head_horizontal="chest", enabled_trackers=["chest"]))
    out = p.process(frame(0.0, tilt(synth_joints(0.0), 10)))
    chest = next(t for t in out.trackers if t.name == "chest").pos
    assert np.allclose(out.head_pos[[0, 2]], chest[[0, 2]])
    assert out.head_pos[1] > chest[1] + 0.15


def test_not_tracked_frame():
    out = Processor(Settings()).process(frame(0.0, None))
    assert not out.tracked and not out.trackers


# ---------------------------------------------------------------- 발 고정

def standing(left_foot=None, noise=0.0, rng=None, shift=(0.0, 0.0, 0.0)):
    """가만히 선 사람. left_foot: 왼발 원본 위치(없으면 제자리), noise: 왼발에만 넣는 잡음 (m)."""
    j = {n: Pose(v.pos + np.asarray(shift), v.rot) for n, v in synth_joints(0.0).items()}
    p = j["left_foot_joint"].pos if left_foot is None else np.asarray(left_foot)
    if noise:
        p = p + rng.normal(0, noise, 3)
    j["left_foot_joint"] = Pose(p, j["left_foot_joint"].rot)
    return j


def tracker(out, name="left_foot"):
    return next(t.pos for t in out.trackers if t.name == name)


def test_foot_lock_steadies_noisy_standing_foot():
    """가만히 선 발에 잡음이 있어도 고정되면 VRChat이 보는 (발 - 머리)가 거의 안 움직인다."""
    def run(foot_lock):
        rng = np.random.default_rng(7)
        p = Processor(Settings(foot_lock=foot_lock))
        rels = []
        for i in range(360):
            out = p.process(frame(i / 60, standing(noise=0.008, rng=rng)))
            if i >= 120:
                rels.append(tracker(out) - out.head_pos)
        return p, np.array(rels)

    p_on, on = run(True)
    _, off = run(False)
    assert "left_foot" in p_on.foot_locked
    std_on, std_off = np.std(on, axis=0).max(), np.std(off, axis=0).max()
    assert std_on < 0.002
    assert std_off > 3 * std_on
    step_on = np.linalg.norm(np.diff(on, axis=0), axis=1)
    assert step_on.max() < 0.001


def test_foot_lock_releases_when_foot_is_lifted():
    p = Processor(Settings())
    rest = synth_joints(0.0)["left_foot_joint"].pos
    for i in range(120):
        p.process(frame(i / 60, standing()))
    assert "left_foot" in p.foot_locked
    ys, released = [], None
    for i in range(1, 31):  # 1 m/s 로 들어 올린다
        out = p.process(frame(2.0 + i / 60, standing(rest + np.array([0.0, i / 60, 0.0]))))
        if released is None and "left_foot" not in p.foot_locked:
            released = i / 60
        ys.append(tracker(out)[1])
    assert released is not None and released <= 0.12  # 6cm 오르는 데 0.06초 + 확인 3프레임
    assert ys[-1] - ys[0] > 0.2  # 출력도 따라 올라온다
    assert np.abs(np.diff(ys)).max() < 0.04  # 계단식 점프 없이


def test_stable_mode_holds_a_ground_drag_that_motion_mode_follows():
    """'움직임 우선'은 바닥에서 10cm 끈 발을 따라가고, '정지 안정 우선'은 들지 않은 미끄러짐으로 보고 그 자리에
    고정한 채로 둔다."""
    rest = synth_joints(0.0)["left_foot_joint"].pos
    moved = {}
    for mode in ("motion", "stable"):
        p = Processor(Settings(foot_lock_mode=mode))
        for i in range(120):
            start = tracker(p.process(frame(i / 60, standing())))
        for i in range(1, 91):  # 1초에 10cm 끈 뒤 0.5초 가만히
            out = p.process(frame(2.0 + i / 60, standing(rest + np.array([0.1 * min(1.0, i / 60), 0.0, 0.0]))))
        moved[mode] = tracker(out)[0] - start[0]
        assert "left_foot" in p.foot_locked
    assert moved["motion"] > 0.065
    assert abs(moved["stable"]) < 0.03  # 고정 위치가 원본 쪽으로 천천히(4초 시간 상수) 옮겨 가는 만큼만


def test_switching_foot_lock_mode_while_planted_does_not_jump():
    """고정된 발이 여유 안에서 4cm 어긋나 있을 때 모드를 바꾸면, 고정을 섞어서 풀고 새 모드로 다시 고정한다."""
    rest = synth_joints(0.0)["left_foot_joint"].pos
    shifted = rest + np.array([0.04, 0.0, 0.0])
    p = Processor(Settings(foot_lock_mode="motion"))
    for i in range(120):
        p.process(frame(i / 60, standing()))
    for i in range(120, 150):  # 고정된 채로 원본만 4cm 옮겨 감 (여유 5cm 안: 출력은 그대로)
        before = tracker(p.process(frame(i / 60, standing(rest + (shifted - rest) * (i - 119) / 30))))
    assert "left_foot" in p.foot_locked
    p.settings.foot_lock_mode = "stable"
    ys = [before]
    for i in range(150, 210):
        ys.append(tracker(p.process(frame(i / 60, standing(shifted)))))
    assert np.linalg.norm(np.diff(ys, axis=0), axis=1).max() < 0.01
    assert "left_foot" in p.foot_locked
    assert abs(ys[-1][0] - before[0] - 0.04) < 0.01


def walker(t, v=0.5, t0=2.0, t1=6.0, step=1.0, swing=0.4, lift=0.10):
    """카메라 앞을 옆으로 걷는 사람 (t0~t1). 발은 디딘 동안 제자리, 한 걸음마다 swing 초 동안 옮긴다.

    반환: (관절, 실제 왼발 위치)."""
    hips = np.array([v * min(max(t - t0, 0.0), t1 - t0), 0.95, -2.6])
    joints = {n: Pose(hips + (p.pos - synth_joints(0.0)["hips_joint"].pos), p.rot)
              for n, p in synth_joints(0.0).items()}
    feet = {}
    for name, phase in (("left_foot_joint", 0.0), ("right_foot_joint", 0.5)):
        base = synth_joints(0.0)[name].pos.copy()
        x, y = base[0], base[1]
        k = 0
        while (k + phase) * step < min(t, t1) - t0:
            ph = min(1.0, (t - t0 - (k + phase) * step) / swing)
            x += v * step * (0.5 - 0.5 * math.cos(math.pi * ph))
            if ph < 1.0:
                y += lift * math.sin(math.pi * ph)
            k += 1
        feet[name] = np.array([x, y, base[2]])
        joints[name] = Pose(feet[name], joints[name].rot)
    return joints, feet["left_foot_joint"]


def test_foot_lock_does_not_drag_walking_feet():
    """걸을 때는 발이 고정에 묶여 끌려가지 않고 따라가며, 멈추면 다시 고정된다."""
    def run(foot_lock):
        p = Processor(Settings(foot_lock=foot_lock, upright_correction=False, head_horizontal="head"))
        errs, locked = [], []
        for i in range(int(10 * 60)):
            t = i / 60
            j, foot = walker(t)
            out = p.process(frame(t, j))
            errs.append(np.linalg.norm((tracker(out) - out.head_pos) - (foot - j["head_joint"].pos)))
            locked.append("left_foot" in p.foot_locked)
        return np.array(errs), np.array(locked)

    err_on, locked = run(True)
    err_off, _ = run(False)
    walking = slice(2 * 60, 6 * 60)
    assert locked[2 * 60 - 1]  # 걷기 전에는 고정
    # 걷는 동안: 발을 들면 풀리고, 몸이 움직이는 동안(앵커가 뒤처짐)은 새로 고정하지 않아 고정 끈 것만큼 따라간다
    assert err_on[walking].max() < err_off[walking].max() + 0.03
    assert err_on[walking].mean() < err_off[walking].mean() + 0.01
    # 멈추면 다시 고정되고, 앵커가 늦게 따라오며 생긴 어긋남은 몇 cm 안으로 남았다가 줄어든다
    assert locked[-60:].all()
    assert err_on[-60:].max() < 0.07


def test_floor_height_is_estimated_from_resting_feet():
    rng = np.random.default_rng(1)
    p = Processor(raw_settings())
    assert p.floor_y is None
    for i in range(600):  # 발을 번갈아 8cm씩 든다 (synth_joints 기본 동작)
        p.process(frame(i / 60, synth_joints(i / 60, noise=0.01, rng=rng)))
    rest_y = synth_joints(0.0)["left_foot_joint"].pos[1]
    assert rest_y - 0.025 < p.floor_y < rest_y + 0.005


def test_floor_is_reestimated_after_camera_moves():
    p = Processor(raw_settings())
    for i in range(120):
        p.process(frame(i / 60, standing()))
    rest_y = synth_joints(0.0)["left_foot_joint"].pos[1]
    assert abs(p.floor_y - rest_y) < 1e-6 and p.foot_locked
    # 폰이 움직였다 (새 기준 좌표에서 몸이 30cm 높게 보인다)
    moved = Rigid(CAM.rot, CAM.pos + np.array([0.5, 0.0, 0.0]))
    out = p.process(frame(2.0, standing(shift=(0, 0.3, 0)), cam=moved))
    assert "camera_moved" in out.events
    assert p.floor_y is None and not p.foot_locked
    for i in range(1, 120):
        p.process(frame(2.0 + i / 60, standing(shift=(0, 0.3, 0)), cam=moved))
    assert abs(p.floor_y - (rest_y + 0.3)) < 1e-6 and p.foot_locked
    # '카메라 기준 재설정' 버튼도 다시 잰다
    p.reset_camera()
    assert p.floor_y is None and not p.foot_locked


def test_slowly_lowered_camera_keeps_floor_until_camera_reference_is_reset():
    """폰을 10초에 걸쳐 30cm 내리면 camera_moved 가 안 걸려 몸이 통째로 30cm 높게 보인다. 발이 바닥보다 높게
    머무는 것만으로는 바닥을 다시 잡지 않는다 (앉아서 발을 띄운 것과 구별이 안 됨): '카메라 기준 재설정'을
    누르면 새 높이에서 다시 재고 고정한다."""
    p = Processor(raw_settings())
    for i in range(120):
        p.process(frame(i / 60, standing()))
    rest_y = synth_joints(0.0)["left_foot_joint"].pos[1]
    events = []
    t = 2.0
    for i in range(14 * 60):
        t = 2.0 + i / 60
        cam = Rigid(CAM.rot, CAM.pos - np.array([0.0, 0.3 * min(1.0, i / 600), 0.0]))
        events += p.process(frame(t, standing(), cam=cam)).events
    assert not events
    assert abs(p.floor_y - rest_y) < 0.01 and not p.foot_locked
    p.reset_camera()
    for i in range(1, 120):
        p.process(frame(t + i / 60, standing(), cam=cam))
    assert abs(p.floor_y - rest_y) < 0.01 and p.foot_locked == {"left_foot", "right_foot"}


def test_locked_foot_survives_tracking_loss_until_body_moves_away():
    """ARKit이 고정된 발을 놓치면 고정 위치를 유지하고, 몸이 멀리 가면 풀어서 마지막 상대 위치로 따라간다."""
    p = Processor(Settings(upright_correction=False))
    for i in range(120):
        out = p.process(frame(i / 60, standing()))
    locked_pos = tracker(out)
    assert "left_foot" in p.foot_locked
    no_left = frozenset(n for n in synth_joints(0.0) if n != "left_foot_joint")
    t = 2.0
    for i in range(60):  # 1초 동안 왼발을 놓침 (엉터리 추정값), 몸은 그대로
        t += 1 / 60
        out = p.process(frame(t, standing(left_foot=[0.0, 0.5, 0.0]), tracked_joints=no_left))
    assert np.allclose(tracker(out), locked_pos, atol=1e-6)
    for i in range(240):  # 몸이 옆으로 1m 걸어감 (왼발은 계속 못 봄)
        t += 1 / 60
        out = p.process(frame(t, standing(left_foot=[0.0, 0.5, 0.0], shift=(min(1.0, i / 120), 0, 0)),
                             tracked_joints=no_left))
    assert "left_foot" not in p.foot_locked
    hip = tracker(out, "hip")
    assert abs((tracker(out) - hip)[0] - (locked_pos - synth_joints(0.0)["hips_joint"].pos)[0]) < 0.05
