"""User-visible foot-lock failures found by the independent QA round (fixed in round 1)."""
import math

import numpy as np
import pytest

from arkit_fbt_relay.engine import Engine
from arkit_fbt_relay.fake_phone import synth_joints
from arkit_fbt_relay.pipeline import Processor, Settings
from arkit_fbt_relay.protocol import Pose, build_frame, parse_frame

CAMERA = Pose(np.array([0., 1., 0.]), np.array([0., 0., 0., 1.]))
REST = synth_joints(0.0)
LEFT = "left_foot_joint"
RIGHT = "right_foot_joint"


def packet(i, *, left=(0., 0., 0.), travel=0., crouch=0., camera=CAMERA, masked=False):
    joints = {}
    for name, p in REST.items():
        if name in (LEFT, "left_toes_joint"):
            delta = left
        elif name in (RIGHT, "right_toes_joint"):
            delta = (0., 0., 0.)
        else:
            delta = (travel, -crouch * (0.5 if name.endswith('_leg_joint') else 1), 0.)
        joints[name] = Pose(p.pos + delta, p.rot)
    return build_frame(i + 1, i / 60, camera, joints, untracked=set(joints) if masked else None)


def feet(out, side="left"):
    return next(t.pos for t in out.trackers if t.name == f"{side}_foot")


def feed(p, i, **kwargs):
    return p.process(parse_frame(packet(i, **kwargs)))


def planted(p):
    for i in range(120):
        out = feed(p, i)
    return out


# 발을 끌거나 작게 디디는 움직임을 보이게 하는 것은 "motion"(움직임 우선) 모드의 약속이다. 나머지(앵커 지연,
# 섞기, 설정 변경)는 두 모드 모두 지킨다.
MODES = pytest.mark.parametrize("mode", ["motion", "stable"])


@pytest.mark.parametrize(("distance", "speed"), [(0.15, 0.30), (0.25, 0.05)], ids=["short", "slow"])
def test_sustained_floor_drag_follows_moving_foot(distance, speed):
    p = Processor(Settings(foot_lock_mode="motion"))
    start = feet(planted(p))[0]
    last = round(distance / speed * 60)
    for j in range(1, last + 1):
        out = feed(p, 120 + j, left=(distance * j / last, 0., 0.))
    assert feet(out)[0] - start > 0.65 * distance


def test_five_centimeter_in_place_step_is_visible():
    p = Processor(Settings(foot_lock_mode="motion"))
    start = feet(planted(p))[1]
    heights = [feet(feed(p, 120 + i, left=(0., 0.05 * math.sin(math.pi * i / 27), 0.)))[1]
               for i in range(28)]
    assert max(heights) - start > 0.03


@MODES
def test_planted_foot_height_in_vrchat_during_crouch(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    standing = planted(p)
    rest_rel = (feet(standing) - standing.head_pos)[1]
    errors = []
    for i in range(120, 241):
        drop = 0.32 * min(1., (i - 120) / 48)
        out = feed(p, i, crouch=drop)
        errors.append(abs((feet(out) - out.head_pos)[1] - (rest_rel + drop)))
    assert max(errors) < 0.05


@MODES
def test_first_support_foot_in_vrchat_while_body_advances(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    standing = planted(p)
    rest_rel = (feet(standing, "right") - standing.head_pos)[0]
    for i in range(120, 150):
        t = (i - 120) / 60
        swing = min(1., t / 0.4)
        out = feed(p, i, left=(0.25 * (1 - math.cos(math.pi * swing)), 0.10 * math.sin(math.pi * swing), 0.),
                   travel=0.5 * t)
    assert abs((feet(out, "right") - out.head_pos)[0] - (rest_rel - 0.5 * t)) < 0.05


def test_floor_reacquires_after_camera_moves_thirty_cm_over_one_second():
    p = Processor(Settings())
    planted(p)
    events = []
    for i in range(120, 300):
        offset = 0.3 * min(1., (i - 120) / 60)
        out = feed(p, i, camera=Pose(CAMERA.pos - (0., offset, 0.), CAMERA.rot))
        events.extend(out.events)
    assert "camera_moved" in events and p.foot_locked == {"left_foot", "right_foot"}


@MODES
def test_release_blend_does_not_add_a_five_cm_step_under_mask_flicker(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    planted(p)
    samples = []
    for i in range(120, 220):
        lift = np.clip((i - 160) / 60, 0., 0.18)
        samples.append(feet(feed(p, i, left=(0., lift, 0.), masked=(i % 2 == 0))))
    steps = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    assert max(steps) < 0.05


@MODES
def test_switching_off_during_lift_does_not_jump(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    planted(p)
    for i in range(120, 127):
        before = feet(feed(p, i, left=(0., (i - 119) / 60, 0.)))
    p.settings.foot_lock = False
    after = feet(feed(p, 127, left=(0., 8 / 60, 0.)))
    assert np.linalg.norm(after - before) < 0.04


def test_switching_off_and_back_on_while_tracking_is_masked_resets_floor():
    engine = Engine(Settings(auto_yaw_on_start=False), osc_port=43862, dry_run=True)
    address = ("127.0.0.1", 42001)
    for i in range(120):
        engine.handle_packet(packet(i), address)
    engine.update_settings(foot_lock=False)
    engine.handle_packet(packet(120, masked=True), address)
    engine.update_settings(foot_lock=True)
    engine.handle_packet(packet(121), address)
    snapshot = engine.snapshot()
    assert not snapshot.floor_ready and not snapshot.foot_locked


@MODES
def test_reenabling_left_tracker_after_moving_it_does_not_restore_old_plant(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    start = feet(planted(p))[0]
    p.settings.enabled_trackers = ["hip", "right_foot"]
    for i in range(120, 210):
        feed(p, i, left=(0.13 * min(1., (i - 120) / 40), 0., 0.))
    p.settings.enabled_trackers.append("left_foot")
    for i in range(210, 239):
        out = feed(p, i, left=(0.13, 0., 0.))
    assert feet(out)[0] - start > 0.10
