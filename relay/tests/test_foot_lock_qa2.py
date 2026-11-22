"""Consumer-visible foot-lock gaps found by the second independent QA pass (fixed in round 2).

Ground drags, small steps and fast lifts are promises of the "motion" foot-lock mode; anchor lag, floor and
settings behaviour hold in both modes. Slow phone moves that `camera_moved` cannot see (QA2 Q2-09, Q2-10) are
accepted by product decision: the user presses "카메라 기준 재설정"."""
import math

import numpy as np
import pytest

from arkit_fbt_relay.engine import Engine
from arkit_fbt_relay.pipeline import Processor, Settings
from arkit_fbt_relay.protocol import Pose, build_frame, parse_frame
from test_foot_lock_qa import CAMERA, LEFT, MODES, REST, RIGHT, feed, feet, packet, planted

BOTH_FEET = {'left_foot', 'right_foot'}


def feed_both(p, i, *, left=(0, 0, 0), right=(0, 0, 0), travel=0, drift=0):
    joints = {}
    for name, pose in REST.items():
        if name in (LEFT, 'left_toes_joint'):
            shift = left
        elif name in (RIGHT, 'right_toes_joint'):
            shift = right
        else:
            shift = (travel, 0, 0)
        joints[name] = Pose(pose.pos + shift + (drift, 0, 0), pose.rot)
    camera = Pose(CAMERA.pos + (drift, 0, 0), CAMERA.rot)
    return p.process(parse_frame(build_frame(i + 1, i / 60, camera, joints)))


@pytest.mark.parametrize(('distance', 'speed'), [
    pytest.param(0.05, 0.05, id='short', marks=pytest.mark.xfail(
        strict=True, reason='trade-off: a 5 cm drag stays inside the 5 cm horizontal slack (still-jitter budget)')),
    pytest.param(0.18, 0.80, id='fast')])
def test_ground_drag_reaches_new_position(distance, speed):
    p = Processor(Settings(foot_lock_mode='motion'))
    start = feet(planted(p))[0]
    for j in range(round(60 * (distance / speed + 0.55)) + 1):
        out = feed(p, 120 + j, left=(min(distance, speed * j / 60), 0, 0))
    assert feet(out)[0] - start > 0.65 * distance


def test_constant_slow_drag_does_not_make_catch_up_steps():
    p = Processor(Settings(foot_lock_mode='motion'))
    before = feet(planted(p))[0]
    steps = []
    for j in range(583):
        after = feet(feed(p, 120 + j, left=(min(0.32, 0.035 * j / 60), 0, 0)))[0]
        steps.append(abs(after - before))
        before = after
    assert max(steps) < 0.005


def test_two_grounded_feet_can_slide_sideways_together():
    p = Processor(Settings(foot_lock_mode='motion'))
    start = planted(p)
    for j in range(86):
        shift = (min(0.13, 0.15 * j / 60), 0, 0)
        out = feed_both(p, 120 + j, left=shift, right=shift)
    for side in ('left', 'right'):
        assert feet(out, side)[0] - feet(start, side)[0] > 0.085


def test_short_low_step_is_visible():
    p = Processor(Settings(foot_lock_mode='motion'))
    start = feet(planted(p))[1]
    heights = [feet(feed(p, 120 + j, left=(0, 0.025 * math.sin(math.pi * min(1, j / 14.4)), 0)))[1]
               for j in range(46)]
    assert max(heights) - start > 0.015


def test_small_step_returns_to_ground_after_landing():
    p = Processor(Settings(foot_lock_mode='motion'))
    start = feet(planted(p))[1]
    for j in range(70):
        height = 0.042 * math.sin(math.pi * min(1, j / 39))
        out = feed(p, 120 + j, left=(0, height, 0))
    assert abs(feet(out)[1] - start) < 0.010


@MODES
def test_locked_foot_does_not_creep_below_plant_during_downward_noise(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    start = feet(planted(p))[1]
    heights = [feet(feed(p, 120 + j, left=(0, -0.045 * math.sin(math.pi * j / 90), 0)))[1]
               for j in range(91)]
    assert min(heights) >= start - 0.003


@MODES
def test_slow_shallow_crouch_keeps_stationary_feet_planted(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    planted(p)
    locks = []
    for j in range(91):
        feed(p, 120 + j, crouch=0.16 * j / 90)
        locks.append(p.foot_locked)
    assert all(lock == BOTH_FEET for lock in locks)


@MODES
def test_deep_crouch_keeps_head_relative_foot_height(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    start = planted(p)
    relative = (feet(start) - start.head_pos)[1]
    errors = []
    for j in range(235):
        descent = min(1, j / 102)
        ascent = min(1, max(0, (j - 117) / 102))
        drop = 0.45 * (descent - ascent)
        out = feed(p, 120 + j, crouch=drop)
        errors.append(abs((feet(out) - out.head_pos)[1] - relative - drop))
    assert max(errors) < 0.05


@MODES
def test_first_support_foot_does_not_slide_at_faster_walking_pace(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    start = planted(p)
    relative = (feet(start, 'right') - start.head_pos)[0]
    errors = []
    for j in range(27):
        t = j / 60
        phase = math.pi * min(1, t / 0.44)
        out = feed(p, 120 + j, travel=0.7 * t,
                   left=(0.7 * 0.44 * (1 - math.cos(phase)), 0.105 * math.sin(phase), 0))
        errors.append(abs((feet(out, 'right') - out.head_pos)[0] - relative + 0.7 * t))
    assert max(errors) < 0.05


def test_faster_lift_does_not_add_a_five_cm_step_under_mask_flicker():
    p = Processor(Settings(foot_lock_mode='motion'))
    before = feet(planted(p))
    steps = []
    for j in range(60):
        out = feed(p, 120 + j, left=(0, min(0.24, 1.35 * j / 60), 0), masked=j % 2 == 1)
        after = feet(out)
        steps.append(np.linalg.norm(after - before))
        before = after
    assert max(steps) < 0.05


def test_common_world_drift_is_not_relocation_when_person_moves_with_lock_disabled():
    p = Processor(Settings(foot_lock=False))
    planted(p)
    events = []
    for j in range(121):
        travel, drift = 0.25 * j / 60, 0.17 * j / 60
        out = feed_both(p, 120 + j, left=(travel, 0, 0), right=(travel, 0, 0), travel=travel, drift=drift)
        events.extend(out.events)
    assert 'camera_moved' not in events


@MODES
def test_slow_upward_camera_move_does_not_leave_feet_above_head_relative_rest_height(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    start = planted(p)
    before = (feet(start) - start.head_pos)[1]
    for j in range(961):
        offset = 0.35 * min(j / 720, 1)
        out = feed(p, 120 + j, camera=Pose(CAMERA.pos + (0, offset, 0), CAMERA.rot))
    assert abs((feet(out) - out.head_pos)[1] - before) < 0.04


def test_seated_hover_does_not_replace_the_known_floor():
    p = Processor(Settings())
    planted(p)
    floor = p.floor_y
    for j in range(211):
        progress = min(1, j / 27)
        feed_both(p, 120 + j, left=(0, 0.08 * progress, 0), right=(0, 0.14 * progress, 0))
    assert p.floor_y is not None and abs(p.floor_y - floor) < 0.03
    assert not p.foot_locked


def test_tracking_gap_does_not_count_toward_continuous_floor_mismatch():
    p = Processor(Settings())
    planted(p)
    floor = p.floor_y
    for j in range(193):
        height = 0.13 * min(1, j / 18)
        if 42 <= j < 192:
            feed(p, 120 + j, masked=True)
        else:
            feed_both(p, 120 + j, left=(0, height, 0), right=(0, height, 0))
    assert p.floor_y == pytest.approx(floor, abs=0.01)


def test_toggling_foot_lock_without_an_intervening_packet_clears_the_floor():
    engine = Engine(Settings(auto_yaw_on_start=False), osc_port=43862, dry_run=True)
    addr = ('127.0.0.1', 42001)
    for i in range(120):
        engine.handle_packet(packet(i), addr)
    engine.update_settings(foot_lock=False)
    engine.update_settings(foot_lock=True)
    engine.handle_packet(packet(120), addr)
    snap = engine.snapshot()
    assert not snap.floor_ready and not snap.foot_locked


def test_masked_tracking_does_not_send_foot_tracker_after_its_checkbox_is_unchecked():
    addresses = []
    engine = Engine(Settings(auto_yaw_on_start=False), osc_port=43862, dry_run=True,
                    osc_recorder=lambda addr, _values: addresses.append(addr))
    source = ('127.0.0.1', 42001)
    for i in range(120):
        engine.handle_packet(packet(i), source)
    left = next(t.index for t in engine.snapshot().frame.trackers if t.name == 'left_foot')
    engine.update_settings(enabled_trackers=['hip', 'right_foot'])
    addresses.clear()
    engine.handle_packet(packet(120, masked=True), source)
    assert any(addr == '/tracking/trackers/head/position' for addr in addresses)
    assert not any(addr.startswith(f'/tracking/trackers/{left}/') for addr in addresses)
