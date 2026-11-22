"""가짜 iPhone -> Engine (UDP, localhost) -> OSC 기록까지 전체 흐름."""

import socket
import time

import numpy as np

from arkit_fbt_relay.engine import Engine
from arkit_fbt_relay.fake_phone import synth_joints
from arkit_fbt_relay.pipeline import Settings
from arkit_fbt_relay.protocol import Pose, build_frame, parse_ack

CAM = Pose(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))


def run_session(seconds=2.2, hands_up_from=None, settings=None):
    messages = []
    eng = Engine(settings or Settings(), listen_host="127.0.0.1", listen_port=0, dry_run=True,
                 osc_recorder=lambda a, v: messages.append((a, v)))
    eng.start()
    port = eng.bound_port
    phone = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    phone.settimeout(0.0)
    acks = []
    try:
        n = int(seconds * 60)
        t0 = time.monotonic()
        for i in range(n):
            t = i / 60
            hands = hands_up_from is not None and t >= hands_up_from
            phone.sendto(build_frame(i + 1, t, CAM, synth_joints(t, hands_up=hands)), ("127.0.0.1", port))
            try:
                while True:
                    acks.append(parse_ack(phone.recvfrom(64)[0]))
            except (BlockingIOError, OSError):
                pass
            time.sleep(max(0.0, t0 + (i + 1) / 60 - time.monotonic()))
        time.sleep(0.1)
        return eng, messages, acks
    finally:
        phone.close()
        eng.stop()


def test_end_to_end():
    eng, messages, acks = run_session()
    addrs = {a for a, _ in messages}
    assert "/tracking/trackers/1/position" in addrs
    assert "/tracking/trackers/2/rotation" in addrs
    assert "/tracking/trackers/3/position" in addrs
    assert "/tracking/trackers/head/position" in addrs
    assert "/tracking/trackers/4/position" not in addrs  # 기본은 3개

    # 자동 방향 정렬은 정확히 한 번만 (VRChat 1회 즉시 정렬)
    yaw_msgs = [v for a, v in messages if a == "/tracking/trackers/head/rotation"]
    assert len(yaw_msgs) == 1
    assert abs(abs(yaw_msgs[0][1]) - 180) < 5

    assert acks, "iPhone으로 응답이 와야 한다"
    snap = eng.snapshot()
    assert snap.lost_packets == 0 and snap.bad_packets == 0

    # Unity 좌표: 골반 높이 약 0.95m, 카메라 앞(ARKit -Z) -> Unity +Z
    hip = [v for a, v in messages if a == "/tracking/trackers/1/position"][-1]
    assert 0.8 < hip[1] < 1.1
    assert hip[2] > 2.0


def test_gesture_triggers_yaw():
    s = Settings(auto_yaw_on_start=False)
    _eng, messages, _ = run_session(seconds=2.6, hands_up_from=0.5, settings=s)
    yaw_msgs = [v for a, v in messages if a == "/tracking/trackers/head/rotation"]
    assert len(yaw_msgs) == 1
