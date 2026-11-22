import math
import random
import struct

import numpy as np

from arkit_fbt_relay.engine import Engine
from arkit_fbt_relay.mathutil import quat_from_axis_angle
from arkit_fbt_relay.pipeline import Settings
from arkit_fbt_relay.preview import (
    CHUNK, CHUNK_PAYLOAD, Reassembler, build_chunks, display_rotation, framing_warnings,
)
from arkit_fbt_relay.protocol import JOINT_NAMES, STATUS_WANT_PREVIEW

JPEG = bytes(range(256)) * 90  # 23KB 가짜 데이터
JOINTS = [(0.5, 0.1 + i * 0.04) for i in range(len(JOINT_NAMES))]


def test_chunk_layout_matches_swift():
    assert CHUNK.size == 16
    chunks = build_chunks(7, 384, 288, 1.5, JOINTS, JPEG)
    assert all(len(c) <= 16 + CHUNK_PAYLOAD for c in chunks)
    magic, ver, _r, idx, count, _r2, fid = CHUNK.unpack_from(chunks[0], 0)
    assert (magic, ver, idx, count, fid) == (b"AFBC", 1, 0, len(chunks), 7)
    # blob 헤더: w, h, n, reserved, timestamp
    assert struct.unpack_from("<HHHHd", chunks[0], 16) == (384, 288, len(JOINT_NAMES), 0, 1.5)


def test_reassemble_out_of_order():
    chunks = build_chunks(1, 384, 288, 2.0, JOINTS[:-1] + [None], JPEG)
    random.Random(0).shuffle(chunks)
    r = Reassembler()
    results = [r.add(c) for c in chunks]
    done = [p for p in results if p is not None]
    assert len(done) == 1
    p = done[0]
    assert p.jpeg == JPEG and (p.width, p.height) == (384, 288)
    assert p.joints[-1] is None
    assert np.allclose(p.joints[:-1], JOINTS[:-1], atol=1e-6)  # f32 정밀도


def test_old_frames_dropped_and_restart():
    r = Reassembler()
    for c in build_chunks(5, 10, 10, 0.0, [], b"x" * 3000):
        r.add(c)
    # 더 오래된 프레임은 무시
    assert all(r.add(c) is None for c in build_chunks(4, 10, 10, 0.0, [], b"y" * 10))
    # 손실 난 프레임이 쌓여도 메모리가 늘지 않는다
    for fid in range(6, 30):
        r.add(build_chunks(fid, 10, 10, 0.0, [], b"z" * 3000)[0])
    assert len(r._pending) <= Reassembler.MAX_PENDING + 1
    # 앱 재시작 (frame id 가 1부터 다시)
    r2 = Reassembler()
    for c in build_chunks(50_000, 10, 10, 0.0, [], b"a"):
        r2.add(c)
    assert [p for c in build_chunks(1, 10, 10, 0.0, [], b"b") if (p := r2.add(c))]


def test_display_rotation():
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    assert display_rotation(ident) == 0  # 가로(센서 기본 방향)
    # 폰을 세로로 세움 = 카메라가 광축(Z) 기준 ±90° 회전
    k1 = display_rotation(quat_from_axis_angle([0, 0, 1], math.pi / 2))
    k2 = display_rotation(quat_from_axis_angle([0, 0, 1], -math.pi / 2))
    assert {k1, k2} == {1, 3}
    assert display_rotation(quat_from_axis_angle([0, 0, 1], math.pi)) == 2
    # 약간 기울어져도 같은 결과
    assert display_rotation(quat_from_axis_angle([0, 0, 1], math.radians(10))) == 0


def test_framing_warnings():
    assert framing_warnings(JOINTS, JOINT_NAMES) == []
    bad = list(JOINTS)
    bad[JOINT_NAMES.index("left_foot_joint")] = (0.5, 1.05)
    bad[JOINT_NAMES.index("head_joint")] = None
    w = framing_warnings(bad, JOINT_NAMES)
    assert any("머리" in x for x in w) and any("왼발" in x for x in w)
    assert framing_warnings([], JOINT_NAMES)


def test_engine_collects_preview_only_when_wanted():
    eng = Engine(Settings(), listen_host="127.0.0.1", listen_port=0, dry_run=True)
    addr = ("127.0.0.1", 1)
    for c in build_chunks(1, 384, 288, 0.0, JOINTS, JPEG):
        eng.handle_packet(c, addr)
    assert eng._preview is None and not eng._status() & STATUS_WANT_PREVIEW
    eng.want_preview = True
    for c in build_chunks(2, 384, 288, 0.0, JOINTS, JPEG):
        eng.handle_packet(c, addr)
    assert eng._preview is not None and eng._preview.frame_id == 2
    assert eng._status() & STATUS_WANT_PREVIEW
    assert eng.snapshot().bad_packets == 0
