import struct

import numpy as np
import pytest

from arkit_fbt_relay.fake_phone import synth_joints
from arkit_fbt_relay.protocol import (
    HEADER, JOINT_NAMES, POSE, Pose, ProtocolError, build_ack, build_frame, parse_ack, parse_frame,
)

CAM = Pose(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))


def test_layout_matches_swift():
    # ios/ARKitFBT/WireProtocol.swift 의 주석과 같은 크기
    assert HEADER.size == 28
    assert POSE.size == 28
    assert len(JOINT_NAMES) == 21 <= 32  # tracked mask는 u32


def test_roundtrip():
    joints = synth_joints(1.0)
    data = build_frame(42, 123.5, CAM, joints, scale=0.97)
    assert len(data) == 84 + 21 * 28
    f = parse_frame(data)
    assert f.seq == 42 and f.tracked and f.timestamp == 123.5
    assert f.scale == pytest.approx(0.97)
    assert set(f.joints) == set(JOINT_NAMES)
    for name, p in joints.items():
        assert np.allclose(f.joints[name].pos, p.pos, atol=1e-6)
        assert np.allclose(f.joints[name].rot, p.rot, atol=1e-6)


def test_not_tracked():
    f = parse_frame(build_frame(1, 0.0, CAM, None))
    assert not f.tracked and f.joints == {} and f.root is None


def test_missing_and_untracked_joints():
    joints = synth_joints(0.0)
    del joints["left_toes_joint"]  # 위치 자체가 없음 (NaN)
    f = parse_frame(build_frame(1, 0.0, CAM, joints, untracked={"head_joint"}))
    assert "left_toes_joint" not in f.joints and "right_toes_joint" in f.joints
    # 추적 안 됨 표시여도 추정 위치는 받되, tracked_joints 에서는 빠진다
    assert "head_joint" in f.joints and "head_joint" not in f.tracked_joints
    assert "hips_joint" in f.tracked_joints


def test_rejects_garbage():
    with pytest.raises(ProtocolError):
        parse_frame(b"hello")
    with pytest.raises(ProtocolError):
        parse_frame(b"XXXX" + bytes(100))
    data = bytearray(build_frame(1, 0.0, CAM, synth_joints(0.0)))
    with pytest.raises(ProtocolError):
        parse_frame(bytes(data[:-10]))  # 잘림


def test_ack_roundtrip_and_swift_offsets():
    data = build_ack(7, 99.25, 0x03)
    assert len(data) == 20
    assert parse_ack(data) == (0x03, 7, 99.25)
    # Swift: status = d[5], seq @8 (u32), timestamp @12 (f64)
    assert data[5] == 0x03
    assert struct.unpack_from("<I", data, 8)[0] == 7
    assert struct.unpack_from("<d", data, 12)[0] == 99.25
