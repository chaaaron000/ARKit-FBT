"""iPhone <-> PC 패킷 형식. ios/ARKitFBT/WireProtocol.swift 와 반드시 같아야 한다."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"AFBT"
ACK_MAGIC = b"AFBA"
VERSION = 1
DEFAULT_PORT = 39570

JOINT_NAMES: tuple[str, ...] = (
    "hips_joint", "spine_4_joint", "spine_7_joint", "neck_1_joint", "head_joint",
    "left_shoulder_1_joint", "left_arm_joint", "left_forearm_joint", "left_hand_joint",
    "right_shoulder_1_joint", "right_arm_joint", "right_forearm_joint", "right_hand_joint",
    "left_upLeg_joint", "left_leg_joint", "left_foot_joint", "left_toes_joint",
    "right_upLeg_joint", "right_leg_joint", "right_foot_joint", "right_toes_joint",
)

# 미리보기에 그릴 뼈대 연결
BONES: tuple[tuple[str, str], ...] = (
    ("hips_joint", "spine_4_joint"), ("spine_4_joint", "spine_7_joint"),
    ("spine_7_joint", "neck_1_joint"), ("neck_1_joint", "head_joint"),
    ("spine_7_joint", "left_shoulder_1_joint"), ("left_shoulder_1_joint", "left_arm_joint"),
    ("left_arm_joint", "left_forearm_joint"), ("left_forearm_joint", "left_hand_joint"),
    ("spine_7_joint", "right_shoulder_1_joint"), ("right_shoulder_1_joint", "right_arm_joint"),
    ("right_arm_joint", "right_forearm_joint"), ("right_forearm_joint", "right_hand_joint"),
    ("hips_joint", "left_upLeg_joint"), ("left_upLeg_joint", "left_leg_joint"),
    ("left_leg_joint", "left_foot_joint"), ("left_foot_joint", "left_toes_joint"),
    ("hips_joint", "right_upLeg_joint"), ("right_upLeg_joint", "right_leg_joint"),
    ("right_leg_joint", "right_foot_joint"), ("right_foot_joint", "right_toes_joint"),
)

HEADER = struct.Struct("<4sBBHIdfI")  # 28 bytes
POSE = struct.Struct("<7f")  # 28 bytes
ACK = struct.Struct("<4sBBHId")  # 20 bytes

FLAG_TRACKED = 0x01

STATUS_OSC_SENDING = 0x01
STATUS_YAW_ALIGNED = 0x02
STATUS_DRY_RUN = 0x04


class ProtocolError(ValueError):
    pass


@dataclass(slots=True)
class Pose:
    pos: np.ndarray  # (3,) meters
    rot: np.ndarray  # (4,) quaternion x, y, z, w


@dataclass(slots=True)
class BodyFrame:
    seq: int
    timestamp: float  # iPhone 캡처 시각 (초)
    tracked: bool
    scale: float
    camera: Pose
    root: Pose | None
    joints: dict[str, Pose] = field(default_factory=dict)


def _read_pose(buf: bytes, offset: int) -> Pose | None:
    v = POSE.unpack_from(buf, offset)
    if any(math.isnan(x) or math.isinf(x) for x in v):
        return None
    return Pose(np.array(v[:3], dtype=float), np.array(v[3:], dtype=float))


def _write_pose(pose: Pose | None) -> bytes:
    if pose is None:
        return POSE.pack(*([math.nan] * 7))
    return POSE.pack(*[float(x) for x in pose.pos], *[float(x) for x in pose.rot])


def parse_frame(data: bytes) -> BodyFrame:
    min_size = HEADER.size + 2 * POSE.size
    if len(data) < min_size:
        raise ProtocolError(f"패킷이 너무 짧음 ({len(data)} bytes)")
    magic, version, flags, count, seq, ts, scale, mask = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ProtocolError("magic 불일치")
    if version != VERSION:
        raise ProtocolError(f"지원하지 않는 버전 {version} (필요: {VERSION})")
    offset = HEADER.size
    camera = _read_pose(data, offset)
    offset += POSE.size
    root = _read_pose(data, offset)
    offset += POSE.size
    if camera is None:
        raise ProtocolError("카메라 자세가 없음")
    if len(data) < offset + count * POSE.size:
        raise ProtocolError("관절 데이터가 잘림")

    joints: dict[str, Pose] = {}
    for i in range(min(count, len(JOINT_NAMES))):
        if not (mask >> i) & 1:
            continue
        pose = _read_pose(data, offset + i * POSE.size)
        if pose is not None:
            joints[JOINT_NAMES[i]] = pose

    return BodyFrame(
        seq=seq,
        timestamp=ts,
        tracked=bool(flags & FLAG_TRACKED),
        scale=scale,
        camera=camera,
        root=root,
        joints=joints,
    )


def build_frame(
    seq: int,
    timestamp: float,
    camera: Pose,
    joints: dict[str, Pose] | None,
    root: Pose | None = None,
    scale: float = 1.0,
) -> bytes:
    """테스트와 가짜 iPhone(tools/fake_phone.py)용 패킷 생성기."""
    tracked = bool(joints)
    count = len(JOINT_NAMES) if tracked else 0
    mask = 0
    body = []
    if tracked:
        for i, name in enumerate(JOINT_NAMES):
            pose = joints.get(name)
            if pose is not None:
                mask |= 1 << i
            body.append(_write_pose(pose))
        if root is None:
            root = joints.get("hips_joint")
    header = HEADER.pack(
        MAGIC, VERSION, FLAG_TRACKED if tracked else 0, count,
        seq & 0xFFFFFFFF, timestamp, scale, mask,
    )
    return header + _write_pose(camera) + _write_pose(root if tracked else None) + b"".join(body)


def build_ack(seq: int, timestamp: float, status: int) -> bytes:
    return ACK.pack(ACK_MAGIC, VERSION, status & 0xFF, 0, seq & 0xFFFFFFFF, timestamp)


def parse_ack(data: bytes) -> tuple[int, int, float]:
    """(status, seq, timestamp)"""
    if len(data) < ACK.size:
        raise ProtocolError("ack가 너무 짧음")
    magic, _version, status, _reserved, seq, ts = ACK.unpack_from(data, 0)
    if magic != ACK_MAGIC:
        raise ProtocolError("ack magic 불일치")
    return status, seq, ts
